// Copyright (c) Aptos Foundation
// SPDX-License-Identifier: Apache-2.0

use crate::{
    metrics::TIMER,
    state_store::{
        state::LedgerState,
        state_update_refs::{BatchedStateUpdateRefs, StateUpdateRefs},
    },
    utils::planned::{Plan, Planned},
    DbReader,
};
use anyhow::Result;
use aptos_crypto::{
    hash::{CryptoHash, CORRUPTION_SENTINEL},
    HashValue,
};
use aptos_metrics_core::TimerHelper;
use aptos_scratchpad::{ProofRead, SparseMerkleTree};
use aptos_types::{
    proof::SparseMerkleProofExt, state_store::state_value::StateValue, transaction::Version,
};
use aptos_vm::AptosVM;
use derive_more::Deref;
use itertools::Itertools;
use once_cell::sync::Lazy;
use once_map::OnceMap;
use rayon::prelude::*;
use std::sync::Arc;

/// The data structure through which the entire state at a given
/// version can be summarized to a concise digest (the root hash).
#[derive(Clone, Debug)]
pub struct StateSummary {
    /// The next version. If this is 0, the state is the "pre-genesis" empty state.
    next_version: Version,
    pub global_state_summary: SparseMerkleTree<StateValue>,
}

impl StateSummary {
    pub fn new_at_version(
        version: Option<Version>,
        global_state_summary: SparseMerkleTree<StateValue>,
    ) -> Self {
        Self {
            next_version: version.map_or(0, |v| v + 1),
            global_state_summary,
        }
    }

    pub fn new_empty() -> Self {
        Self {
            next_version: 0,
            global_state_summary: SparseMerkleTree::new_empty(),
        }
    }

    pub fn root_hash(&self) -> HashValue {
        self.global_state_summary.root_hash()
    }

    pub fn next_version(&self) -> Version {
        self.next_version
    }

    pub fn version(&self) -> Option<Version> {
        self.next_version.checked_sub(1)
    }

    pub fn is_descendant_of(&self, other: &Self) -> bool {
        self.global_state_summary
            .is_descendant_of(&other.global_state_summary)
    }

    pub fn update(
        &self,
        persisted: &StateProofFetcher,
        updates: &BatchedStateUpdateRefs,
    ) -> Result<Self> {
        let _timer = TIMER.timer_with(&["state_summary__update"]);

        assert_ne!(self.global_state_summary.root_hash(), *CORRUPTION_SENTINEL);

        // Persisted must be before or at my version.
        assert!(persisted.next_version() <= self.next_version());
        // Updates must start at exactly my version.
        assert_eq!(updates.first_version(), self.next_version());

        let smt_updates = updates
            .shards
            .par_iter() // clone hashes and sort items in parallel
            // TODO(aldenhu): smt per shard?
            .flat_map(|shard| {
                shard
                    .iter()
                    .sorted_by(|(k1, _u1), (k2, _u2)| {
                        k1.crypto_hash_ref().cmp(k2.crypto_hash_ref())
                    })
                    .map(|(k, u)| (CryptoHash::hash(*k), u.value))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // TODO(aldenhu): smt leaf not carry StateValue
        let smt = self
            .global_state_summary
            .freeze(&persisted.global_state_summary)
            .batch_update(smt_updates, persisted)?
            .unfreeze();

        Ok(Self {
            next_version: updates.next_version(),
            global_state_summary: smt,
        })
    }
}

/// At a given version, the summaries of the state and the last checkpoint state at or before the version.
#[derive(Clone, Debug, Deref)]
pub struct LedgerStateSummary {
    #[deref]
    latest: StateSummary,
    last_checkpoint: StateSummary,
}

impl LedgerStateSummary {
    pub fn new(last_checkpoint: StateSummary, latest: StateSummary) -> Self {
        assert!(last_checkpoint.next_version() <= latest.next_version());

        Self {
            last_checkpoint,
            latest,
        }
    }

    pub fn new_empty() -> Self {
        let state_summary = StateSummary::new_empty();
        Self::new(state_summary.clone(), state_summary)
    }

    pub fn next_version(&self) -> Version {
        self.latest.next_version()
    }

    pub fn assert_versions_match(&self, state: &LedgerState) {
        assert_eq!(self.next_version(), state.next_version());
        assert_eq!(
            self.last_checkpoint.next_version(),
            state.last_checkpoint().next_version()
        );
    }

    pub fn latest(&self) -> &StateSummary {
        &self.latest
    }

    pub fn last_checkpoint(&self) -> &StateSummary {
        &self.last_checkpoint
    }

    pub fn update(
        &self,
        persisted: &StateProofFetcher,
        updates: &StateUpdateRefs,
    ) -> Result<Self> {
        let _timer = TIMER.timer_with(&["ledger_state_summary__update"]);

        let last_checkpoint = if let Some(updates) = &updates.for_last_checkpoint {
            self.latest.update(persisted, updates)?
        } else {
            self.last_checkpoint.clone()
        };

        let base_of_latest = if updates.for_last_checkpoint.is_none() {
            self.latest()
        } else {
            &last_checkpoint
        };
        let latest = if let Some(updates) = &updates.for_latest {
            base_of_latest.update(persisted, updates)?
        } else {
            base_of_latest.clone()
        };

        Ok(Self::new(last_checkpoint, latest))
    }
}

pub static IO_POOL: Lazy<Arc<rayon::ThreadPool>> = Lazy::new(|| {
    Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(AptosVM::get_num_proof_reading_threads()) // More than 8 threads doesn't seem to help much
            .thread_name(|index| format!("proof-read-{}", index))
            .build()
            .unwrap(),
    )
});

#[derive(Deref)]
pub struct StateProofFetcher {
    #[deref]
    state_summary: StateSummary,
    db: Arc<dyn DbReader>,
    // with OnceMap one can get a reference to the proof without locking the whole map up and
    // prevent updating.
    memorized_proofs: OnceMap<HashValue, Box<Planned<SparseMerkleProofExt>>>,
}

impl StateProofFetcher {
    pub fn new_persisted(db: Arc<dyn DbReader>) -> Result<Self> {
        Ok(Self::new(db.get_persisted_state_summary()?, db))
    }

    pub fn new(state_summary: StateSummary, db: Arc<dyn DbReader>) -> Self {
        Self {
            state_summary,
            db,
            memorized_proofs: OnceMap::new(),
        }
    }

    fn root_hash(&self) -> HashValue {
        self.state_summary.root_hash()
    }

    pub fn get_proof_impl(
        db: Arc<dyn DbReader>,
        key_hash: HashValue,
        version: Version,
        root_depth: usize,
        root_hash: HashValue,
    ) -> Result<SparseMerkleProofExt> {
        if rand::random::<usize>() % 10000 == 0 {
            // 1 out of 10000 times, verify the proof.
            let (val_opt, proof) = db
                // verify the full proof
                .get_state_value_with_proof_by_version_ext(&key_hash, version, 0)?;
            proof.verify(root_hash, key_hash, val_opt.as_ref())?;
            Ok(proof)
        } else {
            Ok(db.get_state_proof_by_version_ext(&key_hash, version, root_depth)?)
        }
    }

    pub fn schedule_get_proof_once(
        &self,
        key_hash: HashValue,
        root_depth: usize,
    ) -> Option<&Planned<SparseMerkleProofExt>> {
        self.version().map(|ver| {
            self.memorized_proofs.insert(key_hash, |key_hash| {
                let key_hash = *key_hash;
                let db = self.db.clone();
                let root_hash = self.root_hash();

                Box::new(IO_POOL.plan(move || {
                    Self::get_proof_impl(db, key_hash, ver, root_depth, root_hash)
                        .expect("Failed getting state proof.")
                }))
            })
        })
    }
}

impl ProofRead for StateProofFetcher {
    fn get_proof(&self, key: HashValue, root_depth: usize) -> Option<&SparseMerkleProofExt> {
        self.schedule_get_proof_once(key, root_depth)
            .map(|planned| planned.wait(Some("state_proof_wait")))
    }
}
