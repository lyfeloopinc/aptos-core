// Copyright © Aptos Foundation
// SPDX-License-Identifier: Apache-2.0

use crate::{
    dag::anchor_election::AnchorElection,
    experimental::ordering_state_computer::OrderingStateComputer, state_replication::StateComputer,
};
use aptos_consensus_types::{
    block::Block,
    block_data::BlockData,
    common::{Payload, PayloadFilter},
    executed_block::ExecutedBlock,
    node::Node,
    quorum_cert::QuorumCert,
    vote::Vote,
    vote_data::VoteData,
};
use aptos_crypto::HashValue;
use aptos_executor_types::StateComputeResult;
use aptos_types::{
    aggregate_signature::AggregateSignature,
    block_info::BlockInfo,
    ledger_info::{LedgerInfo, LedgerInfoWithSignatures},
    validator_signer::ValidatorSigner,
    validator_verifier::ValidatorVerifier,
    PeerId,
};
use claims::assert_some;
use itertools::Itertools;
use std::{collections::HashMap, hash::Hash, iter::Extend, sync::Arc};

pub struct Bullshark {
    state_computer: Arc<dyn StateComputer>,
    dag: Vec<HashMap<PeerId, Node>>,
    lowest_unordered_anchor_wave: u64,
    proposer_election: Arc<dyn AnchorElection>,
    verifier: ValidatorVerifier,
    pending_payload: HashMap<HashValue, Payload>, // TODO: dont clone. Either deal with life time or use Arc<Payload> in Node and clone the Arc here.
}

impl Bullshark {
    pub fn new(
        state_computer: Arc<dyn StateComputer>,
        proposer_election: Arc<dyn AnchorElection>,
        verifier: ValidatorVerifier,
    ) -> Self {
        Self {
            state_computer,
            dag: Vec::new(),
            lowest_unordered_anchor_wave: 0,
            proposer_election,
            verifier,
            pending_payload: HashMap::new(),
        }
    }

    fn strong_path(&self, source: &Node, target: &Node) -> bool {
        let target_round = target.round();
        let mut round = source.round();

        let mut reachable_nodes = HashMap::new();
        reachable_nodes.insert(source.digest(), source);

        while round > target_round {
            let mut new_reachable_nodes = HashMap::new();
            reachable_nodes.iter().for_each(|(_, n)| {
                n.strong_links().iter().for_each(|peer_id| {
                    if let Some(node) = self.dag[round as usize - 1].get(&peer_id) {
                        new_reachable_nodes.insert(node.digest(), node);
                    }
                })
            });
            reachable_nodes = new_reachable_nodes;
            round -= 1;
        }

        reachable_nodes.keys().contains(&target.digest())
    }

    fn order_anchors(&mut self, anchor: Node) -> Vec<Node> {
        let mut anchor_stack = Vec::new();
        let mut round = anchor.round();
        assert_eq!(round % 2, 0);
        let mut wave = anchor.round() / 2;
        let new_ordered_wave = wave;

        wave -= 1;
        round -= 2;
        let mut current_anchor = anchor;

        while wave >= self.lowest_unordered_anchor_wave {
            let prev_anchor_peer_id = self.proposer_election.get_round_anchor_peer_id(wave);

            let is_path =
                if let Some(prev_anchor) = self.dag[round as usize].get(&prev_anchor_peer_id) {
                    self.strong_path(&current_anchor, prev_anchor)
                } else {
                    false
                };

            if is_path {
                anchor_stack.push(std::mem::replace(
                    &mut current_anchor,
                    self.dag[round as usize]
                        .remove(&prev_anchor_peer_id)
                        .unwrap(),
                ));
            }

            wave -= 1;
            round -= 2;
        }

        anchor_stack.push(current_anchor);
        self.lowest_unordered_anchor_wave = new_ordered_wave + 1;
        anchor_stack
    }

    fn order_history(&mut self, mut anchor_stack: Vec<Node>) -> Vec<Node> {
        let mut ordered_history = Vec::new();

        while let Some(anchor) = anchor_stack.pop() {
            ordered_history.extend(self.order_anchor_causal_history(anchor));
        }

        ordered_history
    }

    fn order_anchor_causal_history(&mut self, anchor: Node) -> Vec<Node> {
        let mut ordered_history = Vec::new();

        let mut reachable_nodes = HashMap::new();
        reachable_nodes.insert(anchor.digest(), anchor);

        while !reachable_nodes.is_empty() {
            let mut new_reachable_nodes = HashMap::new();
            reachable_nodes.into_iter().for_each(|(_, node)| {
                node.parents().iter().for_each(|metadata| {
                    if let Some(parent) =
                        self.dag[metadata.round() as usize].remove(&metadata.source())
                    {
                        new_reachable_nodes.insert(parent.digest(), parent);
                    }
                });
                self.pending_payload.remove(&node.digest());
                ordered_history.push(node);
            });
            reachable_nodes = new_reachable_nodes;
        }
        ordered_history
    }

    pub async fn try_ordering(&mut self, node: Node) {
        let round = node.round();
        let wave = round / 2;
        let author = node.source();

        assert!(!self
            .dag
            .get(round as usize)
            .map_or(false, |m| m.contains_key(&author)));

        if self.dag.len() < round as usize {
            assert_some!(self.dag.get(round as usize - 1));
            self.dag.push(HashMap::new());
        }

        self.pending_payload
            .insert(node.digest(), node.maybe_payload().unwrap().clone());
        self.dag[round as usize].insert(author, node);

        if round % 2 == 0 || wave < self.lowest_unordered_anchor_wave {
            return;
        }

        // We have one more potential vote in a wave we have not previously ordered
        let anchor_author = self.proposer_election.get_round_anchor_peer_id(wave);
        let voters = self
            .dag
            .get(round as usize)
            .unwrap()
            .iter()
            .filter(|(_, node)| node.strong_links().contains(&anchor_author))
            .map(|(peer_id, _)| peer_id);

        if self.verifier.check_minority_voting_power(voters).is_ok() {
            let anchor = self.dag[round as usize - 1].remove(&anchor_author).unwrap();
            let order_anchor_stack = self.order_anchors(anchor);
            let ordered_history = self.order_history(order_anchor_stack);
            self.push_to_execution(ordered_history).await;
        }
    }

    async fn push_to_execution(&self, ordered_history: Vec<Node>) {
        let blocks: Vec<Arc<ExecutedBlock>> = ordered_history
            .iter()
            .map(|node| {
                let block = ExecutedBlock::new(
                    Block::new_proposal(
                        node.maybe_payload().unwrap().clone(),
                        node.round(),
                        0,
                        QuorumCert::new(
                            VoteData::new(BlockInfo::empty(), BlockInfo::empty()),
                            LedgerInfoWithSignatures::new(
                                LedgerInfo::new(BlockInfo::empty(), HashValue::zero()),
                                AggregateSignature::empty(),
                            ),
                        ),
                        &ValidatorSigner::random(None),
                        Vec::new(),
                    )
                    .unwrap(),
                    StateComputeResult::new_dummy(),
                );
                Arc::new(block)
            })
            .collect();

        self.state_computer
            .commit(
                &blocks,
                LedgerInfoWithSignatures::new(
                    LedgerInfo::new(BlockInfo::empty(), HashValue::zero()),
                    AggregateSignature::empty(),
                ),
                Box::new(|_, _| {}),
            )
            .await
            .unwrap();
    }

    pub fn pending_payload(&self) -> PayloadFilter {
        let excluded_payload = self
            .pending_payload
            .iter()
            .map(|(_, payload)| payload)
            .collect();
        PayloadFilter::from(&excluded_payload)
    }

    // pub async fn start(self, mut rx: Receiver<CertifiedNode>) {
    //     loop {
    //         tokio::select! {
    //         Some(_) = rx.recv() => {
    //
    //         }
    //             }
    //     }
    // }
}
