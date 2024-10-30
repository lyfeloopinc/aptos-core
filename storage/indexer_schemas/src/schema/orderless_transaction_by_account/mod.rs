// Copyright © Aptos Foundation
// Parts of the project are originally copyright © Meta Platforms, Inc.
// SPDX-License-Identifier: Apache-2.0

//! This module defines physical storage schema for a transaction index via which the version of a
//! transaction sent by `account_address` with `nonce` can be found. With the version one
//! can resort to `TransactionSchema` for the transaction content.
//!
//! ```text
//! |<-------key------->|<-value->|
//! | address | nonce   | txn_ver |
//! ```

use crate::{schema::ORDERLESS_TRANSACTION_BY_ACCOUNT_CF_NAME, utils::ensure_slice_len_eq};
use anyhow::Result;
use aptos_schemadb::{
    define_pub_schema,
    schema::{KeyCodec, ValueCodec},
};
use aptos_types::{account_address::AccountAddress, transaction::Version};
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use std::{convert::TryFrom, mem::size_of};

define_pub_schema!(
    OrderlessTransactionByAccountSchema,
    Key,
    Version,
    ORDERLESS_TRANSACTION_BY_ACCOUNT_CF_NAME
);

type Nonce = u64;
type Key = (AccountAddress, Nonce);

impl KeyCodec<OrderlessTransactionByAccountSchema> for Key {
    fn encode_key(&self) -> Result<Vec<u8>> {
        let (ref account_address, nonce) = *self;

        let mut encoded = account_address.to_vec();
        encoded.write_u64::<BigEndian>(nonce)?;

        Ok(encoded)
    }

    fn decode_key(data: &[u8]) -> Result<Self> {
        ensure_slice_len_eq(data, size_of::<Self>())?;

        let address = AccountAddress::try_from(&data[..AccountAddress::LENGTH])?;
        let nonce = (&data[AccountAddress::LENGTH..]).read_u64::<BigEndian>()?;

        Ok((address, nonce))
    }
}

impl ValueCodec<OrderlessTransactionByAccountSchema> for Version {
    fn encode_value(&self) -> Result<Vec<u8>> {
        Ok(self.to_be_bytes().to_vec())
    }

    fn decode_value(mut data: &[u8]) -> Result<Self> {
        ensure_slice_len_eq(data, size_of::<Self>())?;

        Ok(data.read_u64::<BigEndian>()?)
    }
}

#[cfg(test)]
mod test;
