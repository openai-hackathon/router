use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Opaque engine hashes are used only as identities/parent links. Lookups match
/// exact token blocks and follow those links; no Rust recreation of Python's
/// hash serialization is involved. V1 supports one full-attention GPU group,
/// text only, without LoRA, cache salts, multimodal or prompt embeddings.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Block {
    pub hash: String,
    pub parent: Option<String>,
    pub tokens: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum KvEvent {
    BlockStored { blocks: Vec<Block> },
    BlockRemoved { hashes: Vec<String> },
    AllBlocksCleared,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EventBatch {
    pub sequence: u64,
    pub events: Vec<KvEvent>,
}

#[derive(Clone, Debug, Default)]
pub struct KvIndex {
    blocks: HashMap<String, Block>,
    // Multiple identities may temporarily describe the same edge.
    edges: HashMap<(Option<String>, Vec<u32>), Vec<String>>,
}

impl KvIndex {
    pub fn from_blocks(
        blocks: Vec<Block>,
        block_size: usize,
        limit: usize,
    ) -> Result<Self, &'static str> {
        let mut index = Self::default();
        index.apply(&[KvEvent::BlockStored { blocks }], block_size, limit)?;
        Ok(index)
    }

    pub fn apply(
        &mut self,
        events: &[KvEvent],
        block_size: usize,
        limit: usize,
    ) -> Result<(), &'static str> {
        for event in events {
            match event {
                KvEvent::AllBlocksCleared => *self = Self::default(),
                KvEvent::BlockRemoved { hashes } => {
                    for hash in hashes {
                        self.remove(hash);
                    }
                }
                KvEvent::BlockStored { blocks } => {
                    for block in blocks {
                        if block_size == 0
                            || block.tokens.len() != block_size
                            || block.hash.is_empty()
                        {
                            return Err("invalid_block");
                        }
                        if let Some(old) = self.blocks.get(&block.hash) {
                            if old.parent != block.parent || old.tokens != block.tokens {
                                return Err("conflicting_block_identity");
                            }
                            continue;
                        }
                        if self.blocks.len() >= limit {
                            return Err("kv_index_capacity");
                        }
                        self.edges
                            .entry((block.parent.clone(), block.tokens.clone()))
                            .or_default()
                            .push(block.hash.clone());
                        self.blocks.insert(block.hash.clone(), block.clone());
                    }
                }
            }
        }
        Ok(())
    }

    fn remove(&mut self, hash: &str) {
        if let Some(block) = self.blocks.remove(hash) {
            let key = (block.parent, block.tokens);
            if let Some(hashes) = self.edges.get_mut(&key) {
                hashes.retain(|h| h != hash);
                if hashes.is_empty() {
                    self.edges.remove(&key);
                }
            }
        }
    }

    pub fn reusable_tokens(&self, tokens: &[u32], block_size: usize) -> usize {
        if block_size == 0 {
            return 0;
        }
        let mut parents = vec![None];
        let mut matched = 0;
        // vLLM must compute at least the final prompt token to obtain logits.
        for block in tokens[..tokens.len().saturating_sub(1)].chunks_exact(block_size) {
            let mut next = Vec::new();
            for parent in parents {
                if let Some(hashes) = self.edges.get(&(parent, block.to_vec())) {
                    next.extend(hashes.iter().cloned().map(Some));
                }
            }
            if next.is_empty() {
                break;
            }
            parents = next;
            matched += block_size;
        }
        matched
    }
}
