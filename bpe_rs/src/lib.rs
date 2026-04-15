use ahash::{AHashMap, AHashSet};
use aho_corasick::AhoCorasick;
use compact_str::CompactString;
use fancy_regex::Regex;
use memchr::memmem;
use memmap2::Mmap;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::fs::File;

const PAT: &str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

// ---------------- pretokenization ----------------

fn find_chunk_boundaries(data: &[u8], desired: usize, split_token: &[u8]) -> Vec<usize> {
    let file_size = data.len();
    if desired <= 1 || file_size == 0 {
        return vec![0, file_size];
    }
    let chunk_size = file_size / desired;
    let mut boundaries: Vec<usize> = (0..=desired).map(|i| i * chunk_size).collect();
    *boundaries.last_mut().unwrap() = file_size;

    for bi in 1..boundaries.len() - 1 {
        let start = boundaries[bi];
        if start >= file_size {
            boundaries[bi] = file_size;
            continue;
        }
        // scan forward for the split token
        match find_subslice(&data[start..], split_token) {
            Some(off) => boundaries[bi] = start + off,
            None => boundaries[bi] = file_size,
        }
    }

    boundaries.sort_unstable();
    boundaries.dedup();
    boundaries
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > hay.len() {
        return None;
    }
    memmem::find(hay, needle)
}

/// Iterate byte-ranges between occurrences of any separator in `seps`.
/// Yields &str slices pointing into `text`. Empty slices are skipped.
fn split_on_any<'a>(text: &'a str, ac: Option<&AhoCorasick>) -> Vec<&'a str> {
    let Some(ac) = ac else {
        return if text.is_empty() { vec![] } else { vec![text] };
    };
    let bytes = text.as_bytes();
    let mut out: Vec<&str> = Vec::new();
    let mut prev = 0usize;
    for m in ac.find_iter(bytes) {
        if m.start() > prev {
            out.push(&text[prev..m.start()]);
        }
        prev = m.end();
    }
    if prev < bytes.len() {
        out.push(&text[prev..]);
    }
    out
}

fn count_chunk(
    chunk: &str,
    ac: Option<&AhoCorasick>,
    re: &Regex,
    counts: &mut AHashMap<CompactString, u64>,
) {
    for sub in split_on_any(chunk, ac) {
        for m in re.find_iter(sub).flatten() {
            let piece = &sub[m.start()..m.end()];
            if let Some(v) = counts.get_mut(piece) {
                *v += 1;
            } else {
                counts.insert(CompactString::from(piece), 1);
            }
        }
    }
}

#[pyfunction]
#[pyo3(signature = (input_path, special_tokens, num_threads=0))]
fn pretokenize_file(
    py: Python<'_>,
    input_path: &str,
    special_tokens: Vec<String>,
    num_threads: usize,
) -> PyResult<Py<PyDict>> {
    let file = File::open(input_path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("open {}: {}", input_path, e))
    })?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("mmap: {}", e))
    })?;
    let data: &[u8] = &mmap;

    let threads = if num_threads == 0 {
        rayon::current_num_threads()
    } else {
        num_threads
    };
    // More chunks than threads improves work-stealing and keeps each
    // chunk's working set closer to cache size for big files.
    let desired = (threads * 16).max(1);
    let split_token: &[u8] = if special_tokens.is_empty() {
        b"<|endoftext|>"
    } else {
        special_tokens[0].as_bytes()
    };
    let boundaries = find_chunk_boundaries(data, desired, split_token);

    let ac = if special_tokens.is_empty() {
        None
    } else {
        Some(
            AhoCorasick::new(special_tokens.iter().map(|s| s.as_bytes()))
                .expect("aho-corasick build"),
        )
    };

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(if num_threads == 0 { 0 } else { threads })
        .build()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("rayon pool: {}", e)))?;

    let merged = py.allow_threads(|| {
        let ranges: Vec<(usize, usize)> = boundaries
            .windows(2)
            .map(|w| (w[0], w[1]))
            .collect();
        pool.install(|| ranges
            .par_iter()
            .map_init(
                || Regex::new(PAT).expect("regex compile"),
                |re, &(s, e)| {
                    let slice = &data[s..e];
                    let text = std::str::from_utf8(slice)
                        .map(std::borrow::Cow::Borrowed)
                        .unwrap_or_else(|_| String::from_utf8_lossy(slice));
                    let mut local: AHashMap<CompactString, u64> = AHashMap::new();
                    count_chunk(&text, ac.as_ref(), re, &mut local);
                    local
                },
            )
            .reduce(
                AHashMap::<CompactString, u64>::new,
                |mut a, mut b| {
                    // Always merge smaller into larger to minimize work.
                    if a.len() < b.len() {
                        std::mem::swap(&mut a, &mut b);
                    }
                    for (k, v) in b {
                        *a.entry(k).or_insert(0) += v;
                    }
                    a
                },
            ))
    });


    let dict = PyDict::new_bound(py);
    for (k, v) in merged {
        let key = PyBytes::new_bound(py, k.as_bytes());
        dict.set_item(key, v)?;
    }
    Ok(dict.into())
}

// ---------------- train_merges ----------------

#[derive(Clone)]
struct MergeJob {
    count: i64,
    pair: (u32, u32),
    // concatenated bytes of (left, right) for tie-break comparison.
    // We store left and right bytes separately to match Python's tuple comparison.
    left_bytes: Vec<u8>,
    right_bytes: Vec<u8>,
}

impl PartialEq for MergeJob {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count
            && self.left_bytes == other.left_bytes
            && self.right_bytes == other.right_bytes
    }
}
impl Eq for MergeJob {}
impl PartialOrd for MergeJob {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MergeJob {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: higher count wins; on tie, lexicographically greater pair wins
        // (matches Python's `max(ties)` on tuple[bytes, bytes]).
        self.count
            .cmp(&other.count)
            .then_with(|| self.left_bytes.cmp(&other.left_bytes))
            .then_with(|| self.right_bytes.cmp(&other.right_bytes))
    }
}

struct Trainer {
    id_to_bytes: Vec<Vec<u8>>,
    words: Vec<Vec<u32>>,
    word_counts: Vec<u64>,
    pair_counts: AHashMap<(u32, u32), i64>,
    where_to_update: AHashMap<(u32, u32), AHashSet<u32>>,
    heap: BinaryHeap<MergeJob>,
}

impl Trainer {
    fn new(input_words: Vec<(Vec<Vec<u8>>, u64)>) -> Self {
        let mut interner: AHashMap<Vec<u8>, u32> = AHashMap::new();
        let mut id_to_bytes: Vec<Vec<u8>> = Vec::new();
        let mut words: Vec<Vec<u32>> = Vec::with_capacity(input_words.len());
        let mut word_counts: Vec<u64> = Vec::with_capacity(input_words.len());

        for (symbols, count) in input_words {
            let mut ids: Vec<u32> = Vec::with_capacity(symbols.len());
            for sym in symbols {
                let id = match interner.get(&sym) {
                    Some(&i) => i,
                    None => {
                        let i = id_to_bytes.len() as u32;
                        interner.insert(sym.clone(), i);
                        id_to_bytes.push(sym);
                        i
                    }
                };
                ids.push(id);
            }
            words.push(ids);
            word_counts.push(count);
        }

        let mut pair_counts: AHashMap<(u32, u32), i64> = AHashMap::new();
        let mut where_to_update: AHashMap<(u32, u32), AHashSet<u32>> = AHashMap::new();
        for (wi, w) in words.iter().enumerate() {
            let c = word_counts[wi] as i64;
            for i in 0..w.len().saturating_sub(1) {
                let p = (w[i], w[i + 1]);
                *pair_counts.entry(p).or_insert(0) += c;
                where_to_update
                    .entry(p)
                    .or_insert_with(AHashSet::new)
                    .insert(wi as u32);
            }
        }

        let mut heap: BinaryHeap<MergeJob> = BinaryHeap::with_capacity(pair_counts.len());
        for (&pair, &count) in &pair_counts {
            heap.push(MergeJob {
                count,
                pair,
                left_bytes: id_to_bytes[pair.0 as usize].clone(),
                right_bytes: id_to_bytes[pair.1 as usize].clone(),
            });
        }

        Self {
            id_to_bytes,
            words,
            word_counts,
            pair_counts,
            where_to_update,
            heap,
        }
    }

    fn push_pair(&mut self, pair: (u32, u32)) {
        let count = *self.pair_counts.get(&pair).unwrap_or(&0);
        if count <= 0 {
            return;
        }
        self.heap.push(MergeJob {
            count,
            pair,
            left_bytes: self.id_to_bytes[pair.0 as usize].clone(),
            right_bytes: self.id_to_bytes[pair.1 as usize].clone(),
        });
    }

    fn train(&mut self, num_merges: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut out: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(num_merges);

        for _ in 0..num_merges {
            // Pop with lazy validation
            let top = loop {
                let Some(job) = self.heap.pop() else {
                    return out;
                };
                let current = *self.pair_counts.get(&job.pair).unwrap_or(&0);
                if current <= 0 {
                    continue;
                }
                if job.count != current {
                    // stale; re-push with fresh count
                    self.heap.push(MergeJob {
                        count: current,
                        pair: job.pair,
                        left_bytes: job.left_bytes,
                        right_bytes: job.right_bytes,
                    });
                    continue;
                }
                break job;
            };

            let pair = top.pair;
            let left_bytes = &self.id_to_bytes[pair.0 as usize];
            let right_bytes = &self.id_to_bytes[pair.1 as usize];
            let mut merged_bytes = Vec::with_capacity(left_bytes.len() + right_bytes.len());
            merged_bytes.extend_from_slice(left_bytes);
            merged_bytes.extend_from_slice(right_bytes);
            out.push((left_bytes.clone(), right_bytes.clone()));

            let new_id = self.id_to_bytes.len() as u32;
            self.id_to_bytes.push(merged_bytes);

            // Affected word indices
            let word_ids: Vec<u32> = match self.where_to_update.remove(&pair) {
                Some(s) => s.into_iter().collect(),
                None => Vec::new(),
            };

            let mut touched_pairs: AHashSet<(u32, u32)> = AHashSet::new();

            for wi in word_ids {
                let wi_us = wi as usize;
                let count = self.word_counts[wi_us] as i64;

                let old = std::mem::take(&mut self.words[wi_us]);
                let mut new_w: Vec<u32> = Vec::with_capacity(old.len());
                let mut i = 0;
                while i < old.len() {
                    if i + 1 < old.len() && old[i] == pair.0 && old[i + 1] == pair.1 {
                        new_w.push(new_id);
                        i += 2;
                    } else {
                        new_w.push(old[i]);
                        i += 1;
                    }
                }

                // Decrement old pair counts
                for k in 0..old.len().saturating_sub(1) {
                    let p = (old[k], old[k + 1]);
                    if let Some(c) = self.pair_counts.get_mut(&p) {
                        *c -= count;
                    }
                    touched_pairs.insert(p);
                }
                // Increment new pair counts
                for k in 0..new_w.len().saturating_sub(1) {
                    let p = (new_w[k], new_w[k + 1]);
                    *self.pair_counts.entry(p).or_insert(0) += count;
                    self.where_to_update
                        .entry(p)
                        .or_insert_with(AHashSet::new)
                        .insert(wi);
                    touched_pairs.insert(p);
                }

                self.words[wi_us] = new_w;
            }

            // Remove the merged pair's entry
            self.pair_counts.remove(&pair);
            touched_pairs.remove(&pair);

            // Push all touched pairs (with fresh counts) back onto the heap
            for p in touched_pairs {
                self.push_pair(p);
            }
        }

        out
    }
}

#[pyfunction]
fn train_merges(
    py: Python<'_>,
    words: Vec<(Vec<Vec<u8>>, u64)>,
    num_merges: usize,
) -> PyResult<Vec<(Py<PyBytes>, Py<PyBytes>)>> {
    let merges = py.allow_threads(|| {
        let mut t = Trainer::new(words);
        t.train(num_merges)
    });
    let mut out: Vec<(Py<PyBytes>, Py<PyBytes>)> = Vec::with_capacity(merges.len());
    for (a, b) in merges {
        out.push((
            PyBytes::new_bound(py, &a).into(),
            PyBytes::new_bound(py, &b).into(),
        ));
    }
    Ok(out)
}

#[pymodule]
fn bpe_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pretokenize_file, m)?)?;
    m.add_function(wrap_pyfunction!(train_merges, m)?)?;
    Ok(())
}
