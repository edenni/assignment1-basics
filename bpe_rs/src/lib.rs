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

// ---------------- Tokenizer (#[pyclass]) ----------------

#[pyclass]
struct Tokenizer {
    // id -> bytes (owned, in order of id)
    id_to_bytes: Vec<Vec<u8>>,
    // bytes -> id (inverse vocab, including special tokens)
    token_to_id: AHashMap<Vec<u8>, u32>,
    // ordered merges as byte pairs (matches Python self.merges semantics)
    merges_bytes: Vec<(Vec<u8>, Vec<u8>)>,
    // merge-rank map for heap-based encode: (left_id, right_id) -> rank
    merge_rank: AHashMap<(u32, u32), u32>,
    // special tokens (sorted by len desc, matches Python) as bytes
    special_tokens: Vec<Vec<u8>>,
    // AC automaton over special tokens for fast chunk splitting; None if no specials
    ac: Option<AhoCorasick>,
    // compiled pretokenization regex (one instance; single-threaded encode)
    re: Regex,
    // cache: pretoken bytes -> encoded ids. Uses Mutex to be Sync (parallel
    // encode_file does NOT touch this — each thread has its own cache).
    cache: std::sync::Mutex<AHashMap<Vec<u8>, Vec<u32>>>,
}

fn bytes_pair_order(a: &[u8], b: &[u8]) -> std::cmp::Ordering {
    a.cmp(b)
}

impl Tokenizer {
    fn bpe_sequential(&self, pretoken: &[u8]) -> Vec<u32> {
        // Walk through self.merges_bytes in order; for each merge, scan the
        // current symbol sequence left-to-right and fuse adjacent matches.
        // Matches the Python `for token_pair in self.merges` loop exactly.
        let mut symbols: Vec<Vec<u8>> =
            pretoken.iter().map(|&b| vec![b]).collect();
        if symbols.len() <= 1 {
            return symbols
                .into_iter()
                .map(|s| *self.token_to_id.get(&s).expect("byte must be in vocab"))
                .collect();
        }
        for (l, r) in &self.merges_bytes {
            if symbols.len() < 2 {
                break;
            }
            let mut new_syms: Vec<Vec<u8>> = Vec::with_capacity(symbols.len());
            let mut i = 0usize;
            while i < symbols.len() {
                if i + 1 < symbols.len() && symbols[i] == *l && symbols[i + 1] == *r {
                    let mut merged = Vec::with_capacity(l.len() + r.len());
                    merged.extend_from_slice(l);
                    merged.extend_from_slice(r);
                    new_syms.push(merged);
                    i += 2;
                } else {
                    new_syms.push(std::mem::take(&mut symbols[i]));
                    i += 1;
                }
            }
            symbols = new_syms;
        }
        symbols
            .into_iter()
            .map(|s| *self.token_to_id.get(&s).expect("symbol must be in vocab"))
            .collect()
    }

    fn bpe_heap(&self, pretoken: &[u8]) -> Vec<u32> {
        // Priority-queue BPE on id sequence: repeatedly fuse the adjacent pair
        // with the smallest merge rank. Result maps 1:1 to vocabulary ids.
        // Equivalent to the sequential version when merges are a total order
        // of valid BPE merges (which they are here, because that's how
        // train_bpe constructed them).
        let n = pretoken.len();
        if n == 0 {
            return Vec::new();
        }
        let mut ids: Vec<u32> = Vec::with_capacity(n);
        for &b in pretoken {
            let id = *self
                .token_to_id
                .get(&vec![b])
                .expect("byte must be in vocab");
            ids.push(id);
        }
        if n == 1 {
            return ids;
        }

        // Doubly-linked list over ids: prev/next arrays, plus `alive` flag.
        let mut prev: Vec<i32> = (0..n as i32).map(|i| i - 1).collect();
        let mut next: Vec<i32> = (0..n as i32).map(|i| if i + 1 == n as i32 { -1 } else { i + 1 }).collect();
        let mut alive: Vec<bool> = vec![true; n];
        // tokens[i] = current id at node i (mutated when a merge lands here)
        let mut tokens = ids.clone();

        #[derive(Eq, PartialEq)]
        struct Entry {
            rank: u32,
            pos: i32,          // position of LEFT element
            left_id: u32,
            right_id: u32,
        }
        impl Ord for Entry {
            fn cmp(&self, o: &Self) -> std::cmp::Ordering {
                // Min-heap on rank; tie-break by position for determinism
                o.rank.cmp(&self.rank).then_with(|| o.pos.cmp(&self.pos))
            }
        }
        impl PartialOrd for Entry {
            fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(o))
            }
        }

        let mut heap: BinaryHeap<Entry> = BinaryHeap::with_capacity(n);
        for i in 0..n - 1 {
            let pair = (tokens[i], tokens[i + 1]);
            if let Some(&rank) = self.merge_rank.get(&pair) {
                heap.push(Entry {
                    rank,
                    pos: i as i32,
                    left_id: pair.0,
                    right_id: pair.1,
                });
            }
        }

        while let Some(top) = heap.pop() {
            let pos = top.pos;
            if !alive[pos as usize] {
                continue;
            }
            let right = next[pos as usize];
            if right < 0 || !alive[right as usize] {
                continue;
            }
            // Validate: the pair at (pos, right) must still match the
            // heap entry (some earlier merge may have shifted one side).
            if tokens[pos as usize] != top.left_id || tokens[right as usize] != top.right_id {
                continue;
            }

            // Perform the merge. Build the new id by looking up the
            // concatenated bytes in the vocab.
            let merged_bytes: Vec<u8> = {
                let l = &self.id_to_bytes[top.left_id as usize];
                let r = &self.id_to_bytes[top.right_id as usize];
                let mut v = Vec::with_capacity(l.len() + r.len());
                v.extend_from_slice(l);
                v.extend_from_slice(r);
                v
            };
            let new_id = match self.token_to_id.get(&merged_bytes) {
                Some(&id) => id,
                None => {
                    // Shouldn't happen if merges came from this vocab,
                    // but fall back: skip this merge.
                    continue;
                }
            };

            tokens[pos as usize] = new_id;
            // Remove `right` from the linked list
            alive[right as usize] = false;
            let new_next = next[right as usize];
            next[pos as usize] = new_next;
            if new_next >= 0 {
                prev[new_next as usize] = pos;
            }

            // Push new neighbouring pairs
            let p = prev[pos as usize];
            if p >= 0 {
                let pair = (tokens[p as usize], tokens[pos as usize]);
                if let Some(&rank) = self.merge_rank.get(&pair) {
                    heap.push(Entry {
                        rank,
                        pos: p,
                        left_id: pair.0,
                        right_id: pair.1,
                    });
                }
            }
            if new_next >= 0 {
                let pair = (tokens[pos as usize], tokens[new_next as usize]);
                if let Some(&rank) = self.merge_rank.get(&pair) {
                    heap.push(Entry {
                        rank,
                        pos,
                        left_id: pair.0,
                        right_id: pair.1,
                    });
                }
            }
        }

        // Collect surviving nodes in order
        let mut out: Vec<u32> = Vec::new();
        // find head
        let mut head = 0i32;
        while head >= 0 && !alive[head as usize] {
            head += 1;
        }
        let mut cur = head;
        while cur >= 0 {
            if alive[cur as usize] {
                out.push(tokens[cur as usize]);
            }
            cur = next[cur as usize];
        }
        out
    }

    fn encode_chunk_with(
        &self,
        text: &str,
        min_heap: bool,
        re: &Regex,
        cache: &mut AHashMap<Vec<u8>, Vec<u32>>,
        out: &mut Vec<u32>,
    ) {
        // Fast path: if the whole chunk equals a special token, emit its id
        if !self.special_tokens.is_empty() {
            let tb = text.as_bytes();
            for st in &self.special_tokens {
                if tb == st.as_slice() {
                    if let Some(&id) = self.token_to_id.get(st) {
                        out.push(id);
                        return;
                    }
                }
            }
        }

        for m in re.find_iter(text).flatten() {
            let piece = &text.as_bytes()[m.start()..m.end()];
            if let Some(ids) = cache.get(piece) {
                out.extend_from_slice(ids);
                continue;
            }
            let ids = if min_heap {
                self.bpe_heap(piece)
            } else {
                self.bpe_sequential(piece)
            };
            out.extend_from_slice(&ids);
            cache.insert(piece.to_vec(), ids);
        }
    }

    fn encode_text_with(
        &self,
        text: &str,
        min_heap: bool,
        re: &Regex,
        cache: &mut AHashMap<Vec<u8>, Vec<u32>>,
        out: &mut Vec<u32>,
    ) {
        match &self.ac {
            None => self.encode_chunk_with(text, min_heap, re, cache, out),
            Some(ac) => {
                let bytes = text.as_bytes();
                let mut prev = 0usize;
                for m in ac.find_iter(bytes) {
                    if m.start() > prev {
                        self.encode_chunk_with(&text[prev..m.start()], min_heap, re, cache, out);
                    }
                    self.encode_chunk_with(&text[m.start()..m.end()], min_heap, re, cache, out);
                    prev = m.end();
                }
                if prev < bytes.len() {
                    self.encode_chunk_with(&text[prev..], min_heap, re, cache, out);
                }
            }
        }
    }
}

#[pymethods]
impl Tokenizer {
    #[new]
    #[pyo3(signature = (vocab, merges, special_tokens=None))]
    fn new(
        vocab: &Bound<'_, PyDict>,
        merges: Vec<(Vec<u8>, Vec<u8>)>,
        special_tokens: Option<Vec<String>>,
    ) -> PyResult<Self> {
        // Build id_to_bytes from Python dict[int, bytes]. Vocab ids are
        // dense in practice (0..len), but handle sparse cases too.
        let mut max_id: i64 = -1;
        for (k, _v) in vocab.iter() {
            let id: i64 = k.extract()?;
            if id > max_id {
                max_id = id;
            }
        }
        let size = (max_id + 1).max(0) as usize;
        let mut id_to_bytes: Vec<Vec<u8>> = vec![Vec::new(); size];
        for (k, v) in vocab.iter() {
            let id: usize = k.extract()?;
            let bytes: Vec<u8> = v.extract()?;
            id_to_bytes[id] = bytes;
        }
        let mut token_to_id: AHashMap<Vec<u8>, u32> = AHashMap::with_capacity(size);
        for (id, b) in id_to_bytes.iter().enumerate() {
            if !b.is_empty() || id < 256 {
                token_to_id.insert(b.clone(), id as u32);
            }
        }

        // Append special tokens not already in vocab (match Python behaviour)
        let specials: Vec<Vec<u8>> = match special_tokens {
            Some(ref list) => {
                let mut v: Vec<Vec<u8>> = list.iter().map(|s| s.as_bytes().to_vec()).collect();
                v.sort_by(|a, b| b.len().cmp(&a.len())); // desc by length
                for st in &v {
                    if !token_to_id.contains_key(st) {
                        let id = id_to_bytes.len() as u32;
                        id_to_bytes.push(st.clone());
                        token_to_id.insert(st.clone(), id);
                    }
                }
                v
            }
            None => Vec::new(),
        };

        // Build merge_rank for heap-based encode
        let mut merge_rank: AHashMap<(u32, u32), u32> = AHashMap::with_capacity(merges.len());
        for (rank, (l, r)) in merges.iter().enumerate() {
            if let (Some(&li), Some(&ri)) = (token_to_id.get(l), token_to_id.get(r)) {
                merge_rank.insert((li, ri), rank as u32);
            }
        }

        // AC automaton over specials (for encode chunk splitting)
        let ac = if specials.is_empty() {
            None
        } else {
            Some(
                AhoCorasick::new(specials.iter().map(|s| s.as_slice()))
                    .map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!("AC build: {}", e))
                    })?,
            )
        };

        let re = Regex::new(PAT)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("regex: {}", e)))?;

        let _ = bytes_pair_order; // silence unused warning if feature drift

        Ok(Self {
            id_to_bytes,
            token_to_id,
            merges_bytes: merges,
            merge_rank,
            special_tokens: specials,
            ac,
            re,
            cache: std::sync::Mutex::new(AHashMap::new()),
        })
    }

    #[pyo3(signature = (text, min_heap=false))]
    fn encode(&self, text: &str, min_heap: bool) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        let mut cache = self.cache.lock().expect("cache mutex poisoned");
        self.encode_text_with(text, min_heap, &self.re, &mut cache, &mut out);
        out
    }

    /// Encode an entire file in parallel, returning a numpy uint16 array.
    ///
    /// Steps:
    /// 1. mmap `input_path`.
    /// 2. Split into chunks on `<|endoftext|>` (or first special token)
    ///    boundaries via AhoCorasick — boundaries never fall inside a
    ///    pretoken, so per-chunk encoding gives the same ids as whole-file.
    /// 3. Parallel encode each chunk via rayon with per-thread `Regex`
    ///    and per-thread pretoken cache.
    /// 4. Concatenate into a single `Vec<u16>` (downcast from u32).
    #[pyo3(signature = (input_path, min_heap=false, num_threads=0))]
    fn encode_file<'py>(
        &self,
        py: Python<'py>,
        input_path: &str,
        min_heap: bool,
        num_threads: usize,
    ) -> PyResult<Bound<'py, numpy::PyArray1<u16>>> {
        use numpy::IntoPyArray;

        let file = File::open(input_path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("open {}: {}", input_path, e))
        })?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("mmap: {}", e)))?;
        let data: &[u8] = &mmap;

        // If vocab size > 65535, uint16 is insufficient
        if self.id_to_bytes.len() > u16::MAX as usize + 1 {
            return Err(pyo3::exceptions::PyOverflowError::new_err(
                "vocab size exceeds u16::MAX; use a wider dtype",
            ));
        }

        let threads = if num_threads == 0 {
            rayon::current_num_threads()
        } else {
            num_threads
        };
        // Choose chunk boundaries on special-token occurrences so each chunk
        // is a whole multiple of documents. One chunk per document would be
        // too many (owt = millions of docs), so we merge consecutive docs
        // until each chunk is ~file_size / (threads * 64) bytes — plenty for
        // work-stealing without excessive overhead.
        let split_token: &[u8] = if self.special_tokens.is_empty() {
            b"<|endoftext|>"
        } else {
            &self.special_tokens[0]
        };
        let target_chunks = (threads * 64).max(1);
        let target_bytes = (data.len() / target_chunks).max(1 << 20); // at least 1MB

        let mut ranges: Vec<(usize, usize)> = Vec::new();
        let mut start = 0usize;
        let finder = memmem::Finder::new(split_token);
        let mut cursor = 0usize;
        while cursor < data.len() {
            let desired_end = (start + target_bytes).min(data.len());
            if desired_end >= data.len() {
                ranges.push((start, data.len()));
                break;
            }
            match finder.find(&data[desired_end..]) {
                Some(off) => {
                    // end chunk just after the special token so boundary
                    // lands at a safe split point
                    let end = desired_end + off + split_token.len();
                    ranges.push((start, end));
                    start = end;
                    cursor = end;
                }
                None => {
                    ranges.push((start, data.len()));
                    break;
                }
            }
        }
        if ranges.is_empty() {
            ranges.push((0, data.len()));
        }

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(if num_threads == 0 { 0 } else { threads })
            .build()
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("rayon pool: {}", e))
            })?;

        let per_chunk: Vec<Vec<u16>> = py.allow_threads(|| {
            pool.install(|| {
                ranges
                    .par_iter()
                    .map_init(
                        || {
                            (
                                Regex::new(PAT).expect("regex compile"),
                                AHashMap::<Vec<u8>, Vec<u32>>::new(),
                            )
                        },
                        |(re, cache), &(s, e)| {
                            let slice = &data[s..e];
                            let text = std::str::from_utf8(slice)
                                .map(std::borrow::Cow::Borrowed)
                                .unwrap_or_else(|_| String::from_utf8_lossy(slice));
                            let mut ids: Vec<u32> = Vec::new();
                            self.encode_text_with(&text, min_heap, re, cache, &mut ids);
                            ids.into_iter().map(|x| x as u16).collect()
                        },
                    )
                    .collect()
            })
        });

        let total: usize = per_chunk.iter().map(|v| v.len()).sum();
        let mut flat: Vec<u16> = Vec::with_capacity(total);
        for v in per_chunk {
            flat.extend_from_slice(&v);
        }
        Ok(flat.into_pyarray_bound(py))
    }

    fn decode(&self, py: Python<'_>, ids: Vec<u32>) -> PyResult<Py<pyo3::types::PyString>> {
        let mut buf: Vec<u8> = Vec::new();
        for id in ids {
            let idx = id as usize;
            if idx >= self.id_to_bytes.len() {
                return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                    "id {} out of range",
                    id
                )));
            }
            buf.extend_from_slice(&self.id_to_bytes[idx]);
        }
        // decode with replacement (matches Python errors="replace")
        let s = String::from_utf8_lossy(&buf).into_owned();
        Ok(pyo3::types::PyString::new_bound(py, &s).into())
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.id_to_bytes.len()
    }

    /// Return a fresh dict[int, bytes] mirroring the vocab.
    #[getter]
    fn vocab<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        for (id, b) in self.id_to_bytes.iter().enumerate() {
            d.set_item(id, PyBytes::new_bound(py, b))?;
        }
        Ok(d)
    }

    /// Return the ordered list of merges as list[tuple[bytes, bytes]].
    #[getter]
    fn merges<'py>(&self, py: Python<'py>) -> Vec<(Py<PyBytes>, Py<PyBytes>)> {
        self.merges_bytes
            .iter()
            .map(|(a, b)| {
                (
                    PyBytes::new_bound(py, a).into(),
                    PyBytes::new_bound(py, b).into(),
                )
            })
            .collect()
    }
}

#[pymodule]
fn bpe_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pretokenize_file, m)?)?;
    m.add_function(wrap_pyfunction!(train_merges, m)?)?;
    m.add_class::<Tokenizer>()?;
    Ok(())
}
