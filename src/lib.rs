use bit_set::BitSet;
use ordered_float::OrderedFloat;
use pyo3::prelude::*;
use rand::Rng;
use rustc_hash::FxHashMap;
use std::collections::{BTreeSet, BinaryHeap};
use std::f32;

use FxHashMap as Dict;

type Ix = u16;
type Count = u8;
type Legs = Vec<(Ix, Count)>;
type Node = u16;
type Score = f32;
type SSAPath = Vec<Vec<Node>>;
type GreedyScore = OrderedFloat<Score>;

// types for optimal optimization
type ONode = u32;
type Subgraph = BitSet<ONode>;
type BitPath = Vec<(Subgraph, Subgraph)>;
type SubContraction = (Legs, Score, BitPath);

/// helper struct to build contractions from bottom up
struct ContractionProcessor {
    nodes: Dict<Node, Legs>,
    edges: Dict<Ix, Vec<Node>>,
    appearances: Vec<Count>,
    sizes: Vec<Score>,
    ssa: Node,
    ssa_path: SSAPath,
}

/// given log(x) and log(y) compute log(x + y), without exponentiating both
fn logadd(lx: Score, ly: Score) -> Score {
    let max_val = lx.max(ly);
    max_val + f32::ln_1p(f32::exp(-f32::abs(lx - ly)))
}

/// given log(x) and log(y) compute log(x - y), without exponentiating both,
/// if (x - y) is negative, return -log(x - y).
fn logsub(lx: f32, ly: f32) -> f32 {
    if lx < ly {
        -ly - f32::ln_1p(-f32::exp(lx - ly))
    } else {
        lx + f32::ln_1p(-f32::exp(ly - lx))
    }
}

fn compute_legs(ilegs: &Legs, jlegs: &Legs, appearances: &Vec<Count>) -> Legs {
    let mut ip = 0;
    let mut jp = 0;
    let ni = ilegs.len();
    let nj = jlegs.len();
    let mut new_legs: Legs = Vec::with_capacity(ilegs.len() + jlegs.len());

    loop {
        if ip == ni {
            new_legs.extend(jlegs[jp..].iter());
            break;
        }
        if jp == nj {
            new_legs.extend(ilegs[ip..].iter());
            break;
        }

        let (ix, ic) = ilegs[ip];
        let (jx, jc) = jlegs[jp];

        if ix < jx {
            // index only appears in ilegs
            new_legs.push((ix, ic));
            ip += 1;
        } else if ix > jx {
            // index only appears in jlegs
            new_legs.push((jx, jc));
            jp += 1;
        } else {
            // index appears in both
            let new_count = ic + jc;
            if new_count != appearances[ix as usize] {
                // not last appearance -> kept index contributes to new size
                new_legs.push((ix, new_count));
            }
            ip += 1;
            jp += 1;
        }
    }
    new_legs
}

fn compute_size(legs: &Legs, sizes: &Vec<Score>) -> Score {
    // legs.iter().map(|&(ix, _)| sizes[ix as usize]).product()
    legs.iter().map(|&(ix, _)| sizes[ix as usize]).sum()
}

fn is_simplifiable(legs: &Legs, appearances: &Vec<Count>) -> bool {
    let mut prev_ix = Node::MAX;
    for &(ix, ix_count) in legs {
        if (ix == prev_ix) || (ix_count == appearances[ix as usize]) {
            return true;
        }
        prev_ix = ix;
    }
    false
}

fn compute_simplified(legs: &Legs, appearances: &Vec<Count>) -> Legs {
    if legs.len() == 0 {
        return legs.clone();
    }
    let mut new_legs: Legs = Vec::with_capacity(legs.len());

    let (mut cur_ix, mut cur_cnt) = legs[0];
    for &(ix, ix_cnt) in legs.iter().skip(1) {
        if ix == cur_ix {
            cur_cnt += 1;
        } else {
            if cur_cnt != appearances[cur_ix as usize] {
                new_legs.push((cur_ix, cur_cnt));
            }
            cur_ix = ix;
            cur_cnt = ix_cnt;
        }
    }
    new_legs
}

impl ContractionProcessor {
    fn new(
        inputs: Vec<Vec<char>>,
        output: Vec<char>,
        size_dict: Dict<char, u32>,
    ) -> ContractionProcessor {
        let mut nodes: Dict<Node, Legs> = Dict::default();
        let mut edges: Dict<Ix, Vec<Node>> = Dict::default();
        let mut indmap: Dict<char, Ix> = Dict::default();
        let mut sizes: Vec<Score> = Vec::with_capacity(size_dict.len());
        let mut appearances: Vec<Count> = Vec::with_capacity(size_dict.len());
        // enumerate index labels as unsigned integers from 0
        let mut c: Ix = 0;

        for (i, term) in inputs.into_iter().enumerate() {
            let mut legs = Vec::with_capacity(term.len());
            for ind in term {
                match indmap.get(&ind) {
                    None => {
                        // index not parsed yet
                        indmap.insert(ind, c);
                        edges.insert(c, vec![i as Node]);
                        appearances.push(1);
                        sizes.push(f32::log(size_dict[&ind] as f32, 2.0));
                        legs.push((c, 1));
                        c += 1;
                    }
                    Some(&ix) => {
                        // index already present
                        appearances[ix as usize] += 1;
                        edges.get_mut(&ix).unwrap().push(i as Node);
                        legs.push((ix, 1));
                    }
                };
            }
            legs.sort();
            nodes.insert(i as Node, legs);
        }
        output.into_iter().for_each(|ind| {
            appearances[indmap[&ind] as usize] += 1;
        });

        let ssa = nodes.len() as Node;
        let ssa_path: SSAPath = Vec::with_capacity(2 * ssa as usize - 1);

        ContractionProcessor {
            nodes,
            edges,
            appearances,
            sizes,
            ssa,
            ssa_path,
        }
    }

    fn neighbors(&self, i: Node) -> BTreeSet<Node> {
        let mut js = BTreeSet::default();
        for (ix, _) in self.nodes[&i].iter() {
            self.edges[&ix].iter().for_each(|&j| {
                if j != i {
                    js.insert(j);
                };
            });
        }
        js
    }

    /// remove an index from the graph, updating all legs
    fn remove_ix(&mut self, ix: Ix) {
        for j in self.edges.remove(&ix).unwrap() {
            self.nodes.get_mut(&j).unwrap().retain(|(k, _)| *k != ix);
        }
    }

    /// remove a node from the graph, update the edgemap, return the legs
    fn pop_node(&mut self, i: Node) -> Legs {
        let legs = self.nodes.remove(&i).unwrap();
        for (ix, _) in legs.iter() {
            let nodes = self.edges.get_mut(&ix).unwrap();
            if nodes.len() == 1 {
                self.edges.remove(&ix);
            } else {
                nodes.retain(|&j| j != i);
            }
        }
        legs
    }

    /// add a new node to the graph, update the edgemap, return the new id
    fn add_node(&mut self, legs: Legs) -> Node {
        let i = self.ssa;
        self.ssa += 1;
        for (ix, _) in &legs {
            self.edges
                .entry(*ix)
                .and_modify(|nodes| nodes.push(i))
                .or_insert(vec![i]);
        }
        self.nodes.insert(i, legs);
        i
    }

    /// contract two nodes, return the new node id
    fn contract_nodes(&mut self, i: Node, j: Node) -> Node {
        let ilegs = self.pop_node(i);
        let jlegs = self.pop_node(j);
        let new_legs = compute_legs(&ilegs, &jlegs, &self.appearances);
        let k = self.add_node(new_legs);
        self.ssa_path.push(vec![i, j]);
        k
    }

    /// find any indices that appear in all terms and just remove/ignore them
    fn simplify_batch(&mut self) {
        let mut ix_to_remove = Vec::new();
        let nterms = self.nodes.len();
        for (ix, ix_nodes) in self.edges.iter() {
            if ix_nodes.len() >= nterms {
                ix_to_remove.push(*ix);
            }
        }
        for ix in ix_to_remove {
            self.remove_ix(ix);
        }
    }

    /// perform any simplifications involving single terms
    fn simplify_single_terms(&mut self) {
        for (i, legs) in self.nodes.clone().into_iter() {
            if is_simplifiable(&legs, &self.appearances) {
                self.pop_node(i);
                let legs_reduced = compute_simplified(&legs, &self.appearances);
                self.add_node(legs_reduced);
                self.ssa_path.push(vec![i]);
            }
        }
    }

    /// combine and remove all scalars
    fn simplify_scalars(&mut self) {
        let mut scalars = Vec::new();
        for (i, legs) in self.nodes.iter() {
            if legs.len() == 0 {
                scalars.push(*i);
            }
        }
        if scalars.len() > 0 {
            for &i in &scalars {
                self.pop_node(i);
            }
            let (res, con) = match self.nodes.iter().min_by_key(|&(_, legs)| legs.len()) {
                Some((&j, _)) => {
                    let res = self.pop_node(j);
                    let con: Vec<Node> = scalars.into_iter().chain(vec![j].into_iter()).collect();
                    (res, con)
                }
                None => {
                    let res = Vec::new();
                    (res, scalars)
                }
            };
            self.add_node(res);
            self.ssa_path.push(con);
        }
    }

    /// combine all terms that have the same legs
    fn simplify_hadamard(&mut self) {
        // group all nodes by their legs (including permutations)
        let mut groups: Dict<BTreeSet<Ix>, Vec<Node>> = Dict::default();
        // keep track of groups with size >= 2
        let mut hadamards: BTreeSet<BTreeSet<Ix>> = BTreeSet::default();
        for (i, legs) in self.nodes.iter() {
            let key: BTreeSet<Ix> = legs.iter().map(|&(ix, _)| ix).collect();
            match groups.get_mut(&key) {
                Some(group) => {
                    hadamards.insert(key);
                    group.push(*i);
                }
                None => {
                    groups.insert(key, vec![*i]);
                }
            }
        }
        for key in hadamards.into_iter() {
            let mut group = groups.remove(&key).unwrap();
            while group.len() > 1 {
                let i = group.pop().unwrap();
                let j = group.pop().unwrap();
                let k = self.contract_nodes(i, j);
                group.push(k);
            }
        }
    }

    /// iteratively perform all simplifications until nothing left to do
    fn simplify(&mut self) {
        self.simplify_batch();
        let mut should_run = true;
        while should_run {
            self.simplify_single_terms();
            self.simplify_scalars();
            let ssa_before = self.ssa;
            self.simplify_hadamard();
            should_run = ssa_before != self.ssa;
        }
    }

    /// find disconnected subgraphs
    fn subgraphs(&self) -> Vec<Vec<Node>> {
        let mut remaining: BTreeSet<Node> = BTreeSet::default();
        self.nodes.keys().for_each(|i| {
            remaining.insert(*i);
        });
        let mut groups: Vec<Vec<Node>> = Vec::new();
        while remaining.len() > 0 {
            let i = remaining.pop_first().unwrap();
            let mut queue: Vec<Node> = vec![i];
            let mut group: BTreeSet<Node> = vec![i].into_iter().collect();
            while queue.len() > 0 {
                let i = queue.pop().unwrap();
                for j in self.neighbors(i) {
                    if !group.contains(&j) {
                        group.insert(j);
                        queue.push(j);
                    }
                }
            }
            group.iter().for_each(|i| {
                remaining.remove(i);
            });
            groups.push(group.into_iter().collect());
        }
        groups
    }

    /// greedily optimize the contraction order of all terms
    fn optimize_greedy(&mut self, costmod: Option<f32>, temperature: Option<f32>) {
        let mut rng = rand::thread_rng();
        let coeff_t = temperature.unwrap_or(0.0);
        let coeff_a = costmod.unwrap_or(1.0);
        let log_coeff_a = f32::ln(coeff_a);

        let mut local_score = |sa: Score, sb: Score, sab: Score| -> Score {
            let gumbel = if coeff_t.ne(&(0.0 as f32)) {
                -f32::ln(-f32::ln(rng.gen()))
            } else {
                0.0 as f32
            };
            logsub(sab, logadd(sa, sb) + log_coeff_a) - coeff_t * gumbel
        };

        // cache all current nodes sizes as we go
        let mut node_sizes: Dict<Node, Score> = Dict::default();
        self.nodes.iter().for_each(|(&i, legs)| {
            node_sizes.insert(i, compute_size(&legs, &self.sizes));
        });

        // we will *deincrement* c, since its a max-heap
        let mut c: i32 = 0;
        let mut queue: BinaryHeap<(GreedyScore, i32)> =
            BinaryHeap::with_capacity(self.edges.len() * 2);

        // the heap keeps a reference to actual contraction info in this
        let mut contractions: Dict<i32, (Node, Node, Score, Legs)> = Dict::default();

        // get the initial candidate contractions
        for ix_nodes in self.edges.values() {
            // for all combinations of nodes with a connected edge
            for ip in 0..ix_nodes.len() {
                let i = ix_nodes[ip];
                let isize = node_sizes[&i];
                for jp in (ip + 1)..ix_nodes.len() {
                    let j = ix_nodes[jp];
                    let jsize = node_sizes[&j];
                    let klegs = compute_legs(&self.nodes[&i], &self.nodes[&j], &self.appearances);
                    let ksize = compute_size(&klegs, &self.sizes);
                    let score = local_score(isize, jsize, ksize);
                    queue.push((OrderedFloat(-score), c));
                    contractions.insert(c, (i, j, ksize, klegs));
                    c -= 1;
                }
            }
        }

        // greedily contract remaining
        while let Some((_, c0)) = queue.pop() {
            let (i, j, ksize, klegs) = contractions.remove(&c0).unwrap();
            if !self.nodes.contains_key(&i) || !self.nodes.contains_key(&j) {
                // one of the nodes has been removed -> skip
                continue;
            }

            // perform contraction:
            // we already have the legs, so don't call contract_nodes
            self.pop_node(i);
            self.pop_node(j);
            let k = self.add_node(klegs.clone());
            self.ssa_path.push(vec![i, j]);
            node_sizes.insert(k, ksize);

            for l in self.neighbors(k) {
                // assess all neighboring contractions of new node
                let llegs = &self.nodes[&l];
                let lsize = node_sizes[&l];
                // get candidate legs and size
                let mlegs = compute_legs(&klegs, llegs, &self.appearances);
                let msize = compute_size(&mlegs, &self.sizes);
                let score = local_score(ksize, lsize, msize);
                queue.push((OrderedFloat(-score), c));
                contractions.insert(c, (k, l, msize, mlegs));
                c -= 1;
            }
        }
    }

    /// Optimize the contraction order of all terms using a greedy algorithm
    /// that contracts the smallest two terms. Typically only called once
    /// only disconnected subgraph terms (outer products) remain.
    fn optimize_remaining_by_size(&mut self) {
        if self.nodes.len() == 1 {
            // nothing to do
            return;
        };

        let mut nodes_sizes: BinaryHeap<(GreedyScore, Node)> = BinaryHeap::default();
        self.nodes.iter().for_each(|(node, legs)| {
            nodes_sizes.push((OrderedFloat(-compute_size(&legs, &self.sizes)), *node));
        });

        let (_, mut i) = nodes_sizes.pop().unwrap();
        let (_, mut j) = nodes_sizes.pop().unwrap();
        let mut k = self.contract_nodes(i, j);

        while self.nodes.len() > 1 {
            // contract the smallest two nodes until only one remains
            let ksize = compute_size(&self.nodes[&k], &self.sizes);
            nodes_sizes.push((OrderedFloat(-ksize), k));
            (_, i) = nodes_sizes.pop().unwrap();
            (_, j) = nodes_sizes.pop().unwrap();
            k = self.contract_nodes(i, j);
        }
    }
}

fn single_el_bitset(x: usize, n: usize) -> BitSet<ONode> {
    let mut a: BitSet<ONode> = BitSet::with_capacity(n);
    a.insert(x);
    a
}

fn compute_con_cost_flops(
    temp_legs: Legs,
    appearances: &Vec<Count>,
    sizes: &Vec<Score>,
    iscore: Score,
    jscore: Score,
    _factor: Score,
) -> (Legs, Score) {
    // remove indices that have reached final appearance
    // and compute cost and size of local contraction
    let mut new_legs: Legs = Legs::with_capacity(temp_legs.len());
    let mut cost: Score = 0.0;
    for (ix, ix_count) in temp_legs.into_iter() {
        // all involved indices contribute to the cost
        let d = sizes[ix as usize];
        cost += d;
        if ix_count != appearances[ix as usize] {
            // not last appearance -> kept index contributes to new size
            new_legs.push((ix, ix_count));
        }
    }
    let new_score = logadd(logadd(iscore, jscore), cost);
    (new_legs, new_score)
}

fn compute_con_cost_size(
    temp_legs: Legs,
    appearances: &Vec<Count>,
    sizes: &Vec<Score>,
    iscore: Score,
    jscore: Score,
    _factor: Score,
) -> (Legs, Score) {
    // remove indices that have reached final appearance
    // and compute cost and size of local contraction
    let mut new_legs: Legs = Legs::with_capacity(temp_legs.len());
    let mut size: Score = 0.0;
    for (ix, ix_count) in temp_legs.into_iter() {
        if ix_count != appearances[ix as usize] {
            // not last appearance -> kept index contributes to new size
            new_legs.push((ix, ix_count));
            size += sizes[ix as usize];
        }
    }
    let new_score = iscore.max(jscore).max(size);
    (new_legs, new_score)
}

fn compute_con_cost_write(
    temp_legs: Legs,
    appearances: &Vec<Count>,
    sizes: &Vec<Score>,
    iscore: Score,
    jscore: Score,
    _factor: Score,
) -> (Legs, Score) {
    // remove indices that have reached final appearance
    // and compute cost and size of local contraction
    let mut new_legs: Legs = Legs::with_capacity(temp_legs.len());
    let mut size: Score = 0.0;
    for (ix, ix_count) in temp_legs.into_iter() {
        if ix_count != appearances[ix as usize] {
            // not last appearance -> kept index contributes to new size
            new_legs.push((ix, ix_count));
            size += sizes[ix as usize];
        }
    }
    let new_score = logadd(logadd(iscore, jscore), size);
    (new_legs, new_score)
}

fn compute_con_cost_combo(
    temp_legs: Legs,
    appearances: &Vec<Count>,
    sizes: &Vec<Score>,
    iscore: Score,
    jscore: Score,
    factor: Score,
) -> (Legs, Score) {
    // remove indices that have reached final appearance
    // and compute cost and size of local contraction
    let mut new_legs: Legs = Legs::with_capacity(temp_legs.len());
    let mut size: Score = 0.0;
    let mut cost: Score = 0.0;
    for (ix, ix_count) in temp_legs.into_iter() {
        // all involved indices contribute to the cost
        let d = sizes[ix as usize];
        cost += d;
        if ix_count != appearances[ix as usize] {
            // not last appearance -> kept index contributes to new size
            new_legs.push((ix, ix_count));
            size += d;
        }
    }
    // the score just for this contraction
    let new_local_score = logadd(cost, factor + size);

    // the total score including history
    let new_score = logadd(logadd(iscore, jscore), new_local_score);

    (new_legs, new_score)
}

impl ContractionProcessor {
    fn optimize_optimal_connected(
        &mut self,
        subgraph: Vec<Node>,
        minimize: Option<String>,
        factor: Option<Score>,
        cost_cap: Option<Score>,
    ) {
        let minimize = minimize.unwrap_or("flops".to_string());
        let factor = f32::ln(factor.unwrap_or(64.0));
        let compute_cost = match minimize.as_str() {
            "flops" => compute_con_cost_flops,
            "size" => compute_con_cost_size,
            "write" => compute_con_cost_write,
            "combo" => compute_con_cost_combo,
            _ => panic!(
                "minimize must be one of 'flops', 'size', 'write', or 'combo', got {}",
                minimize
            ),
        };

        // storage for each possible contraction to reach subgraph of size m
        let mut contractions: Vec<Dict<Subgraph, SubContraction>> =
            vec![Dict::default(); subgraph.len() + 1];
        // intermediate storage for the entries we are expanding
        let mut contractions_m_temp: Vec<(Subgraph, SubContraction)> = Vec::new();
        // need to keep these separately
        let mut best_scores: Dict<Subgraph, Score> = Dict::default();

        // we use linear index within terms given during optimization, this maps
        // back to the original node index
        let nterms = subgraph.len();
        let mut termmap: Dict<Subgraph, Node> = Dict::default();

        for (i, node) in subgraph.into_iter().enumerate() {
            let isubgraph = single_el_bitset(i, nterms);
            termmap.insert(isubgraph.clone(), node);
            let ilegs = self.nodes[&node].clone();
            let iscore: Score = 0.0;
            let ipath: BitPath = Vec::new();
            contractions[1].insert(isubgraph, (ilegs, iscore, ipath));
        }

        let mut ip: usize;
        let mut jp: usize;
        let mut outer: bool;

        let cost_cap_incr = f32::ln(2.0);
        let mut cost_cap = cost_cap.unwrap_or(cost_cap_incr);
        while contractions[nterms].len() == 0 {
            // try building subgraphs of size m
            for m in 2..=nterms {
                // out of bipartitions of size (k, m - k)
                for k in 1..=m / 2 {
                    for (isubgraph, (ilegs, iscore, ipath)) in contractions[k].iter() {
                        for (jsubgraph, (jlegs, jscore, jpath)) in contractions[m - k].iter() {
                            // filter invalid combinations first
                            if !isubgraph.is_disjoint(&jsubgraph) || {
                                (k == m - k) && isubgraph.gt(&jsubgraph)
                            } {
                                // subgraphs overlap -> not valid, or
                                // equal subgraph size -> only process sorted pairs
                                continue;
                            }

                            let mut temp_legs: Legs = Vec::with_capacity(ilegs.len() + jlegs.len());
                            ip = 0;
                            jp = 0;
                            outer = true;
                            while ip < ilegs.len() && jp < jlegs.len() {
                                if ilegs[ip].0 < jlegs[jp].0 {
                                    // index only appears in ilegs
                                    temp_legs.push(ilegs[ip]);
                                    ip += 1;
                                } else if ilegs[ip].0 > jlegs[jp].0 {
                                    // index only appears in jlegs
                                    temp_legs.push(jlegs[jp]);
                                    jp += 1;
                                } else {
                                    // index appears in both
                                    temp_legs.push((ilegs[ip].0, ilegs[ip].1 + jlegs[jp].1));
                                    ip += 1;
                                    jp += 1;
                                    outer = false;
                                }
                            }
                            if outer {
                                // no shared indices -> outer product
                                continue;
                            }
                            // add any remaining indices
                            temp_legs.extend(ilegs[ip..].iter().chain(jlegs[jp..].iter()));

                            // compute candidate contraction result and score
                            let (new_legs, new_score) = compute_cost(
                                temp_legs,
                                &self.appearances,
                                &self.sizes,
                                *iscore,
                                *jscore,
                                factor,
                            );

                            if new_score > cost_cap {
                                // contraction not allowed yet due to cost
                                continue;
                            }

                            // check candidate against current best subgraph path
                            let new_subgraph: Subgraph = isubgraph.union(&jsubgraph).collect();

                            // because we have to do a delayed update of
                            // contractions[m] for borrowing reasons, we check
                            // against a non-delayed score lookup so we don't
                            // overwrite best scores within the same iteration
                            let found_new_best = match best_scores.get(&new_subgraph) {
                                Some(current_score) => new_score < *current_score,
                                None => true,
                            };
                            if found_new_best {
                                best_scores.insert(new_subgraph.clone(), new_score);
                                // only need the path if updating
                                let mut new_path: BitPath =
                                    Vec::with_capacity(ipath.len() + jpath.len() + 1);
                                new_path.extend_from_slice(&ipath);
                                new_path.extend_from_slice(&jpath);
                                new_path.push((isubgraph.clone(), jsubgraph.clone()));
                                contractions_m_temp
                                    .push((new_subgraph, (new_legs, new_score, new_path)));
                            }
                        }
                    }
                    // move new contractions from temp into the main storage, there
                    // might be contractions for the same subgraph in this, but
                    // because we check eagerly best_scores above, later entries
                    // are guaranteed to be better
                    contractions_m_temp.drain(..).for_each(|(k, v)| {
                        contractions[m].insert(k, v);
                    });
                }
            }
            cost_cap += cost_cap_incr;
        }
        // can only ever be a single entry in contractions[nterms] -> the best one
        let (_, _, best_path) = contractions[nterms].values().next().unwrap();

        // convert from the bitpath to the actual (subgraph) node ids
        for (isubgraph, jsubgraph) in best_path.into_iter() {
            let i = termmap[&isubgraph];
            let j = termmap[&jsubgraph];
            let k = self.contract_nodes(i, j);
            let ksubgraph: Subgraph = isubgraph.union(&jsubgraph).collect();
            termmap.insert(ksubgraph, k);
        }
    }

    fn optimize_optimal(
        &mut self,
        minimize: Option<String>,
        factor: Option<Score>,
        cost_cap: Option<Score>,
    ) {
        for subgraph in self.subgraphs() {
            self.optimize_optimal_connected(subgraph, minimize.clone(), factor, cost_cap);
        }
    }
}

// --------------------------- PYTHON FUNCTIONS ---------------------------- //

#[pyfunction]
#[pyo3()]
fn find_subgraphs(
    inputs: Vec<Vec<char>>,
    output: Vec<char>,
    size_dict: Dict<char, u32>,
) -> Vec<Vec<Node>> {
    let cp = ContractionProcessor::new(inputs, output, size_dict);
    cp.subgraphs()
}

#[pyfunction]
#[pyo3()]
fn optimize_simplify(
    inputs: Vec<Vec<char>>,
    output: Vec<char>,
    size_dict: Dict<char, u32>,
) -> SSAPath {
    let mut cp = ContractionProcessor::new(inputs, output, size_dict);
    cp.simplify();
    cp.ssa_path
}

#[pyfunction]
#[pyo3()]
fn optimize_greedy(
    inputs: Vec<Vec<char>>,
    output: Vec<char>,
    size_dict: Dict<char, u32>,
    costmod: Option<f32>,
    temperature: Option<f32>,
    simplify: Option<bool>,
) -> Vec<Vec<Node>> {
    let mut cp = ContractionProcessor::new(inputs, output, size_dict);
    if simplify.unwrap_or(true) {
        cp.simplify();
    }
    cp.optimize_greedy(costmod, temperature);
    // optimize any remaining disconnected terms
    cp.optimize_remaining_by_size();
    cp.ssa_path
}

#[pyfunction]
#[pyo3()]
fn optimize_optimal(
    inputs: Vec<Vec<char>>,
    output: Vec<char>,
    size_dict: Dict<char, u32>,
    minimize: Option<String>,
    factor: Option<Score>,
    cost_cap: Option<Score>,
    simplify: Option<bool>,
) -> Vec<Vec<Node>> {
    let mut cp = ContractionProcessor::new(inputs, output, size_dict);
    if simplify.unwrap_or(true) {
        cp.simplify();
    }
    cp.optimize_optimal(minimize, factor, cost_cap);
    // optimize any remaining disconnected terms
    cp.optimize_remaining_by_size();
    cp.ssa_path
}

/// A Python module implemented in Rust.
#[pymodule]
fn cotengrust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_subgraphs, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_simplify, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_greedy, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_optimal, m)?)?;
    Ok(())
}
