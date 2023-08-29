// use bit_set::BitSet;
use std::collections::{BTreeSet, BinaryHeap};
use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use FxHashMap as Dict;

type Ix = u16;
type Count = u8;
type Legs = Vec<(Ix, Count)>;
type Node = u32;
// type Subgraph = BitSet<Node>;
type Score = u128;
type GreedyScore = i128;
// type BitPath = Vec<(Subgraph, Subgraph)>;
type SSAPath = Vec<Vec<Node>>;
// type SubContraction = (Legs, Score, BitPath);

struct HypergraphProcessor {
    nodes: Dict<Node, Legs>,
    edges: Dict<Ix, Vec<Node>>,
    appearances: Vec<Count>,
    sizes: Vec<Score>,
    ssa: Node,
    ssa_path: SSAPath,
}

fn compute_legs(ilegs: &Legs, jlegs: &Legs, appearances: &Vec<Count>) -> Legs {
    let mut ip = 0;
    let mut jp = 0;
    let ni = ilegs.len();
    let nj = jlegs.len();
    let mut new_legs: Legs = Vec::new();

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
    legs.iter().map(|&(ix, _)| sizes[ix as usize]).product()
}

impl HypergraphProcessor {
    fn new(
        inputs: Vec<Vec<char>>,
        output: Vec<char>,
        size_dict: Dict<char, u32>,
    ) -> HypergraphProcessor {
        let mut nodes: Dict<Node, Legs> = Dict::default();
        let mut edges: Dict<Ix, Vec<Node>> = Dict::default();
        let mut indmap: Dict<char, Ix> = Dict::default();
        let mut sizes: Vec<Score> = Vec::with_capacity(size_dict.len());
        let mut appearances: Vec<Count> = Vec::with_capacity(size_dict.len());
        let mut c: Ix = 0;

        for (i, term) in inputs.into_iter().enumerate() {
            let mut legs = Vec::new();
            for ind in term {
                match indmap.get(&ind) {
                    None => {
                        // index not parsed yet
                        indmap.insert(ind, c);
                        edges.insert(c, vec![i as Node]);
                        appearances.push(1);
                        sizes.push(size_dict[&ind] as Score);
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
        let ssa_path: SSAPath = Vec::new();

        HypergraphProcessor {
            nodes,
            edges,
            appearances,
            sizes,
            ssa,
            ssa_path,
        }
    }

    fn neighbors(&self, i: Node) -> Vec<Node> {
        let mut js = Vec::new();
        for (ix, _) in self.nodes[&i].iter() {
            for j in self.edges[&ix].iter() {
                if *j != i {
                    js.push(*j);
                }
            }
        }
        js
    }

    fn remove_ix(&mut self, ix: Ix) {
        for j in self.edges.remove(&ix).unwrap() {
            self.nodes.get_mut(&j).unwrap().retain(|(k, _)| *k != ix);
        }
    }

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

    fn contract(&mut self, i: Node, j: Node) -> Node {
        let ilegs = self.pop_node(i);
        let jlegs = self.pop_node(j);
        let new_legs = compute_legs(&ilegs, &jlegs, &self.appearances);
        let k = self.add_node(new_legs);
        self.ssa_path.push(vec![i, j]);
        k
    }

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

    fn simplify_single_terms(&mut self) {
        for (i, legs) in self.nodes.clone().into_iter() {
            if legs
                .iter()
                .any(|&(ix, c)| c == self.appearances[ix as usize])
            {
                let mut legs_reduced = self.pop_node(i);
                legs_reduced.retain(|&(ix, c)| c != self.appearances[ix as usize]);
                self.add_node(legs_reduced);
                self.ssa_path.push(vec![i]);
            }
        }
    }

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

    fn simplify_hadamard(&mut self) {
        let mut groups: Dict<BTreeSet<Ix>, Vec<Node>> = Dict::default();
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
                let k = self.contract(i, j);
                group.push(k);
            }
        }
    }

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

    fn optimize_greedy(&mut self) {
        // cache all nodes sizes as we go
        let mut node_sizes: Dict<Node, Score> = Dict::default();
        self.nodes.iter().for_each(|(&i, legs)| {
            node_sizes.insert(i, compute_size(&legs, &self.sizes));
        });

        let mut queue: BinaryHeap<(GreedyScore, usize)> =
            BinaryHeap::with_capacity(self.edges.len() * 2);
        let mut contractions: Dict<usize, (Node, Node, Score, Legs)> = Dict::default();
        let mut checked: FxHashSet<(Node, Node)> = FxHashSet::default();
        let mut c: usize = 0;

        // get the initial candidate contractions
        for ix_nodes in self.edges.values() {
            for ip in 0..ix_nodes.len() {
                let i = ix_nodes[ip];
                let isize = node_sizes[&i];
                for jp in (ip + 1)..ix_nodes.len() {
                    let j = ix_nodes[jp];
                    let jsize = node_sizes[&j];
                    let klegs = compute_legs(
                        &self.nodes[&i],
                        &self.nodes[&j],
                        &self.appearances,
                    );
                    let ksize = compute_size(&klegs, &self.sizes);
                    let score = (isize as GreedyScore) - ((jsize + ksize) as GreedyScore);
                    queue.push((-score, c));
                    contractions.insert(c, (i, j, ksize, klegs));
                    c += 1;
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

            self.pop_node(i);
            self.pop_node(j);
            let k = self.add_node(klegs.clone());
            self.ssa_path.push(vec![i, j]);
            node_sizes.insert(k, ksize);

            for l in self.neighbors(k) {
                let key = if k < l { (k, l) } else { (l, k) };
                if checked.contains(&key) {
                    continue;
                } else {
                    checked.insert(key);
                }
                let llegs = &self.nodes[&l];
                let lsize = node_sizes[&l];
                let mlegs = compute_legs(&klegs, llegs, &self.appearances);
                let msize = compute_size(&mlegs, &self.sizes);
                let score = (msize as GreedyScore) - ((ksize + lsize) as GreedyScore);
                queue.push((-score, c));
                contractions.insert(c, (k, l, msize, mlegs));
                c += 1;
            }
        }
    }
}

#[pyfunction]
#[pyo3()]
fn test_simplify(
    inputs: Vec<Vec<char>>,
    output: Vec<char>,
    size_dict: Dict<char, u32>,
) -> SSAPath {
    let mut hgp = HypergraphProcessor::new(inputs, output, size_dict);
    hgp.simplify();
    hgp.ssa_path
}

#[pyfunction]
#[pyo3()]
fn test_subgraphs(
    inputs: Vec<Vec<char>>,
    output: Vec<char>,
    size_dict: Dict<char, u32>,
) -> Vec<Vec<Node>> {
    let hgp = HypergraphProcessor::new(inputs, output, size_dict);
    hgp.subgraphs()
}

#[pyfunction]
#[pyo3()]
fn test_greedy(
    inputs: Vec<Vec<char>>,
    output: Vec<char>,
    size_dict: Dict<char, u32>,
) -> Vec<Vec<Node>> {
    let mut hgp = HypergraphProcessor::new(inputs, output, size_dict);
    hgp.simplify();
    hgp.optimize_greedy();
    hgp.ssa_path
}

// fn single_el_bitset(x: usize, n: usize) -> BitSet<Node> {
//     let mut a: BitSet<Node> = BitSet::with_capacity(n);
//     a.insert(x);
//     a
// }

// fn compute_con_cost_flops(
//     temp_legs: Legs,
//     appearances: &Vec<Count>,
//     sizes: &Vec<Score>,
//     iscore: &Score,
//     jscore: &Score,
//     _factor: Score,
// ) -> (Legs, Score) {
//     // remove indices that have reached final appearance
//     // and compute cost and size of local contraction
//     let mut new_legs: Legs = Legs::new();
//     let mut cost: Score = 1;
//     for (ix, ix_count) in temp_legs.into_iter() {
//         // all involved indices contribute to the cost
//         let d = sizes[ix as usize];
//         cost *= d;
//         if ix_count != appearances[ix as usize] {
//             // not last appearance -> kept index contributes to new size
//             new_legs.push((ix, ix_count));
//         }
//     }
//     let new_score = iscore + jscore + cost;
//     (new_legs, new_score)
// }

// fn compute_con_cost_size(
//     temp_legs: Legs,
//     appearances: &Vec<Count>,
//     sizes: &Vec<Score>,
//     iscore: &Score,
//     jscore: &Score,
//     _factor: Score,
// ) -> (Legs, Score) {
//     // remove indices that have reached final appearance
//     // and compute cost and size of local contraction
//     let mut new_legs: Legs = Legs::new();
//     let mut size: Score = 1;
//     for (ix, ix_count) in temp_legs.into_iter() {
//         if ix_count != appearances[ix as usize] {
//             // not last appearance -> kept index contributes to new size
//             new_legs.push((ix, ix_count));
//             size *= sizes[ix as usize];
//         }
//     }
//     let new_score = *iscore.max(jscore).max(&size);
//     (new_legs, new_score)
// }

// fn compute_con_cost_write(
//     temp_legs: Legs,
//     appearances: &Vec<Count>,
//     sizes: &Vec<Score>,
//     iscore: &Score,
//     jscore: &Score,
//     _factor: Score,
// ) -> (Legs, Score) {
//     // remove indices that have reached final appearance
//     // and compute cost and size of local contraction
//     let mut new_legs: Legs = Legs::new();
//     let mut size: Score = 1;
//     for (ix, ix_count) in temp_legs.into_iter() {
//         if ix_count != appearances[ix as usize] {
//             // not last appearance -> kept index contributes to new size
//             new_legs.push((ix, ix_count));
//             size *= sizes[ix as usize];
//         }
//     }
//     let new_score = iscore + jscore + size;
//     (new_legs, new_score)
// }

// fn compute_con_cost_combo(
//     temp_legs: Legs,
//     appearances: &Vec<Count>,
//     sizes: &Vec<Score>,
//     iscore: &Score,
//     jscore: &Score,
//     factor: Score,
// ) -> (Legs, Score) {
//     // remove indices that have reached final appearance
//     // and compute cost and size of local contraction
//     let mut new_legs: Legs = Legs::new();
//     let mut size: Score = 1;
//     let mut cost: Score = 1;
//     for (ix, ix_count) in temp_legs.into_iter() {
//         // all involved indices contribute to the cost
//         let d = sizes[ix as usize];
//         cost *= d;
//         if ix_count != appearances[ix as usize] {
//             // not last appearance -> kept index contributes to new size
//             new_legs.push((ix, ix_count));
//             size *= d;
//         }
//     }
//     // the score just for this contraction
//     let new_local_score = cost + factor * size;

//     // the total score including history
//     let new_score = iscore + jscore + new_local_score;

//     (new_legs, new_score)
// }

// fn convert_bitpath_to_ssapath(bitpath: &BitPath, nterms: usize) -> SSAPath {
//     let mut subgraph_to_ssa = Dict::default();
//     let mut ssa = 0;
//     let mut ssa_path = Vec::new();
//     // create ssa leaves
//     for i in 0..nterms {
//         subgraph_to_ssa.insert(single_el_bitset(i, nterms), ssa);
//         ssa += 1;
//     }
//     // process the path, creating parent ssa ids as we go
//     for (isubgraph, jsubgraph) in bitpath.into_iter() {
//         ssa_path.push((subgraph_to_ssa[isubgraph], subgraph_to_ssa[jsubgraph]));
//         subgraph_to_ssa.insert(isubgraph.union(&jsubgraph).collect(), ssa);
//         ssa += 1;
//     }
//     ssa_path
// }

// #[pyfunction]
// #[pyo3()]
// fn optimal(
//     inputs: Vec<Vec<char>>,
//     output: Vec<char>,
//     size_dict: Dict<char, Score>,
//     minimize: Option<String>,
//     factor: Option<Score>,
//     cost_cap: Option<Score>,
// ) -> SSAPath {
//     let minimize = minimize.unwrap_or("flops".to_string());
//     let factor = factor.unwrap_or(64);
//     let compute_cost = match minimize.as_str() {
//         "flops" => compute_con_cost_flops,
//         "size" => compute_con_cost_size,
//         "write" => compute_con_cost_write,
//         "combo" => compute_con_cost_combo,
//         _ => panic!(
//             "minimize must be one of 'flops', 'size', 'write', or 'combo', got {}",
//             minimize
//         ),
//     };

//     let nterms = inputs.len();
//     let mut indmap: Dict<char, Ix> = Dict::default();
//     let mut sizes: Vec<Score> = Vec::with_capacity(size_dict.len());
//     let mut appearances: Vec<Count> = Vec::with_capacity(size_dict.len());
//     let mut c: Ix = 0;
//     // storage for each possible contraction to reach subgraph of size k
//     let mut contractions: Vec<Dict<Subgraph, SubContraction>> = vec![Dict::default(); nterms + 1];
//     // intermediate storage for the entries we are expanding
//     let mut contractions_m_temp: Vec<(Subgraph, SubContraction)> = Vec::new();
//     // need to keep these separately
//     let mut best_scores: Dict<Subgraph, Score> = Dict::default();

//     // map the string indices to integers, forming the int input terms as well
//     for (j, term) in inputs.into_iter().enumerate() {
//         let mut legs: Legs = Vec::new();
//         for ind in term {
//             match indmap.get(&ind) {
//                 Some(&cex) => {
//                     // index already present
//                     appearances[cex as usize] += 1;
//                     legs.push((cex, 1));
//                 }
//                 None => {
//                     // index not parsed yet
//                     indmap.insert(ind.clone(), c);
//                     sizes.push(size_dict[&ind]);
//                     appearances.push(1);
//                     legs.push((c, 1));
//                     c += 1;
//                 }
//             };
//         }
//         legs.sort();
//         contractions[1].insert(single_el_bitset(j, nterms), (legs, 0, Vec::new()));
//     }
//     // parse the output -> just needed for appearances sake
//     output.into_iter().for_each(|ind| {
//         appearances[indmap[&ind] as usize] += 1;
//     });

//     // let mut inds_to_remove: Vec<Ix> = Vec::new();
//     let mut ip: usize;
//     let mut jp: usize;
//     let mut outer: bool;

//     let mut cost_cap = cost_cap.unwrap_or(1);
//     while contractions[nterms].len() == 0 {
//         // try building subgraphs of size m
//         for m in 2..=nterms {
//             // out of bipartitions of size (k, m - k)
//             for k in 1..=m / 2 {
//                 for (isubgraph, (ilegs, iscore, ipath)) in contractions[k].iter() {
//                     for (jsubgraph, (jlegs, jscore, jpath)) in contractions[m - k].iter() {
//                         // filter invalid combinations first
//                         if !isubgraph.is_disjoint(&jsubgraph) || {
//                             (k == m - k) && isubgraph.gt(&jsubgraph)
//                         } {
//                             // subgraphs overlap -> not valid, or
//                             // equal subgraph size -> only process sorted pairs
//                             continue;
//                         }

//                         let mut temp_legs: Legs = Vec::new();
//                         ip = 0;
//                         jp = 0;
//                         outer = true;
//                         while ip < ilegs.len() && jp < jlegs.len() {
//                             if ilegs[ip].0 < jlegs[jp].0 {
//                                 // index only appears in ilegs
//                                 temp_legs.push(ilegs[ip]);
//                                 ip += 1;
//                             } else if ilegs[ip].0 > jlegs[jp].0 {
//                                 // index only appears in jlegs
//                                 temp_legs.push(jlegs[jp]);
//                                 jp += 1;
//                             } else {
//                                 // index appears in both
//                                 temp_legs.push((ilegs[ip].0, ilegs[ip].1 + jlegs[jp].1));
//                                 ip += 1;
//                                 jp += 1;
//                                 outer = false;
//                             }
//                         }
//                         if outer {
//                             // no shared indices -> outer product
//                             continue;
//                         }
//                         // add any remaining indices
//                         temp_legs.extend(ilegs[ip..].iter().chain(jlegs[jp..].iter()));

//                         // compute candidate contraction result and score
//                         let (new_legs, new_score) =
//                             compute_cost(temp_legs, &appearances, &sizes, iscore, jscore, factor);

//                         if new_score > cost_cap {
//                             // contraction not allowed yet due to cost
//                             continue;
//                         }

//                         // check candidate against current best subgraph path
//                         let new_subgraph: Subgraph = isubgraph.union(&jsubgraph).collect();

//                         // because we have to do a delayed update of
//                         // contractions[m] for borrowing reasons, we check
//                         // against a non-delayed score lookup so we don't
//                         // overwrite best scores within the same iteration
//                         let new_best = match best_scores.get(&new_subgraph) {
//                             Some(current_score) => new_score < *current_score,
//                             None => true,
//                         };
//                         if new_best {
//                             best_scores.insert(new_subgraph.clone(), new_score);
//                             // only need the path if updating
//                             let mut new_path = ipath.clone();
//                             new_path.extend(jpath.clone());
//                             new_path.push((isubgraph.clone(), jsubgraph.clone()));
//                             contractions_m_temp
//                                 .push((new_subgraph, (new_legs, new_score, new_path)));
//                         }
//                     }
//                 }
//                 // move new contractions from temp into the main storage, there
//                 // might be contractions for the same subgraph in this, but
//                 // because we check eagerly best_scores above, later entries
//                 // are guaranteed to be better
//                 contractions_m_temp.drain(..).for_each(|(k, v)| {
//                     contractions[m].insert(k, v);
//                 });
//             }
//         }
//         cost_cap *= 2;
//     }
//     // can only ever be a single entry in contractions[nterms] -> the best one
//     let (_, _, best_path) = contractions[nterms].values().next().unwrap();

//     // need to convert the path to 'ssa' ids rather than bitset
//     return convert_bitpath_to_ssapath(best_path, nterms);
// }

/// A Python module implemented in Rust.
#[pymodule]
fn cotengrust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test_simplify, m)?)?;
    m.add_function(wrap_pyfunction!(test_subgraphs, m)?)?;
    m.add_function(wrap_pyfunction!(test_greedy, m)?)?;
    Ok(())
}
