#![allow(dead_code)]

use crate::not_nan_util::{dist, NotNan1D};
use ahash::{HashMap, HashMapExt, HashSet, HashSetExt};
use itertools::{any, Itertools};
use lazy_static::lazy_static;
use log::{debug, info, trace, warn};
use once_cell::sync::Lazy;
use ordered_float::NotNan;
use petgraph::visit::GetAdjacencyMatrix;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use ringbuffer::{AllocRingBuffer, ConstGenericRingBuffer, RingBuffer};
use std::cmp::{max, min};
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::BinaryHeap;
use std::fmt;
use std::ops::{Deref, DerefMut, Index};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const INTERSECT_THRESHOLD: f32 = 0.0;

/*
 * Hope that these are different types... (usize should be u64)
 * Wait until this RFC is implemented to make this easier
 * https://github.com/rust-lang/rfcs/issues/261
 */
pub type PhysicalIndex = petgraph::matrix_graph::NodeIndex<u32>;
pub type NagIndex = petgraph::graph::NodeIndex<usize>;

/**
 * We use a RingBuffer under the assumption that old nag nodes are unlikely to be relevant
 * This is to prevent find_equivalent_nodes from blowing up
 */
const BUFFER_SIZE: usize = 5;
pub type CoincidentalNags = Vec<NagIndex>;
// pub type CoincidentalNags = ConstGenericRingBuffer<NagIndex, BUFFER_SIZE>;
pub type PhysicalNagMap = HashMap<PhysicalIndex, CoincidentalNags>;

/**
 * Matrix Graph allows fast retrieval of adjacency matrix (most important)
 * and edge insertion
 * but poor edge retrieval
 */
// pub type Graph = petgraph::matrix_graph::UnMatrix<GraphNode, NotNan<f32>, Option<NotNan<f32>>, u32>;
pub type Graph = petgraph::matrix_graph::DiMatrix<GraphNode, NotNan<f32>, Option<NotNan<f32>>, u32>;

/**
 * Normal graph allows fast edge retrieval
 * but slow edge insertion
 */
// pub type Nag = petgraph::graph::UnGraph<NagNode, (), u32>;

type NagGraphType = petgraph::matrix_graph::UnMatrix<NagNode, (), Option<()>, usize>;

lazy_static! {
  static ref neighborhood_cache: Mutex<HashMap<NagIndex, HashSet<NagIndex>>> =
    Mutex::new(HashMap::new());
}

/**
 * Combine the best of both worlds:  Have a matrix graph for fast adjacency matrix retrieval
 * But pair it with an adjacency list for fast edge retrieval
 */
#[derive(Clone)]
pub struct Nag {
  pub nag: NagGraphType,
  pub edge_map: HashMap<NagIndex, HashSet<NagIndex>>,
}

// impl Index<NagIndex> for Nag {
//   type Output = NagNode;
//   fn index(&self, index: NagIndex) -> &NagNode {
//     &self.nag[index]
//   }
// }

impl Deref for Nag {
  type Target = NagGraphType;

  fn deref(&self) -> &Self::Target {
    &self.nag
  }
}

impl DerefMut for Nag {
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.nag
  }
}

static EMPTY_SET: Lazy<HashSet<NagIndex>> = Lazy::new(HashSet::new);
impl Nag {
  fn with_capacity(size: usize) -> Self {
    Nag {
      nag: NagGraphType::with_capacity(size),
      edge_map: HashMap::new(),
    }
  }

  fn add_edge(&mut self, a: NagIndex, b: NagIndex) {
    if self.nag.update_edge(a, b, ()).is_none() {
      /* Only add edge if edge doesn't alread exist */
      match self.edge_map.entry(a) {
        Occupied(ent) => {
          ent.into_mut().insert(b);
        }
        Vacant(ent) => {
          ent.insert(HashSet::from_iter(vec![b]));
        }
      }
      match self.edge_map.entry(b) {
        Occupied(ent) => {
          ent.into_mut().insert(a);
        }
        Vacant(ent) => {
          ent.insert(HashSet::from_iter(vec![a]));
        }
      }
      self.nag[a].parents.insert(b);
      self.nag[b].parents.insert(a);
      neighborhood_cache.lock().unwrap().remove(&a);
      neighborhood_cache.lock().unwrap().remove(&b);
    }
  }

  fn neighbors(&self, a: NagIndex) -> &HashSet<NagIndex> {
    if let Some(set) = self.edge_map.get(&a) {
      set
    } else {
      &EMPTY_SET
    }
  }

  // fn adjacency_matrix(&self) -> Self::AdjMatrix {
  //   self.nag.adjacency_matrix()
  // }
}

pub type GraphNode = NotNan1D;
// #[derive(Clone)]
// pub struct GraphNode {
//   pub position: NotNan1D,
// }

// impl From<NotNan1D> for GraphNode {
//   fn from(position: NotNan1D) -> Self {
//     GraphNode {
//       position,
//     }
//   }
// }

// impl<'a> From<&'a GraphNode> for &'a NotNan1D {
//   fn from(node: &'a GraphNode) -> Self {
//     &node.position
//   }
// }

// impl From<GraphNode> for NotNan1D {
//   fn from(node: GraphNode) -> Self {
//     node.position
//   }
// }

// impl GraphNode {
//   pub fn new(position: NotNan1D) -> Self {
//     GraphNode {
//       position,
//     }
//   }
// }

#[derive(Clone, Eq, PartialEq)]
pub struct NagNode {
  pub value: NotNan<f32>,
  pub parent: Option<NagIndex>,
  pub physical_index: PhysicalIndex,
  pub nag_index: Option<NagIndex>,
  pub parents: HashSet<NagIndex>,
}

impl Ord for NagNode {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    other.value.cmp(&self.value)
  }
}

impl PartialOrd for NagNode {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    Some(self.cmp(other))
  }
}

impl NagNode {
  fn new(physical_index: PhysicalIndex) -> Self {
    Self {
      value: NotNan::new(0.0).unwrap(),
      parent: None,
      physical_index,
      nag_index: None,
      parents: HashSet::new(),
    }
  }
  fn create_child(
    &self,
    graph: &Graph,
    nag: &Nag,
    physical_index: PhysicalIndex,
    edge_weight: NotNan<f32>,
  ) -> NagNode {
    // let mut neighborhood = HashSet::from_iter(nag.neighbors(self.nag_index.unwrap()));
    // neighborhood.insert(self.nag_index.unwrap()); // Add self to child's neighborhood
    let child = NagNode {
      value: self.value + edge_weight,
      parent: Some(self.nag_index.unwrap()),
      physical_index,
      nag_index: None,
      parents: HashSet::from_iter([self.nag_index.unwrap()]),
    };
    child
  }
  fn add_to_nag(self, nag: &mut Nag) -> NagIndex {
    let parent = self.parent;
    let nag_index = nag.add_node(self);
    nag[nag_index].nag_index = Some(nag_index);
    if let Some(parent) = parent {
      nag.add_edge(nag_index, parent);
      nag[nag_index].parents.insert(parent);
      nag[parent].parents.insert(nag_index);
    }
    // nag[nag_index].neighborhood.insert(nag_index); // self is always in the neighborhood
    nag.add_edge(nag_index, nag_index); // self is always in the neighborhood
    nag[nag_index].parents.insert(nag_index);

    nag_index
  }
}

impl fmt::Display for NagNode {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(
      f,
      "(PHY{}, NAG{})",
      self.physical_index.index(),
      if self.nag_index.is_some() {
        self.nag_index.unwrap().index().to_string()
      } else {
        "?".to_string()
      }
    )
  }
}

impl fmt::Debug for NagNode {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(
      f,
      "NagNode {{ PHY{}, NAG{}, value={}, parent=NAG{} }}",
      self.physical_index.index(),
      if self.nag_index.is_some() {
        self.nag_index.unwrap().index().to_string()
      } else {
        "?".to_string()
      },
      self.value,
      if self.parent.is_some() {
        self.parent.unwrap().index().to_string()
      } else {
        "?".to_string()
      },
    )
  }
}

// struct NagMeta {
//   graph: Nag,
//   nag_mapping: HashMap<NodeIndex, Rc<NagNode>>,
//   physical_mapping: HashMap<NodeIndex, Vec<Rc<NagNode>>>
// }

// impl NagMeta {
//   pub fn new() -> Self {
//     NagMeta {
//       graph: Nag::new_undirected(),
//       nag_mapping: HashMap::new(),
//       physical_mapping: HashMap::new()
//     }
//   }

//   pub fn add_node(&self, nag_node) {

//     nag_mapping.insert(nag_index, nag_node.clone());
//     match physical_mapping.entry(physical_index) {
//       Occupied(ent) => {
//         ent.get_mut().push(root.clone());
//       }
//       Vacant(ent) => {
//         ent.insert(array![root.clone()]);
//       }
//     }
//   }
// }

static mut find_equivalent_nodes_callcount: usize = 0;
static mut total_num_coincidental_nodes: usize = 0;
static mut find_equivalent_nodes_total_duration: Duration = Duration::new(0, 0);
fn find_equivalent_nodes(
  nag: &Nag,
  physical_mapping: &PhysicalNagMap,
  node: &NagNode,
) -> Vec<NagIndex> {
  let now = Instant::now();
  if let Some(coincidental_nodes) = physical_mapping.get(&node.physical_index) {
    unsafe {
      find_equivalent_nodes_callcount += 1;
      total_num_coincidental_nodes += coincidental_nodes.len();
    }
    let ret = coincidental_nodes
      .iter()
      .filter(|&&coincidental_node| {
        // If existing node, ignore self equivalence
        if node.nag_index.is_some() && coincidental_node == node.nag_index.unwrap() {
          return false;
        }
        if neighborhood_intersects(nag, &nag[coincidental_node], node) {
          return true;
        }
        false
      })
      .cloned()
      .collect();
    unsafe {
      find_equivalent_nodes_total_duration += now.elapsed();
    }
    return ret;
  } else {
    trace!("No coincidental nodes for {:?}", node);
    Vec::new()
  }
}

static mut get_path_neighborhood_call_count: usize = 0;
static mut get_path_neighborhood_total_duration: Duration = Duration::new(0, 0);

pub fn get_path_neighborhood(nag: &Nag, node: &NagNode) -> HashSet<NagIndex> {
  let now = Instant::now();
  if let Some(nag_index) = node.nag_index {
    if neighborhood_cache.lock().unwrap().contains_key(&nag_index) {
      return neighborhood_cache.lock().unwrap()[&nag_index].clone();
    }
  }

  // node.neighborhood.clone()
  // let parents = if let Some(nag_index) = node.nag_index {
  //   nag.neighbors(nag_index).clone()
  // } else {
  //   if let Some(parent) = node.parent {
  //     HashSet::from_iter([parent])
  //   } else {
  //     EMPTY_SET.clone()
  //   }
  // };
  let parents = node.parents.clone();

  // let now = Instant::now();
  let path_neighborhood: HashSet<NagIndex> = parents
    .into_iter()
    .map(|parent| nag.neighbors(parent))
    .flatten()
    .cloned()
    .collect();

  unsafe {
    get_path_neighborhood_call_count += 1;
    get_path_neighborhood_total_duration += now.elapsed();
  }

  if let Some(nag_index) = node.nag_index {
    neighborhood_cache
      .lock()
      .unwrap()
      .insert(nag_index, path_neighborhood.clone());
  }

  path_neighborhood
}

/// Using this version means we no longer need to keep track of NagNode.neighborhood
/// Average neighborhood size is around 300
fn neighborhood_intersects(nag: &Nag, a: &NagNode, b: &NagNode) -> bool {
  let a_neighborhood = get_path_neighborhood(nag, a);
  let b_neighborhood = get_path_neighborhood(nag, b);
  trace!("{:?} neighborhood: {:?}", a, a_neighborhood);
  trace!("{:?} neighborhood: {:?}", b, b_neighborhood);
  // let adjacency_matrix = &nag.adjacency_matrix();
  // any(
  //   a_neighborhood
  //     .iter()
  //     .cartesian_product(b_neighborhood.iter()),
  //   |(&a_neighbor, &b_neighbor)| {
  //     trace!(
  //       "{:?}, {:?} is adjacent: {}",
  //       a_neighbor,
  //       b_neighbor,
  //       nag.is_adjacent(adjacency_matrix, a_neighbor, b_neighbor)
  //     );
  //     nag.is_adjacent(adjacency_matrix, a_neighbor, b_neighbor)
  //   },
  // )
  !a_neighborhood.is_disjoint(&b_neighborhood)
}

/* DANGER! Do NOT USE!  This will ruin parent tracking */
// fn combine_nodes(nag: &mut Nag, target: &mut NagNode, node_indices: &Vec<NagIndex>) {
//   for &node_index in node_indices {
//     let node = &nag[node_index];
//     if node.value < target.value {
//       target.value = node.value;
//       target.parent = Some(node.parent.unwrap());
//     }
//     let mut edges_to_add = Vec::new();
//     for &neighbor_index in nag.neighbors(node_index) {
//       edges_to_add.push((target.nag_index.unwrap(), neighbor_index));
//     }
//     for edge_to_add in edges_to_add {
//       nag.add_edge(edge_to_add.0, edge_to_add.1);
//     }
//     nag.remove_node(node_index);
//   }
// }

fn contains(nag: &Nag, set: &Vec<NagIndex>, node: &NagNode) -> bool {
  for &item in set {
    if neighborhood_intersects(nag, &nag[item], node) {
      return true;
    }
  }
  false
}

fn try_handle_equivalent_nodes(
  nag: &mut Nag,
  visited_physical_mapping: &PhysicalNagMap,
  tentative_node: &NagNode,
) -> bool {
  let equivalent_nodes = find_equivalent_nodes(&nag, &visited_physical_mapping, &tentative_node);
  if !equivalent_nodes.is_empty() {
    trace!(
      "{:?} is not a new node, equivalent nodes are {:?}",
      tentative_node,
      equivalent_nodes
        .iter()
        .map(|x| &nag[*x])
        .collect::<Vec<&NagNode>>()
    );
    for equivalent_node in equivalent_nodes {
      let parent = tentative_node.parent.unwrap();
      nag.add_edge(equivalent_node, parent)
    }
    return true;
  }
  false
}

fn handle_nonequivalent_nodes(
  graph: &Graph,
  mut nag: &mut Nag,
  visited_physical_mapping: &mut PhysicalNagMap,
  visit_next: &mut BinaryHeap<NagNode>,
  tentative_node: NagNode,
) -> NagIndex {
  let nag_index = tentative_node.add_to_nag(&mut nag);
  debug!("Visiting new node {:?}", nag[nag_index]);
  match visited_physical_mapping.entry(nag[nag_index].physical_index) {
    Occupied(mut ent) => {
      ent.get_mut().push(nag_index);
    }
    Vacant(ent) => {
      let mut buffer = CoincidentalNags::new();
      buffer.push(nag_index);
      ent.insert(buffer);
    }
  }
  /* Root node doesn't have parent, skip adding edge */
  // if let Some(parent_index) = nag[nag_index].parent {
  //   trace!(
  //     "Adding nag edge (child): {}, {}",
  //     nag[nag_index],
  //     nag[parent_index]
  //   );
  //   nag_add_edge(graph, &mut nag, nag_index, parent_index);
  // } else {
  //   info!("This is a root node, this should only happen once!");
  // }

  let nag_node = &nag[nag_index];

  trace!(
    "Neighbors are (PHY): {:?}",
    graph
      .neighbors(nag_node.physical_index)
      .collect::<Vec<PhysicalIndex>>()
  );
  let mut tentative_children = Vec::new();
  for (_physical_index, adjacent_physical_index, &edge_weight) in
    graph.edges(nag_node.physical_index)
  {
    assert!(_physical_index == nag_node.physical_index);
    let nag_node = &nag[nag_index];
    let child = nag_node.create_child(graph, &nag, adjacent_physical_index, edge_weight);
    tentative_children.push(child);
  }
  while tentative_children.len() > 0 {
    let mut is_inserted_equivalent = false;
    tentative_children.retain(|tentative_child| {
      if try_handle_equivalent_nodes(&mut nag, &visited_physical_mapping, &tentative_child) {
        is_inserted_equivalent = true;
        return false; // remove element
      }
      true // keep element
    });
    if !is_inserted_equivalent {
      break;
    }
  }
  while let Some(child) = tentative_children.pop() {
    debug!(
      "New child {:?} is not equivalent at the moment, adding...",
      child
    );
    visit_next.push(child);
  }

  nag_index
}

fn heuristic(graph: Graph, p: PhysicalIndex, goal: &HashSet<PhysicalIndex>) -> f32 {
  goal.iter().fold(f32::INFINITY, |min_dist, &g| {
    min_dist.min(dist(&graph[p], &graph[g]))
  })
}

pub fn nag_dijkstra(
  graph: &Graph,
  start: &HashSet<PhysicalIndex>,
  goal: &HashSet<PhysicalIndex>,
  num_paths: usize,
  mut visualize: impl FnMut(&Nag),
  vis_freq: usize,
) -> (Nag, Vec<NagIndex>) {
  info!("Start: {:?}", start);
  info!("Goal: {:?}", goal);

  let start_time = Instant::now();
  let mut nag = Nag::with_capacity(graph.node_count() * (num_paths + 1));
  let mut nodes_visited = 0;
  /*
   Since our definition of equivalence is not transitive,
   a HashSet for recording visited nodes doesn't work.
   We must check the neighborhood of all previously visited nodes
  */
  let mut visited_physical_mapping: PhysicalNagMap = HashMap::new();
  let mut visit_next = BinaryHeap::new();
  let mut found_path_goals = Vec::new();
  for &physical_index in start {
    let root = NagNode::new(physical_index);
    // match visited_physical_mapping.entry(physical_index) {
    //   Occupied(ent) => {
    //     ent.into_mut().push(nag_index);
    //   }
    //   Vacant(ent) => {
    //     // println!("This shouldn't happen...");
    //     ent.insert(vec![nag_index]);
    //   }
    // }
    visit_next.push(root);
  }

  let mut now = Instant::now();
  while let Some(tentative_node) = visit_next.pop() {
    debug!("Visiting {:?}", tentative_node);
    nodes_visited += 1;
    /* Skip if equivalent */
    if !try_handle_equivalent_nodes(&mut nag, &visited_physical_mapping, &tentative_node) {
      let nag_index = handle_nonequivalent_nodes(
        graph,
        &mut nag,
        &mut visited_physical_mapping,
        &mut visit_next,
        tentative_node,
      );

      let nag_node = &nag[nag_index];
      if goal.contains(&nag_node.physical_index) && !contains(&nag, &found_path_goals, &nag_node) {
        found_path_goals.push(nag_node.nag_index.unwrap());
        info!("Found {}/{} paths!", found_path_goals.len(), num_paths);
        if found_path_goals.len() >= num_paths {
          break;
        }
      }
    }
    if nodes_visited % vis_freq == 0 {
      info!(
        "{}/{} paths found. Another {} in {}ms. Total {} in {}s",
        found_path_goals.len(),
        num_paths,
        vis_freq,
        now.elapsed().as_millis(),
        nodes_visited,
        start_time.elapsed().as_secs(),
      );
      let (max_phy, max_nag_list) = visited_physical_mapping
        .iter()
        .reduce(|(mut max_phy, mut max_nag_list), (phy, nag_list)| {
          if max_nag_list.len() < nag_list.len() {
            max_nag_list = nag_list;
            max_phy = phy;
          }
          (max_phy, max_nag_list)
        })
        .unwrap();
      info!(
        "PHY({}) has {} distinct nodes: {:?}",
        max_phy.index(),
        max_nag_list.len(),
        max_nag_list
      );
      // for (phy, nag_list) in &visited_physical_mapping {
      //   if nag_list.len() > 1 {
      //     info!("{:?} has nag nodes: {:?}", phy, nag_list);
      //   }
      // }
      unsafe {
        // println!(
        //   "get_path_neighborhood called {} times",
        //   get_path_neighborhood_call_count
        // );
        // println!(
        //   "get_path_neighborhood average duration: {}us",
        //   get_path_neighborhood_total_duration.as_micros() as f32
        //     / get_path_neighborhood_call_count as f32
        // );
        // println!(
        //   "get_path_neighborhood total duration: {}ms",
        //   get_path_neighborhood_total_duration.as_millis()
        // );
        get_path_neighborhood_call_count = 0;
        get_path_neighborhood_total_duration = Duration::new(0, 0);
        println!(
          "# find_equivalent_nodes calls: (total) {}, (per) {}",
          find_equivalent_nodes_callcount,
          find_equivalent_nodes_callcount as f32 / vis_freq as f32
        );
        // println!(
        //   "Average num coincidental_nodes = {}",
        //   total_num_coincidental_nodes as f32
        //     / find_equivalent_nodes_callcount as f32
        // );
        println!(
          "find_equivalent_nodes duration: (total) {}ms, (per) {}ms",
          find_equivalent_nodes_total_duration.as_millis(),
          find_equivalent_nodes_total_duration.as_millis() as f32
            / find_equivalent_nodes_callcount as f32
        );
        total_num_coincidental_nodes = 0;
        find_equivalent_nodes_callcount = 0;
        find_equivalent_nodes_total_duration = Duration::new(0, 0);
      }
      visualize(&nag);
      now = Instant::now();
    }
  }
  info!("Total nodes expanded: {}", nodes_visited);
  visualize(&nag);
  (nag, found_path_goals)
}
