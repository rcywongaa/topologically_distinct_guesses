// position & path_idx together constitutes the NodeId

use std::{collections::{HashSet, VecDeque}, rc::Rc, cmp::Ordering, hash::{Hash, Hasher}, ops::Index, fmt};
use ndarray::prelude::*;
use ndarray_linalg::Norm;
use ordered_float::NotNan;
use sorted_vec::SortedVec;
use petgraph::graph::NodeIndex;

// pub type Graph = petgraph::graph::UnGraph<Vec1D, ()>;
pub type Graph = petgraph::graph::UnGraph<NotNan1D, ()>;

// type Vec1D = Array::<f64, Ix1>;
type NotNan1D = Array::<NotNan<f64>, Ix1>;

fn dist(p1: &NotNan1D, p2: &NotNan1D) -> f64 {
  let p1_vec = p1.map(|x| x.into_inner());
  let p2_vec = p2.map(|x| x.into_inner());
  (p1_vec - p2_vec).norm_l2()
}

pub struct Path {
  nodes: VecDeque<NodeIndex>,
  cost: f64
}

struct VirtualCell {
  value: f64,
  parent: Option<Rc<VirtualCell>>,
  actual_node_idx: NodeIndex,
}

impl VirtualCell {
  fn new(node_idx: NodeIndex) -> Self {
    Self {
      actual_node_idx: node_idx,
      value: 0.0,
      parent: None,
    }
  }
  fn create_child(self: Rc<Self>, graph: &Graph, child_node_idx: NodeIndex) -> VirtualCell{
    let child = VirtualCell {
      value: self.value + dist(graph.index(self.actual_node_idx), graph.index(child_node_idx)),
      actual_node_idx: child_node_idx,
      parent: Some(self.clone()),
    };
    child
  }
}

impl Ord for VirtualCell {
  fn cmp(&self, other: &Self) -> Ordering {
    (NotNan::new(self.value).unwrap()).cmp(&NotNan::new(other.value).unwrap())
  }
}
impl PartialOrd for VirtualCell {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

// FIXME: This is wrong.  The same Eq should not be used for testing identity and equality of value
impl Eq for VirtualCell {}
impl PartialEq for VirtualCell {
  fn eq(&self, other: &Self) -> bool {
    // self.actual_node_idx == other.actual_node_idx
    self.value == other.value
  }
}

impl Hash for VirtualCell {
  fn hash<H: Hasher>(&self, state: &mut H)
  {
    self.actual_node_idx.hash(state);
  }
}

impl fmt::Display for VirtualCell {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      write!(f, "({:?})", self.actual_node_idx)
  }
}

impl fmt::Debug for VirtualCell {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "({:?})", self.actual_node_idx)
  }
}

fn is_adjacent(graph: &Graph, cell1: &VirtualCell, cell2: &VirtualCell) -> bool {
  cell1.actual_node_idx == cell2.actual_node_idx // Treat same cell as adjacent
  || graph.neighbors(cell1.actual_node_idx).find(|x| x == &cell2.actual_node_idx).is_some()
}

fn are_different_paths(graph: &Graph, cell1: &VirtualCell, cell2: &VirtualCell) -> bool {
  // check if the parents responsible for adding the cell-to-be-visited are adjacent
  if is_adjacent(&graph, &cell1.parent.as_ref().unwrap(), &cell2) || is_adjacent(&graph, &cell1, &cell2.parent.as_ref().unwrap()) {
    false
  } else {
    true
  }
}

fn to_simple_path(goal: &Rc<VirtualCell>) -> Path {
  let mut current_goal = goal;
  let mut cost = 0.0;
  let mut path = Vec::from([current_goal.actual_node_idx]);
  while let Some(cell) = &current_goal.parent {
    path.push(cell.actual_node_idx);
    cost += cell.value;
    current_goal = cell;
  }
  Path {
    nodes: VecDeque::from(path),
    cost: cost
  }
}

/**
 * Given a start and goal, find the shortest path to the goal
 * as well as paths to collision points
 */
pub fn dijkstra(graph: &Graph, start: &HashSet<NodeIndex>, goal: &HashSet<NodeIndex>) -> (Vec<Rc<VirtualCell>>, Vec<(Rc<VirtualCell>, Rc<VirtualCell>)>){
  let mut visited = HashSet::new();
  let mut visit_next = SortedVec::new();
  let mut merges = Vec::new();
  let mut goals = Vec::new();
  for &cell in start {
    let root = Rc::new(VirtualCell::new(cell));
    visit_next.push(root.clone());
  }
  while let Some(cell) = visit_next.pop() {
    println!("Visiting {}", cell);

    if visited.contains(&cell) {
      println!("Already visited {}, skipping...", cell);
      continue;
    }
    if goal.contains(&cell.actual_node_idx) {
      goals.push(cell.clone());
    }

    for adjacent_cell in graph.neighbors(cell.actual_node_idx) {
      let child = Rc::new(cell.clone().create_child(&graph, adjacent_cell.clone()));
      println!("Checking child {}...", child);
      if visited.contains(&child) {
        println!("Already visited child {}, skipping...", child);
        continue;
      }
      match visit_next.find_or_push(child.clone()) {
        sorted_vec::FindOrInsert::Found(idx) => {
          println!("{} already in wavefront", &child);
          // cell to be visited is already part of the wavefront
          let existing_cell = &visit_next[idx];
          if are_different_paths(&graph, &child, &existing_cell) {
            println!("Found merge!");
            merges.push((existing_cell.clone(), child.clone()));
          } else {
            if child.value < visit_next[idx].value {
              visit_next.remove_index(idx);
              visit_next.push(child.clone());
              println!("Found lower cost, replacing...");
            } else {
              println!("Dropping...");
            }
          }
        },
        sorted_vec::FindOrInsert::Inserted(_) => {
          println!("Adding new {} to wavefront", &child);
        }
      }
    }
    visited.insert(cell.clone());
    if visited.len() % 100 == 0 {
      println!("Visited {} cells", visited.len());
    }
  }
  (goals, merges)

}

pub fn topo_dijkstra(graph: &Graph, start: &HashSet<NodeIndex>, goal: &HashSet<NodeIndex>, num_detours: usize) -> Vec<Path> {
  /*
  1. Run normal dijkstra, keeping track of collisions (denoted as 1st collisions)
  2. For each collision, run dijkstra again and keep track of collisions (denoted as 2nd collisions)
  3. Repeat until the desired n-th collisions are found
  4. Form paths by considering every combination of start -- 1st collision -- ... -- i-th collision -- goal
     for all i < n
  5. Prune paths by checking if every pair of paths (which forms a cycle) contains an obstacle or not
     https://math.stackexchange.com/questions/692837/how-to-know-if-a-node-is-surrounded
   */
  let mut paths = Vec::new();
  let mut current_starts = Vec::from([start.clone()]);
  for _ in 0..num_detours {
    let mut detoured_start = Vec::new();
    while let Some(next_start) = current_starts.pop() {
      let (mut goals, merges) = dijkstra(graph, &next_start, goal);
      let mut goal_paths = goals.iter().map(|x| to_simple_path(x)).collect();
      let mut merge_paths: Vec<(Path, Path)> = merges.iter().map(|(p1, p2)| (to_simple_path(p1), to_simple_path(p2))).collect();
      paths.append(&mut goal_paths);
      // TODO: Add merge points into detoured_start
      detoured_start.append(&mut merges.into_iter().map(|(p1, p2)| -> HashSet<NodeIndex> {
        assert!(p1.actual_node_idx == p2.actual_node_idx);
        HashSet::from([p1.actual_node_idx])
      }).collect::<Vec<HashSet<NodeIndex>>>());
    }
    current_starts = detoured_start;
  }
  paths
}
