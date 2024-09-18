#![allow(dead_code)]
use ahash::{HashMap, HashMapExt, HashSet, HashSetExt};

use bimap::BiHashMap;
use itertools::Itertools;
use ndarray::{Array, Ix1};
use ordered_float::NotNan;

use crate::{
  nag_dijkstra::{Graph, PhysicalIndex},
  not_nan_util::{to_not_nan_1d, NotNan1D},
};

fn get_adjacent_indices(x_idx: usize, y_idx: usize) -> Vec<(usize, usize)> {
  let mut ret = vec![
    (x_idx, y_idx + 1),
    (x_idx + 1, y_idx),
    (x_idx + 1, y_idx + 1),
  ];
  if x_idx > 0 {
    ret.push((x_idx - 1, y_idx));
    ret.push((x_idx - 1, y_idx + 1));
  }
  if y_idx > 0 {
    ret.push((x_idx, y_idx - 1));
    ret.push((x_idx + 1, y_idx - 1));
  }
  if x_idx > 0 && y_idx > 0 {
    ret.push((x_idx - 1, y_idx - 1));
  }
  ret
}

/**
 * vertex_func is responsible for processing the input node into a NotNan1D and inserting it into the Graph
 */
pub fn create_grid_graph(
  lb: [f32; 2],
  ub: [f32; 2],
  x_num_steps: usize,
  y_num_steps: usize,
  vertex_func: &mut impl FnMut(&mut Graph, [f32; 2]) -> Option<PhysicalIndex>,
  edge_func: &mut impl FnMut(&NotNan1D, &NotNan1D) -> Option<f32>,
) -> (Graph, BiHashMap<(usize, usize), PhysicalIndex>) {
  let xvals = Array::<f32, Ix1>::linspace(lb[0], ub[0], x_num_steps);
  let yvals = Array::<f32, Ix1>::linspace(lb[1], ub[1], y_num_steps);
  let mut pos_idx_to_node_idx = BiHashMap::new();
  let mut node_to_position = HashMap::new();
  let mut graph = Graph::default();

  for ((x_idx, &x_val), (y_idx, &y_val)) in xvals
    .iter()
    .enumerate()
    .cartesian_product(yvals.iter().enumerate())
  {
    if let Some(node_index) = vertex_func(&mut graph, [x_val, y_val]) {
      pos_idx_to_node_idx.insert((x_idx, y_idx), node_index);
      node_to_position.insert(node_index, [x_val, y_val]);
    }
  }

  for (x_idx, y_idx) in (0..xvals.len())
    .into_iter()
    .cartesian_product(0..yvals.len())
  {
    if !pos_idx_to_node_idx.contains_left(&(x_idx, y_idx)) {
      continue;
    }
    for adj_idx in get_adjacent_indices(x_idx, y_idx) {
      if pos_idx_to_node_idx.contains_left(&adj_idx) {
        let node_a = *pos_idx_to_node_idx.get_by_left(&(x_idx, y_idx)).unwrap();
        let node_b = *pos_idx_to_node_idx.get_by_left(&adj_idx).unwrap();
        if let Some(weight) = edge_func(&graph[node_a], &graph[node_b]) {
          // We may attempt to repeatedly add the same edge, ignore...
          graph.update_edge(node_a, node_b, NotNan::new(weight).unwrap());
          graph.update_edge(node_b, node_a, NotNan::new(weight).unwrap());
        }
      }
    }
  }
  // let mut file = File::create("graph.dot").unwrap();
  // file.write_all(format!("{:?}", Dot::new(&graph)).as_bytes()).unwrap();
  (graph, pos_idx_to_node_idx)
}
