use std::f32::consts::PI;

use ahash::{HashMap, HashMapExt};
use bimap::BiHashMap;
use itertools::min;
use log::info;
use ndarray::s;
use num_traits::ToPrimitive;
use petgraph::{graph::NodeIndex, visit::IntoNodeReferences};

use crate::{
  nag_dijkstra::Graph,
  not_nan_util::{from_not_nan_1d, to_not_nan_1d},
  swept_sphere_graph::create_swept_sphere_graph,
};

const EPSILON: f32 = 0.05;

// Should have stable indices
// Hence do not check obstacles here
fn calc_base_position(
  elbow_position: [f32; 3],
  upperarm_length: f32,
  discretization: usize,
  n: usize,
) -> Option<[f32; 3]> {
  if elbow_position[2] <= 0.0 {
    return None;
  }
  let theta = 2.0 * PI / discretization.to_f32().unwrap() * n.to_f32().unwrap();
  let r = (upperarm_length.powi(2) - elbow_position[2].powi(2)).sqrt();
  if r.is_nan() || r < EPSILON {
    return None;
  }
  Some([
    elbow_position[0] + r * f32::sin(theta),
    elbow_position[1] + r * f32::cos(theta),
    0.0,
  ])
}

fn base_position_to_grid(base_position: [f32; 3], cell_size: f32, lb: [f32; 2]) -> (usize, usize) {
  let x = base_position[0] - lb[0];
  let y = base_position[1] - lb[1];
  assert!(x >= 0.0, "{} not positive!", x);
  assert!(y >= 0.0, "{} not positive!", y);
  (
    (x / cell_size).floor() as usize,
    (y / cell_size).floor() as usize,
  )
}

fn grid_to_base_position(grid_position: (usize, usize), cell_size: f32, lb: [f32; 2]) -> [f32; 3] {
  [
    (grid_position.0 as f32) * cell_size + lb[0],
    (grid_position.1 as f32) * cell_size + lb[1],
    0.0,
  ]
}

/**
 * Generates a two link arm graph where each vertex is
 * [f32; 8] = [elbow_x, elbow_y, elbow_z, eef_x, eef_y, eef_z, base_x, base_y]
 */
pub fn create_two_link_arm_graph<'a>(
  end_effector_path: impl IntoIterator<Item = &'a [f32; 3]>,
  forearm_length: f32,
  upperarm_length: f32,
  subdivisions: usize,
  base_discretization: usize,
  obs_func: impl Fn([f32; 3], [f32; 3]) -> bool,
) -> (
  HashMap<NodeIndex, [f32; 8]>,
  Graph,
  BiHashMap<((usize, usize), (usize, usize)), NodeIndex>,
) {
  // let min_end_effector_position = end_effector_path.into_iter()
  /* FIXME: This doesn't take into account the end effector position */
  let grid_lb = [
    // -(forearm_length + upperarm_length) * 1.1,
    // -(forearm_length + upperarm_length) * 1.1,
    -2.0, -2.0,
  ];
  info!("Using grid lowerbound: {:?}", grid_lb);
  // let cell_size = ((forearm_length + upperarm_length) / base_discretization as f32) / 2.0;
  // info!("Using base grid cell size: {}", cell_size);
  let cell_size = 0.1;

  let mut arm_graph = Graph::default();
  let mut position_grid_index_to_node_index = BiHashMap::new();
  let mut position_index_to_position_grid_index = HashMap::new();
  let mut arm_node_index_to_actual_positions = HashMap::new();
  let (ss_node_index_map, ss_graph, ss_position_to_node) = create_swept_sphere_graph(
    end_effector_path,
    forearm_length,
    subdivisions,
    &obs_func,
    0.01,
  );

  for (ss_index, ss_node) in ss_graph.node_references() {
    let elbow_position = from_not_nan_1d(&ss_node.slice(s![0..3]).to_owned());
    let &ss_position_index = ss_position_to_node.get_by_right(&ss_index).unwrap();

    // Create nodes
    for base_position_index in 0..base_discretization {
      let base_position = calc_base_position(
        elbow_position,
        upperarm_length,
        base_discretization,
        base_position_index,
      );
      if base_position.is_none() {
        continue;
      }

      let base_position = base_position.unwrap();

      if obs_func(base_position, elbow_position) {
        continue;
      }

      let base_grid_position = base_position_to_grid(base_position, cell_size, grid_lb);
      let binned_base_position = grid_to_base_position(base_grid_position, cell_size, grid_lb);

      let arm_position = [
        ss_node[0].to_f32().unwrap(),
        ss_node[1].to_f32().unwrap(),
        ss_node[2].to_f32().unwrap(), // elbow position
        ss_node[3].to_f32().unwrap(),
        ss_node[4].to_f32().unwrap(),
        ss_node[5].to_f32().unwrap(), // end effector position
        binned_base_position[0],
        binned_base_position[1],
      ];

      let arm_position_index = (ss_position_index, base_position_index);
      let arm_position_grid_index = (ss_position_index, base_grid_position);
      position_index_to_position_grid_index.insert(arm_position_index, arm_position_grid_index);

      /* Only add node if new position after binning */
      if position_grid_index_to_node_index.contains_left(&arm_position_grid_index) {
        continue;
      }

      let arm_index = arm_graph.add_node(to_not_nan_1d(&arm_position));
      position_grid_index_to_node_index.insert(arm_position_grid_index, arm_index);
      arm_node_index_to_actual_positions.insert(arm_index, arm_position);
    }
  }

  // Add edges
  for (&arm_position_index, &arm_position_grid_index) in &position_index_to_position_grid_index {
    /* This is base_position_grid_index */
    let (ss_position_index, base_position_index) = arm_position_index;
    let mut neighbor_base_position_indices = vec![base_position_index];
    if base_position_index > 0 {
      neighbor_base_position_indices.push(base_position_index - 1);
    }
    // If last position on the circle, also add first position as neighbor
    if base_position_index == base_discretization - 1 {
      neighbor_base_position_indices.push(0);
    }

    let &node_index = position_grid_index_to_node_index
      .get_by_left(&arm_position_grid_index)
      .unwrap();

    for neighbor_base_position_index in neighbor_base_position_indices {
      let neighbor_grid_index = position_index_to_position_grid_index
        .get(&(ss_position_index, neighbor_base_position_index));
      if neighbor_grid_index.is_none() {
        continue;
      }
      let &neighbor_grid_index = neighbor_grid_index.unwrap();

      {
        let &neighbor_node_index = position_grid_index_to_node_index
          .get_by_left(&neighbor_grid_index)
          .unwrap();
        // Do not add self edge
        if neighbor_node_index != node_index {
          // Add edges between current base positions
          arm_graph.update_edge(node_index, neighbor_node_index, ());
        }
      }

      // Add edges between base position circles
      let &ss_index = ss_position_to_node.get_by_left(&ss_position_index).unwrap();
      for neighbor in ss_graph.neighbors(ss_index) {
        let &neighbor_ss_position = ss_position_to_node.get_by_right(&neighbor).unwrap();
        let neighbor_arm_position = (neighbor_ss_position, neighbor_base_position_index);
        if let Some(&neighbor_grid_index) =
          position_index_to_position_grid_index.get(&neighbor_arm_position)
        {
          let &neighbor_arm_index = position_grid_index_to_node_index
            .get_by_left(&neighbor_grid_index)
            .unwrap();
          /* Ensure that the end effector position changes at most 1 */
          assert!(usize::abs_diff(neighbor_ss_position.1, arm_position_index.0 .1) <= 1);
          if neighbor_arm_index != node_index {
            arm_graph.update_edge(node_index, neighbor_arm_index, ());
          }
        }
      }
    }
  }
  (
    arm_node_index_to_actual_positions,
    arm_graph,
    position_grid_index_to_node_index,
  )
}
