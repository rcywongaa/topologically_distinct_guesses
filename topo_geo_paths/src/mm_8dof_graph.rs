use std::cmp::min;

use ahash::{HashMap, HashMapExt, HashSet, HashSetExt};
use bimap::BiHashMap;
use itertools::Itertools;
use log::info;
use ndarray::Array;
use ndarray_linalg::normalize;
use nshare::{ToNalgebra, ToNdarray1};
use ordered_float::NotNan;
use petgraph::{
  graph::NodeIndex,
  visit::{IntoEdgeReferences, IntoNeighbors, IntoNodeIdentifiers, IntoNodeReferences},
};

use crate::{
  graph_utils::{is_point_edge_valid, is_pose_edge_valid, is_pose_valid},
  grid_graph::create_grid_graph,
  kinematics::calc_elbow_position,
  nag_dijkstra::{Graph, PhysicalIndex},
  not_nan_util::{dist, from_not_nan_1d, to_not_nan, to_not_nan_1d, NotNan1D},
};

// fn try_update_edge(
//   graph: &mut Graph,
//   v1: Option<&NodeIndex>,
//   v2: Option<&NodeIndex>,
//   edge_func: &impl Fn(&NotNan1D, &NotNan1D) -> Option<f32>,
// ) {
//   if v1.is_some() && v2.is_some() {
//     let &v1 = v1.unwrap();
//     let &v2 = v2.unwrap();
//     if let Some(weight) = edge_func(&graph[v1], &graph[v2]) {
//       graph.update_edge(v1, v2, weight);
//     }
//   }
// }

// fn strong_product(
//   graph1: &Graph,
//   graph2: &Graph,
//   weight_func: &impl Fn(&NotNan1D, &NotNan1D) -> Option<NotNan1D>,
//   edge_func: &impl Fn(&NotNan1D, &NotNan1D) -> Option<f32>,
// ) -> Graph {
//   let mut graph = Graph::default();
//   let mut combined_to_separated = BiHashMap::new();
//   for ((index1, weight1), (index2, weight2)) in graph1
//     .node_references()
//     .cartesian_product(graph2.node_references())
//   {
//     // let weight = Array::from([weight1.to_vec(), weight2.to_vec()].concat());
//     if let Some(weight) = weight_func(weight1, weight2) {
//       let index = graph.add_node(weight);
//       graph.add_edge(index, index, 0.0); // Add self-edge so it shows up in neighbors()
//       combined_to_separated.insert(index, (index1, index2));
//     }
//   }
//   for (index1, index2) in graph1
//     .node_identifiers()
//     .cartesian_product(graph2.node_identifiers())
//   {
//     for (neighbor1, neighbor2) in graph1
//       .neighbors(index1)
//       .cartesian_product(graph2.neighbors(index2))
//     {
//       try_update_edge(
//         &mut graph,
//         combined_to_separated.get_by_right(&(index1, index2)),
//         combined_to_separated.get_by_right(&(neighbor1, neighbor2)),
//         edge_func,
//       );
//     }
//   }
//   graph
// }

fn combine_graphs(
  graph1: Graph,
  graph2: Graph,
) -> (Graph, BiHashMap<PhysicalIndex, PhysicalIndex>) {
  // let mut graph1_to_merged_index = BiHashMap::new();
  let mut original_to_merged_index = BiHashMap::new();
  let mut graph = graph1;
  /* Do not add original as it violates bijective mapping */
  // for index in graph.node_identifiers() {
  //   original_to_merged_index.insert(index, index);
  // }

  for (index, weight) in graph2.node_references() {
    let new_index = graph.add_node(weight.clone());
    original_to_merged_index.insert(index, new_index);
  }
  for (idx1, idx2, &edge_weight) in graph2.edge_references() {
    graph.add_edge(
      *original_to_merged_index.get_by_left(&idx1).unwrap(),
      *original_to_merged_index.get_by_left(&idx2).unwrap(),
      edge_weight,
    );
  }
  (graph, original_to_merged_index)
}

// fn create_path_graph<'a>(
//   eef_points: impl IntoIterator<Item = &'a [f32; 3]>,
//   edge_func: &impl Fn(&NotNan1D, &NotNan1D) -> Option<f32>,
// ) -> Graph {
//   let mut graph = Graph::default();
//   let mut node_indices = Vec::new();
//   for eef_point in eef_points {
//     let last_index = node_indices.last();
//     let node_index = graph.add_node(to_not_nan_1d(eef_point));
//     if let Some(&last_index) = last_index {
//       if let Some(edge_weight) = edge_func(&graph[node_index], &graph[last_index]) {
//         graph.add_edge(node_index, last_index, edge_weight);
//       }
//     }
//     node_indices.push(node_index);
//   }
//   graph
// }

/**
 * Merges v2 into v1
 */
fn merge_vertex(graph: &mut Graph, v1: NodeIndex, v2: NodeIndex) {
  for (_source, target, weight) in graph
    .edges(v2)
    .map(|(a, b, c)| (a, b, *c)) // dereference weight
    .collect::<Vec<(NodeIndex, NodeIndex, NotNan<f32>)>>()
  {
    assert!(_source == v2);
    graph.update_edge(v1, target, weight);
  }
  graph.remove_node(v2);
}

const EPSILON: f32 = 1e-6;

fn is_within(center: &[f32; 3], radius: f32, p: &[f32; 3]) -> bool {
  (p[0] - center[0]).powf(2.0) + (p[1] - center[1]).powf(2.0) + (p[2] - center[2]).powf(2.0)
    <= radius.powf(2.0)
}

fn is_boundary(center: &[f32; 3], radius: f32, p: &[f32; 3], step_size: f32) -> bool {
  !is_within(center, radius, &[p[0], p[1] - step_size, p[2]])
    || !is_within(center, radius, &[p[0], p[1] + step_size, p[2]])
    || !is_within(center, radius, &[p[0] - step_size, p[1], p[2]])
    || !is_within(center, radius, &[p[0] + step_size, p[1], p[2]])
}

fn create_single_eef_point_graph(
  forearm_length: f32,
  upperarm_length: f32,
  base_height: f32,
  lb: [f32; 2],
  ub: [f32; 2],
  eef: [f32; 3],
  base_step_size: f32,
  arm_link_obs_func: &impl Fn([f32; 3], [f32; 3]) -> bool,
  base_obs_func: &impl Fn([f32; 3]) -> bool,
) -> (Graph, BiHashMap<(usize, usize, i8), PhysicalIndex>) {
  let full_arm_length = forearm_length + upperarm_length;
  let boundary_func =
    |base_point: &[f32; 3]| is_boundary(&eef, full_arm_length, base_point, base_step_size);
  let mut boundary = HashSet::new();
  let mut vertex_func = |graph: &mut Graph, base: [f32; 2], is_up: bool| {
    let base = [base[0], base[1], base_height];
    let Some(elbow) = calc_elbow_position(forearm_length, upperarm_length, &eef, &base, is_up)
    else {
      return None;
    };
    let weight = to_not_nan_1d(&[
      elbow[0], elbow[1], elbow[2], eef[0], eef[1], eef[2], base[0], base[1], base[2],
    ]);
    if !is_pose_valid(&weight, &arm_link_obs_func, &base_obs_func) {
      return None;
    }
    let node_index = graph.add_node(weight.clone());

    /* Only collect boundary points for up case, down case would be the same */
    if is_up && boundary_func(&[base[0], base[1], base[2]]) {
      boundary.insert(node_index);
    }
    Some(node_index)
  };

  let mut edge_func = |a: &NotNan1D, b: &NotNan1D| {
    if !is_pose_edge_valid(&a, &b, &arm_link_obs_func, &base_obs_func, 0.01) {
      return None;
    }
    Some(dist(&a, &b))
  };
  let base_x_discretization = ((ub[0] - lb[0]) / base_step_size).ceil() as usize;
  let base_y_discretization = ((ub[1] - lb[1]) / base_step_size).ceil() as usize;
  let (elbow_up_graph, elbow_up_graph_pos_indices) = create_grid_graph(
    lb,
    ub,
    base_x_discretization,
    base_y_discretization,
    &mut |graph: &mut Graph, p: [f32; 2]| vertex_func(graph, p, true),
    &mut edge_func,
  );

  let (elbow_down_graph, mut elbow_down_graph_pos_indices) = create_grid_graph(
    lb,
    ub,
    base_x_discretization,
    base_y_discretization,
    &mut |graph: &mut Graph, p: [f32; 2]| vertex_func(graph, p, false),
    &mut edge_func,
  );

  let (mut merged_graph, elbow_down_graph_old_new_idx) =
    combine_graphs(elbow_up_graph, elbow_down_graph);

  /* Merge boundary vertices */
  for &elbow_up_graph_boundary_vertex in &boundary {
    let boundary_index = elbow_up_graph_pos_indices
      .get_by_right(&elbow_up_graph_boundary_vertex)
      .unwrap();
    let Some(&old_elbow_down_graph_boundary_vertex) =
      elbow_down_graph_pos_indices.get_by_left(boundary_index)
    else {
      continue;
    };
    merge_vertex(
      &mut merged_graph,
      elbow_up_graph_boundary_vertex,
      *elbow_down_graph_old_new_idx
        .get_by_left(&old_elbow_down_graph_boundary_vertex)
        .unwrap(),
    );
    elbow_down_graph_pos_indices.remove_by_right(&old_elbow_down_graph_boundary_vertex);

    /* This may not be a good idea since it would create non-bridged 4-cycles */
    // merged_graph.add_edge(
    //   elbow_up_graph_boundary_vertex,
    //   *elbow_down_graph_old_new_idx
    //     .get_by_left(&old_elbow_down_graph_boundary_vertex)
    //     .unwrap(),
    //   to_not_nan(0.001),
    // );
  }

  let mut pos_idx_to_node_idx = BiHashMap::new();
  for (pos_idx, node_idx) in elbow_up_graph_pos_indices {
    let mut is_up_marker: i8 = 1;
    if boundary.contains(&node_idx) {
      is_up_marker = 0;
    }
    pos_idx_to_node_idx.insert((pos_idx.0, pos_idx.1, is_up_marker), node_idx);
  }
  for (pos_idx, node_idx) in elbow_down_graph_pos_indices {
    pos_idx_to_node_idx.insert(
      (pos_idx.0, pos_idx.1, -1),
      *elbow_down_graph_old_new_idx.get_by_left(&node_idx).unwrap(),
    );
  }
  (merged_graph, pos_idx_to_node_idx)
}

pub fn create_mm_6dof_arm_graph<'a>(
  forearm_length: f32,
  upperarm_length: f32,
  base_height: f32,
  base_lb: [f32; 2],
  base_ub: [f32; 2],
  base_resolution: f32,
  eef_points: impl IntoIterator<Item = &'a [f32; 3]>,
  line_obs_func: &impl Fn([f32; 3], [f32; 3]) -> bool,
  base_obs_func: &impl Fn([f32; 3]) -> bool,
) -> Graph {
  let edge_func = |a: &NotNan1D, b: &NotNan1D| {
    if !is_pose_edge_valid(&a, &b, line_obs_func, &base_obs_func, 0.01) {
      return None;
    }
    Some(dist(&a, &b))
  };

  let mut merged_graph_option = None;
  let mut last_pos_idx_to_node_idx_option = None;
  let mut last_original_to_merged_idx_option: Option<BiHashMap<PhysicalIndex, PhysicalIndex>> =
    None;
  for &eef_point in eef_points {
    let (single_point_graph, pos_idx_to_node_idx) = create_single_eef_point_graph(
      forearm_length,
      upperarm_length,
      base_height,
      base_lb,
      base_ub,
      eef_point,
      base_resolution,
      &line_obs_func,
      &base_obs_func,
    );

    /* First run */
    let Some(merged_graph) = merged_graph_option else {
      merged_graph_option = Some(single_point_graph);
      last_pos_idx_to_node_idx_option = Some(pos_idx_to_node_idx);
      continue;
    };

    let (mut new_graph, original_to_merged_index) =
      combine_graphs(merged_graph, single_point_graph);

    let last_pos_idx_to_node_idx = last_pos_idx_to_node_idx_option.unwrap();
    /* Update last_pos_idx_to_node_idx_option */

    for (&pos_idx, &node_idx) in &pos_idx_to_node_idx {
      let adjacent_pos_idcs = [
        (pos_idx.0 - 1, pos_idx.1 - 1, pos_idx.2),
        (pos_idx.0 - 1, pos_idx.1, pos_idx.2),
        (pos_idx.0 - 1, pos_idx.1 + 1, pos_idx.2),
        (pos_idx.0, pos_idx.1 - 1, pos_idx.2),
        (pos_idx.0, pos_idx.1, pos_idx.2),
        (pos_idx.0, pos_idx.1 + 1, pos_idx.2),
        (pos_idx.0 + 1, pos_idx.1 - 1, pos_idx.2),
        (pos_idx.0 + 1, pos_idx.1, pos_idx.2),
        (pos_idx.0 + 1, pos_idx.1 + 1, pos_idx.2),
      ];
      for adjacent_pos_idx in adjacent_pos_idcs {
        let Some(&(mut last_adjacent_pos_node_idx)) =
          last_pos_idx_to_node_idx.get_by_left(&adjacent_pos_idx)
        else {
          continue;
        };

        if let Some(ref last_original_to_merged_idx) = last_original_to_merged_idx_option {
          last_adjacent_pos_node_idx = *last_original_to_merged_idx
            .get_by_left(&last_adjacent_pos_node_idx)
            .unwrap();
        }

        let &merged_index = original_to_merged_index.get_by_left(&node_idx).unwrap();
        // let &last_adjacent_merged_index = original_to_merged_index
        //   .get_by_left(&last_adjacent_pos_node_idx)
        //   .unwrap();
        let Some(edge_weight) = edge_func(
          &new_graph[merged_index],
          &new_graph[last_adjacent_pos_node_idx],
        ) else {
          continue;
        };

        new_graph.update_edge(
          last_adjacent_pos_node_idx,
          merged_index,
          NotNan::new(edge_weight).unwrap(),
        );
      }
    }
    last_pos_idx_to_node_idx_option = Some(pos_idx_to_node_idx);
    merged_graph_option = Some(new_graph);
    last_original_to_merged_idx_option = Some(original_to_merged_index);
  }

  merged_graph_option.unwrap()
}
