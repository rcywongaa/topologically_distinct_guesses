#![allow(dead_code)]
use ahash::{HashMap, HashMapExt};
use itertools::Itertools;
use ordered_float::NotNan;
use petgraph::visit::IntoNodeReferences;
use std::f32::INFINITY;

use ndarray::{array, concatenate, s, Array, Axis};
use ndarray_linalg::Norm;

use crate::{
  nag_dijkstra::{Graph, GraphNode, Nag, NagIndex, NagNode, PhysicalIndex},
  not_nan_util::{dist, from_not_nan_1d, to_not_nan_1d, NotNan1D},
};

pub fn output_trajectory(graph: &Graph, nag: &Nag, goal_index: &NagIndex) -> String {
  let path = get_path(nag, goal_index);
  let mut trajectory = String::new();
  for node in path {
    let graph_node = &graph[node.physical_index];
    trajectory.push_str(&format!("{}\n", graph_node));
  }
  trajectory
}

pub fn get_path<'a>(nag: &'a Nag, goal_index: &NagIndex) -> Vec<&'a NagNode> {
  let mut path = Vec::new();
  let mut nag_index = Some(goal_index);
  while nag_index.is_some() {
    let node = &nag[*nag_index.unwrap()];
    path.push(node);
    nag_index = node.parent.as_ref();
  }
  path.reverse(); // Vector goes from goal -> start, reverse it
  path // path from start -> goal
}

pub fn get_physical_path<'a>(graph: &'a Graph, nag: &Nag, goal_index: &NagIndex) -> Vec<GraphNode> {
  get_path(nag, &goal_index)
    .iter()
    .map(|&x| graph[x.physical_index].clone())
    .collect()
}

pub fn print_path(nag: &Nag, goal_indices: &[NagIndex]) {
  for (idx, goal_index) in goal_indices.iter().enumerate() {
    print!("Path {}: ", idx);
    let path = get_path(nag, goal_index);
    print!("{:?}", path);
    println!();
  }
}

pub fn print_physical_path(path: &[GraphNode]) {
  for node in path {
    print!("({}) ", node);
  }
  println!();
}

pub fn find_nearest_node_index<T>(
  node_index_to_position: &HashMap<PhysicalIndex, T>,
  position: T,
) -> PhysicalIndex
where
  T: IntoIterator<Item = f32> + Copy,
{
  let mut nearest_index = PhysicalIndex::default();
  let mut nearest_distance = INFINITY;
  for (index, node_position) in node_index_to_position {
    let distance = (Array::from_iter(position.into_iter())
      - Array::from_iter(node_position.into_iter()))
    .norm_l2();
    if distance < nearest_distance {
      nearest_index = *index;
      nearest_distance = distance;
    }
  }
  nearest_index
}

pub fn find_nearest_node_index_at_t<T>(
  node_index_to_position: &HashMap<PhysicalIndex, (usize, T)>,
  position: T,
  t: usize,
) -> PhysicalIndex
where
  T: IntoIterator<Item = f32> + Copy,
{
  let mut nearest_index = None;
  let mut nearest_distance = INFINITY;
  for (&index, &(node_t, node_position)) in node_index_to_position {
    if node_t != t {
      continue;
    }
    let distance = (Array::from_iter(position.into_iter())
      - Array::from_iter(node_position.into_iter()))
    .norm_l2();
    if distance < nearest_distance {
      nearest_index = Some(index);
      nearest_distance = distance;
    }
  }
  nearest_index.unwrap()
}

pub fn unpack_position(position: &NotNan1D) -> (NotNan1D, NotNan1D, NotNan1D) {
  let elbow_position = position.slice(s![0..3]).to_owned();
  let eef_position = position.slice(s![3..6]).to_owned();
  let base_position = if position.len() == 9 {
    position.slice(s![6..9]).to_owned()
  } else {
    concatenate(
      Axis(0),
      &[
        position.slice(s![6..8]).to_owned().view(),
        array![NotNan::try_from(0.0).unwrap()].view(),
      ],
    )
    .unwrap()
  };
  (elbow_position, eef_position, base_position)
}

pub fn is_point_edge_valid<const T: usize>(
  p1: &NotNan1D,
  p2: &NotNan1D,
  point_obs_intersect_func: impl Fn([f32; T]) -> bool,
  step_size: f32,
) -> bool {
  let dist = dist(&p1, &p2);
  let direction = p2 - p1;
  let num_steps = f32::ceil(dist / step_size) as usize;
  for i in 0..num_steps {
    let sample = p1 + &direction * (i as f32) / (num_steps as f32);
    if point_obs_intersect_func(from_not_nan_1d(&sample)) {
      return false;
    }
  }
  true
}

pub fn is_pose_valid(
  p: &NotNan1D,
  line_obs_intersect_func: impl Fn([f32; 3], [f32; 3]) -> bool,
  base_obs_intersect_func: impl Fn([f32; 3]) -> bool,
) -> bool {
  assert!(p.len() == 9);
  let (elbow_position, eef_position, base_position) = unpack_position(p);

  if base_obs_intersect_func([*base_position[0], *base_position[1], *base_position[2]]) {
    return false;
  }
  /* This is wrong because p is ordered by [elbow, eef, base] */
  // for (point1, point2) in p.exact_chunks(3).into_iter().tuple_windows::<(_, _)>() {
  //   if line_obs_intersect_func(
  //     from_not_nan_1d(&point1.to_owned()),
  //     from_not_nan_1d(&point2.to_owned()),
  //   ) {
  //     return false;
  //   }
  // }
  if line_obs_intersect_func(
    [*elbow_position[0], *elbow_position[1], *elbow_position[2]],
    [*eef_position[0], *eef_position[1], *eef_position[2]],
  ) || line_obs_intersect_func(
    [*elbow_position[0], *elbow_position[1], *elbow_position[2]],
    [*base_position[0], *base_position[1], *base_position[2]],
  ) {
    return false;
  }
  true
}

pub fn is_pose_edge_valid(
  p1: &NotNan1D,
  p2: &NotNan1D,
  line_obs_intersect_func: impl Fn([f32; 3], [f32; 3]) -> bool,
  base_obs_intersect_func: impl Fn([f32; 3]) -> bool,
  step_size: f32,
) -> bool {
  let dist = dist(&p1, &p2);
  let direction = p2 - p1;
  let num_steps = f32::ceil(dist / step_size) as usize;
  for i in 0..num_steps {
    let sample = p1 + &direction * (i as f32) / (num_steps as f32);
    if !is_pose_valid(&sample, &line_obs_intersect_func, &base_obs_intersect_func) {
      return false;
    }
  }
  true
}

pub fn generate_node_index_to_position_map<const T: usize>(
  graph: &Graph,
) -> HashMap<PhysicalIndex, [f32; T]> {
  graph
    .node_references()
    .fold(HashMap::new(), |mut acc, (node_index, position)| {
      acc.insert(node_index, from_not_nan_1d(position));
      acc
    })
}

pub fn default_vertex_func<const T: usize>(
  graph: &mut Graph,
  weight: [f32; T],
  obs_func: impl Fn([f32; T]) -> bool,
) -> Option<PhysicalIndex> {
  if obs_func(weight) {
    return None;
  }
  Some(graph.add_node(to_not_nan_1d(&weight)))
}

pub fn default_point_edge_func<const T: usize>(
  a: &NotNan1D,
  b: &NotNan1D,
  obs_func: impl Fn([f32; T]) -> bool,
) -> Option<f32> {
  if !is_point_edge_valid(&a, &b, &obs_func, 0.01) {
    return None;
  }
  Some(dist(&a, &b))
}

// pub fn default_pose_edge_func(
//   a: &NotNan1D,
//   b: &NotNan1D,
//   obs_func: impl Fn([f32; 3], [f32; 3]) -> bool,
// ) -> Option<f32> {
//   if !is_pose_edge_valid(&a, &b, &obs_func, 0.01) {
//     return None;
//   }
//   Some(dist(&a, &b))
// }
