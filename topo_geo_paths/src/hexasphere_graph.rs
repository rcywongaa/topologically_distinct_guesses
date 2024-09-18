use ahash::{HashMap, HashMapExt};

use bimap::BiHashMap;
use hexasphere::shapes::IcoSphere;

use crate::{
  graph_utils::is_point_edge_valid,
  nag_dijkstra::{Graph, PhysicalIndex},
  not_nan_util::{dist, from_not_nan_1d, to_not_nan_1d, NotNan1D},
};

pub fn max_edge_length(radius: f32, subdivisions: usize) -> f32 {
  let sphere = IcoSphere::new(subdivisions, |_| ());
  let mut max_edge_length = 0.0;
  for triangle in sphere.get_all_indices().chunks(3) {
    max_edge_length = f32::max(
      max_edge_length,
      sphere.linear_distance(triangle[0], triangle[1], radius),
    );
    max_edge_length = f32::max(
      max_edge_length,
      sphere.linear_distance(triangle[0], triangle[2], radius),
    );
    max_edge_length = f32::max(
      max_edge_length,
      sphere.linear_distance(triangle[1], triangle[2], radius),
    );
  }
  max_edge_length
}

/// position_index_to_node_index is stable
/// meaning the same point_index always refers to the same hexasphere position
/// provided the same subdivision is used
pub fn create_hexasphere_graph(
  center: [f32; 3],
  radius: f32,
  subdivisions: usize,
  obs_func: &impl Fn([f32; 3]) -> bool,
  collision_check_step_size: f32,
) -> (
  HashMap<PhysicalIndex, [f32; 3]>,
  Graph,
  BiHashMap<usize, PhysicalIndex>,
) {
  let sphere = IcoSphere::new(subdivisions, |_| ());

  let mut graph = Graph::default();
  let mut position_index_to_node_index = BiHashMap::new();
  let mut node_index_to_position = HashMap::new(); // Cannot use BiHashMap since f32 does not implement Eq
  for (point_index, point) in sphere.raw_points().iter().enumerate() {
    let x = point.x * radius + center[0];
    let y = point.y * radius + center[1];
    let z = point.z * radius + center[2];
    if obs_func([x, y, z]) {
      continue;
    }
    let node_index = graph.add_node(to_not_nan_1d(&[x, y, z]));
    position_index_to_node_index.insert(point_index, node_index);
    node_index_to_position.insert(node_index, [x, y, z]);
  }
  for triangle in sphere.get_all_indices().chunks(3) {
    let index0 = usize::try_from(triangle[0]).unwrap();
    let index1 = usize::try_from(triangle[1]).unwrap();
    let index2 = usize::try_from(triangle[2]).unwrap();
    if !position_index_to_node_index.contains_left(&index0)
      || !position_index_to_node_index.contains_left(&index1)
      || !position_index_to_node_index.contains_left(&index2)
    {
      continue;
    }
    let node0 = *position_index_to_node_index.get_by_left(&index0).unwrap();
    let node1 = *position_index_to_node_index.get_by_left(&index1).unwrap();
    let node2 = *position_index_to_node_index.get_by_left(&index2).unwrap();
    if is_point_edge_valid(
      &graph[node0],
      &graph[node1],
      obs_func,
      collision_check_step_size,
    ) {
      // Use update_edge to avoid panic when adding the same edge
      graph.update_edge(node0, node1, ());
    }
    if is_point_edge_valid(
      &graph[node0],
      &graph[node2],
      obs_func,
      collision_check_step_size,
    ) {
      // Use update_edge to avoid panic when adding the same edge
      graph.update_edge(node0, node2, ());
    }
    if is_point_edge_valid(
      &graph[node1],
      &graph[node2],
      obs_func,
      collision_check_step_size,
    ) {
      // Use update_edge to avoid panic when adding the same edge
      graph.update_edge(node1, node2, ());
    }
  }
  (node_index_to_position, graph, position_index_to_node_index)
}
