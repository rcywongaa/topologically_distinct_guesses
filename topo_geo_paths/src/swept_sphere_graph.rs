use crate::graph_utils::is_pose_edge_valid;
use crate::not_nan_util::{dist, from_not_nan_1d, NotNan1D};
use crate::{
  hexasphere_graph::create_hexasphere_graph, nag_dijkstra::Graph, not_nan_util::to_not_nan_1d,
};
use ahash::{HashMap, HashMapExt};
use bimap::BiHashMap;
use ndarray::s;
use petgraph::visit::EdgeRef;
use petgraph::{graph::NodeIndex, visit::IntoEdgeReferences};

/**
 * Abbreviations:
 * ss: swept sphere
 *
 * The obstacle function is defined such that the first argument is the sample point
 * and the second argument is the center of the sphere
 */
pub fn create_swept_sphere_graph<'a>(
  sweep_points: impl IntoIterator<Item = &'a [f32; 3]>,
  radius: f32,
  subdivisions: usize,
  obs_func: impl Fn([f32; 3], [f32; 3]) -> bool,
  collision_check_step_size: f32,
) -> (
  HashMap<NodeIndex, (usize, [f32; 3])>,
  Graph,
  BiHashMap<(usize, usize), NodeIndex>,
) {
  let mut ss_graph = Graph::default();
  let mut position_index_to_node_index = BiHashMap::new();
  let mut last_hexasphere_position_to_ss_index_map = BiHashMap::new();
  let mut node_index_map = HashMap::new();
  for (t, &center) in sweep_points.into_iter().enumerate() {
    let (node_to_position, hexasphere, hexasphere_position_to_node) = create_hexasphere_graph(
      center,
      radius,
      subdivisions,
      &|x: [f32; 3]| obs_func([x[0], x[1], x[2]], center),
      collision_check_step_size,
    );
    // Maps between node indices from the current hexasphere graph at t to the full swept sphere graph
    let mut hexasphere_graph_to_ss_graph_map = BiHashMap::new();
    // Maps between position indices from the current hexasphere graph at t to the full swept sphere graph
    let mut hexasphere_position_to_ss_index_map = BiHashMap::new();
    // Add nodes from current hexasphere graph
    for (&hexasphere_index, &position) in &node_to_position {
      let ss_index = ss_graph.add_node(to_not_nan_1d(&[position, center].concat()));
      hexasphere_graph_to_ss_graph_map.insert(hexasphere_index, ss_index);
      node_index_map.insert(ss_index, (t, position));
      position_index_to_node_index.insert(
        (
          *hexasphere_position_to_node
            .get_by_right(&hexasphere_index)
            .unwrap(),
          t,
        ),
        ss_index,
      );
    }

    // Add edges from current hexasphere graph
    for edge in hexasphere.edge_references() {
      ss_graph.add_edge(
        *hexasphere_graph_to_ss_graph_map
          .get_by_left(&edge.source())
          .unwrap(),
        *hexasphere_graph_to_ss_graph_map
          .get_by_left(&edge.target())
          .unwrap(),
        (),
      );
    }

    for (&position_index, &node_index) in &hexasphere_position_to_node {
      let &ss_index = hexasphere_graph_to_ss_graph_map
        .get_by_left(&node_index)
        .unwrap();
      hexasphere_position_to_ss_index_map.insert(position_index, ss_index);
    }

    // Add edges between hexasphere graphs
    for hexasphere_position_index in last_hexasphere_position_to_ss_index_map.left_values() {
      if hexasphere_position_to_ss_index_map.contains_left(hexasphere_position_index) {
        let &ss_node_of_last_hexasphere_at_position = last_hexasphere_position_to_ss_index_map
          .get_by_left(hexasphere_position_index)
          .unwrap();
        let &ss_node_of_current_hexasphere_at_position = hexasphere_position_to_ss_index_map
          .get_by_left(hexasphere_position_index)
          .unwrap();
        if is_pose_edge_valid(
          &ss_graph[ss_node_of_last_hexasphere_at_position],
          &ss_graph[ss_node_of_current_hexasphere_at_position],
          &obs_func,
          collision_check_step_size,
        ) {
          ss_graph.add_edge(
            ss_node_of_last_hexasphere_at_position,
            ss_node_of_current_hexasphere_at_position,
            (),
          );
        }

        // Add "diagonals" between hexaspheres
        let mut diagonal_edges = Vec::new();
        let last_neighbors = ss_graph.neighbors(ss_node_of_last_hexasphere_at_position);
        for last_neighbor in last_neighbors {
          if last_hexasphere_position_to_ss_index_map.contains_right(&last_neighbor) {
            // Only add diagonal if the neighbor belongs to the immediate last hexasphere
            diagonal_edges.push((last_neighbor, ss_node_of_current_hexasphere_at_position));
          }
        }
        for diagonal_edge in diagonal_edges {
          if is_pose_edge_valid(
            &ss_graph[diagonal_edge.0],
            &ss_graph[diagonal_edge.1],
            &obs_func,
            collision_check_step_size,
          ) {
            ss_graph.update_edge(diagonal_edge.0, diagonal_edge.1, ());
          }
        }
      }
    }
    last_hexasphere_position_to_ss_index_map = hexasphere_position_to_ss_index_map;
  }
  (node_index_map, ss_graph, position_index_to_node_index)
}
