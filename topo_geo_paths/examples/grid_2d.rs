use std::rc::Rc;

use ahash::{HashSet, HashSetExt};

use topo_geo_paths::graph_utils::{
  default_point_edge_func, default_vertex_func, generate_node_index_to_position_map,
};
use topo_geo_paths::nag_dijkstra::Graph;
use topo_geo_paths::not_nan_util::NotNan1D;
use topo_geo_paths::obstacles::{combine_obstacles, sphere_obstacle, to_rc};
use topo_geo_paths::visualization::Visualizer;
use topo_geo_paths::{
  graph_utils::find_nearest_node_index, grid_graph::create_grid_graph, nag_dijkstra,
};

fn main() {
  env_logger::init();
  let obs_func = combine_obstacles(
    [
      to_rc(sphere_obstacle([0.0, 0.0], 0.2)), // This works
    ]
    .to_vec(),
  );
  let (graph, _) = create_grid_graph(
    [-0.5, -0.5],
    [0.5, 0.5],
    50,
    50,
    &mut |graph: &mut Graph, weight: [f32; 2]| default_vertex_func(graph, weight, &obs_func),
    &mut |a: &NotNan1D, b: &NotNan1D| default_point_edge_func(a, b, &obs_func),
  );
  let node_index_to_position = generate_node_index_to_position_map(&graph);

  let mut start = HashSet::new();
  start.insert(find_nearest_node_index(
    &node_index_to_position,
    [-0.25, -0.25],
  ));
  let mut goal = HashSet::new();
  goal.insert(find_nearest_node_index(&node_index_to_position, [1., 1.]));

  let mut visualizer = Visualizer::new();
  let (nag, goal_indices) = nag_dijkstra::nag_dijkstra(
    &graph,
    &start,
    &goal,
    6,
    |nag| {
      visualizer.visualize_nag(nag.clone(), graph.clone());
    },
    500,
  );
  visualizer.visualize_goals(&nag, &graph, &goal_indices);
}
