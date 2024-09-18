use ahash::{HashSet, HashSetExt};
use itertools::Itertools;
use log::debug;
use ndarray::Array;

use std::rc::Rc;
use std::time::Instant;
use topo_geo_paths::graph_utils::find_nearest_node_index_at_t;
use topo_geo_paths::nag_dijkstra;

use topo_geo_paths::link_obstacles::{
  combine_link_obstacles, link_aabb_obstacle, link_sphere_obstacle, to_rc,
};
use topo_geo_paths::{
  array_utils::linspace, graph_utils::print_path, swept_sphere_graph::create_swept_sphere_graph,
  visualization::Visualizer,
};

fn main() {
  env_logger::init();

  // let no_obs_func = |_, _| false;
  let obs_func = combine_link_obstacles(
    [
      to_rc(link_aabb_obstacle([-0.3, 0.3, 0.0], [0.5, 0.5, 1.5])),
      // Rc::new(projected_sphere_obstacle([-0.5, 0.5, 0.0], 0.45)),
      // Rc::new(projected_sphere_obstacle([0.5, -0.5, 0.0], 0.5)),
      // Rc::new(projected_sphere_obstacle([-0.5, -0.5, 0.0], 0.4)),
    ]
    .to_vec(),
  );

  let num_discretization = 20;
  let now = Instant::now();
  let (node_index_to_position, graph, _) = create_swept_sphere_graph(
    &linspace(
      Array::from_iter([-1., 0., 0.]),
      Array::from_iter([1., 0., 0.]),
      num_discretization,
    )
    .outer_iter()
    .map(|center| center.to_vec().try_into().unwrap())
    .collect_vec(),
    0.4, // Panda forearm is 0.384m
    4,   // 0.1m resolution
    &obs_func,
    0.01,
  );
  println!("Graph creation took {}ms", now.elapsed().as_millis());

  println!(
    "Graph has {} nodes and {} edges",
    graph.node_count(),
    graph.edge_count()
  );

  let mut visualizer = Visualizer::new();
  // visualizer.visualize_graph(graph.clone());

  debug!("{:?}", node_index_to_position);

  let mut start = HashSet::new();
  start.insert(find_nearest_node_index_at_t(
    &node_index_to_position,
    [-1., 5., 0.],
    0,
  ));
  let mut goal = HashSet::new();
  goal.insert(find_nearest_node_index_at_t(
    &node_index_to_position,
    [1., 5., 0.],
    num_discretization - 1,
  ));

  println!("Started...");
  let num_paths = 3;
  let now = Instant::now();
  let (nag, goal_node_indices) = nag_dijkstra::nag_dijkstra(
    &graph,
    &start,
    &goal,
    num_paths,
    |nag| {
      // visualizer.visualize_nag(nag.clone(), graph.clone());
    },
    500,
  );

  println!(
    "Creating {}/{} elbow trajectories took {}ms",
    goal_node_indices.len(),
    num_paths,
    now.elapsed().as_millis()
  );

  if goal_node_indices.is_empty() {
    println!("No paths found!");
    return;
  }

  print_path(&nag, &goal_node_indices);
  visualizer.visualize_goals(&nag, &graph, &goal_node_indices);
}
