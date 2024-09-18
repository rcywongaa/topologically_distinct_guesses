use std::time::Instant;

use ahash::{HashMap, HashMapExt, HashSet, HashSetExt};
use itertools::Itertools;
use log::debug;
use ndarray::Array;
use petgraph::visit::IntoNodeReferences;
use topo_geo_paths::{
  array_utils::linspace,
  graph_utils::{
    find_nearest_node_index, find_nearest_node_index_at_t, generate_node_index_to_position_map,
    print_path,
  },
  kinematics::calc_elbow_position,
  link_obstacles::{combine_link_obstacles, link_aabb_obstacle, link_sphere_obstacle, to_rc},
  mm_8dof_graph::{self, create_mm_6dof_arm_graph},
  nag_dijkstra,
  not_nan_util::from_not_nan_1d,
  visualization::Visualizer,
};
fn main() {
  env_logger::init();

  let obs_func = combine_link_obstacles(
    [
      // to_rc(projected_aabb_obstacle([-0.6, 0.0, 0.0], [0.2, 0.2, 0.2])),
      to_rc(link_aabb_obstacle([0.0, 0.0, 0.0], [0.2, 0.2, 0.2])),
      // to_rc(projected_aabb_obstacle([0.6, 0.0, 0.0], [0.2, 0.2, 0.2])),
      // to_rc(projected_sphere_obstacle([0.6, 0.0, 0.0], 0.2)),
      // to_rc(projected_sphere_obstacle([0.0, 0.0, 0.0], 0.2)),
      // to_rc(projected_sphere_obstacle([-0.6, 0.0, 0.0], 0.2)),
      // to_rc(projected_sphere_obstacle([0.5, 0.0, 0.0], 0.2)),
      // to_rc(projected_sphere_obstacle([-0.5, 0.0, 0.0], 0.2)),
    ]
    .to_vec(),
  );

  let eef_discretization = 10;
  let now = Instant::now();
  let forearm_length = 0.4; // Panda forearm is 0.384m
  let upperarm_length = 0.3;
  let base_lb = [-2., -2.];
  let base_ub = [2., 2.];

  let graph = create_mm_6dof_arm_graph(
    forearm_length,
    upperarm_length,
    base_lb,
    base_ub,
    0.1,
    &linspace(
      Array::from_iter([-1., 0.05, 0.3]),
      Array::from_iter([1., 0.05, 0.3]),
      eef_discretization,
    )
    .outer_iter()
    .map(|center| center.to_vec().try_into().unwrap())
    .collect_vec(),
    &obs_func,
  );
  println!("Graph creation took {}ms", now.elapsed().as_millis());

  println!(
    "Graph has {} nodes and {} edges",
    graph.node_count(),
    graph.edge_count()
  );

  let mut visualizer = Visualizer::new();
  visualizer.visualize_arm_graph(graph.clone());

  let node_index_to_position = generate_node_index_to_position_map(&graph);
  debug!("{:?}", node_index_to_position);

  let start_base_position = [-1., -0.3];
  let start_eef_position = [-1., 0., 0.3];
  let start_elbow_position = calc_elbow_position(
    forearm_length,
    upperarm_length,
    &start_eef_position,
    &start_base_position,
    true,
  )
  .unwrap();
  let start_pose = [
    start_elbow_position[0],
    start_elbow_position[1],
    start_elbow_position[2],
    start_eef_position[0],
    start_eef_position[1],
    start_eef_position[2],
    start_base_position[0],
    start_base_position[1],
    0.0,
  ];
  let mut start = HashSet::new();
  start.insert(find_nearest_node_index(&node_index_to_position, start_pose));

  let goal_base_position = [1., -0.3];
  let goal_eef_position = [1., 0., 0.3];
  let goal_elbow_position = calc_elbow_position(
    forearm_length,
    upperarm_length,
    &goal_eef_position,
    &goal_base_position,
    true,
  )
  .unwrap();
  let goal_pose = [
    goal_elbow_position[0],
    goal_elbow_position[1],
    goal_elbow_position[2],
    goal_eef_position[0],
    goal_eef_position[1],
    goal_eef_position[2],
    goal_base_position[0],
    goal_base_position[1],
    0.0,
  ];
  let mut goal = HashSet::new();
  goal.insert(find_nearest_node_index(&node_index_to_position, goal_pose));

  println!("Started...");
  let num_paths = 4;
  let now = Instant::now();
  let (nag, goal_node_indices) = nag_dijkstra::nag_dijkstra(
    &graph,
    &start,
    &goal,
    num_paths,
    |nag| {
      // visualizer.visualize_arm_nag(nag.clone(), graph.clone());
    },
    2000,
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
  visualizer.visualize_arm_path(&nag, &graph, &goal_node_indices);
}
