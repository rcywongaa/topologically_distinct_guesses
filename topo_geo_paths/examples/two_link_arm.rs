use std::{
  fs::{self, File},
  io::Write,
  rc::Rc,
  time::Instant,
};

use ahash::{HashSet, HashSetExt};
use itertools::Itertools;
use ndarray::Array;
use petgraph::dot::{Config, Dot};
// use rand::Rng;
use topo_geo_paths::{
  array_utils::linspace,
  graph_utils::{find_nearest_node_index, output_trajectory, print_path},
  link_obstacles::{combine_link_obstacles, link_sphere_obstacle, to_rc},
  nag_dijkstra,
  two_link_arm_graph::create_two_link_arm_graph,
  visualization::Visualizer,
};

fn main() {
  env_logger::init();

  // let mut rng = rand::thread_rng();

  let obs_func = combine_link_obstacles(
    [
      // to_rc(projected_sphere_obstacle([0.5, 0.0, 0.0], 0.2)),
      // to_rc(projected_sphere_obstacle([-0.5, 0.0, 0.0], 0.2)),
      // Rc::new(projected_sphere_obstacle(
      //   [rng.gen_range(0.0..1.0), rng.gen_range(-1.0..1.0), 0.0],
      //   rng.gen_range(0.1..0.4),
      // )),
      // Rc::new(projected_sphere_obstacle(
      //   [rng.gen_range(-1.0..0.0), rng.gen_range(-1.0..1.0), 0.0],
      //   rng.gen_range(0.1..0.4),
      // )),
    ]
    .to_vec(),
  );

  let num_discretization = 20;
  let now = Instant::now();
  let (node_index_to_position, graph, _) = create_two_link_arm_graph(
    &linspace(
      Array::from_iter([-1., 0., 0.5]),
      Array::from_iter([1., 0., 0.5]),
      num_discretization,
    )
    .outer_iter()
    .map(|center| center.to_vec().try_into().unwrap())
    .collect_vec(),
    0.4, // Panda forearm is 0.384m
    0.3, // Panda upper arm is 0.316m
    3,   // 0.1m resolution
    6,
    &obs_func,
  );
  println!("Graph creation took {}ms", now.elapsed().as_millis());

  // fs::write(
  //   "two_link_arm_graph.dot",
  //   format!("{:?}", Dot::with_config(&graph, &[Config::EdgeNoLabel])),
  // )
  // .unwrap();

  println!(
    "Graph has {} nodes and {} edges",
    graph.node_count(),
    graph.edge_count()
  );

  let mut start = HashSet::new();
  start.insert(find_nearest_node_index(
    &node_index_to_position,
    [
      -1., 5., 0., // elbow position
      -1., 0., 0., // end effector position
      -1., 5., // base position
    ],
  ));
  let mut goal = HashSet::new();
  goal.insert(find_nearest_node_index(
    &node_index_to_position,
    [
      1., 5., 0., // elbow position
      1., 0., 0., // end effector position
      1., 5., // base position
    ],
  ));

  println!("Start: {:?}, Goal: {:?}", start, goal);

  let num_paths = 4;
  let mut visualizer = Visualizer::new();
  visualizer.visualize_arm_graph(graph.clone());

  let now = Instant::now();
  let (nag, goal_node_indices) = nag_dijkstra::nag_dijkstra(
    &graph,
    &start,
    &goal,
    num_paths,
    |nag| {
      // visualizer.visualize_arm_nag(nag.clone(), graph.clone());
    },
    5000,
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
  // for (idx, goal_node_index) in goal_node_indices.iter().enumerate() {
  //   let mut trajectory_file = File::create(format!("trajectory{}.txt", idx)).unwrap();
  //   trajectory_file
  //     .write_all(output_trajectory(&graph, &nag, &goal_node_index).as_bytes())
  //     .unwrap();
  // }
  visualizer.visualize_arm_path(&nag, &graph, &goal_node_indices);
}
