use std::{
  thread::sleep,
  time::{Duration, Instant},
};

use itertools::Itertools;
use ndarray::Array;
use petgraph::algo::dijkstra;
use topo_geo_paths::{
  array_utils::linspace,
  graph_utils::find_nearest_node_index,
  link_obstacles::{combine_link_obstacles, link_sphere_obstacle, to_rc},
  two_link_arm_graph::create_two_link_arm_graph,
};

fn main() {
  env_logger::init();

  let obs_func = combine_link_obstacles(
    [
      to_rc(link_sphere_obstacle([0.5, 0.0, 0.0], 0.2)),
      to_rc(link_sphere_obstacle([-0.5, 0.0, 0.0], 0.2)),
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

  let start = find_nearest_node_index(
    &node_index_to_position,
    [
      -1., 5., 0., // elbow position
      -1., 0., 0., // end effector position
      -1., 5., // base position
    ],
  );

  let now = Instant::now();
  let mut num_edge_expansions = 0;
  let res = dijkstra(&graph, start, None, |_| {
    num_edge_expansions += 1;
    1
  });
  println!("Took {}ms", now.elapsed().as_millis());
  println!("Expanded {} edges", num_edge_expansions);
}
