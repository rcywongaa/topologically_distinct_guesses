use ahash::{HashSet, HashSetExt};
use std::{rc::Rc, time::Instant};

use topo_geo_paths::{
  graph_utils::{find_nearest_node_index, print_path},
  hexasphere_graph::create_hexasphere_graph,
  nag_dijkstra,
  obstacles::{combine_obstacles, sphere_obstacle, to_rc},
  visualization::Visualizer,
};

fn main() {
  env_logger::init();

  let obs_func = combine_obstacles(
    [
      to_rc(sphere_obstacle([0.0, 1.0, 0.0], 0.3)),
      // sphere_obstacle([-0.5, 0.5, 0.0], 0.5),
      // sphere_obstacle([-0.707, 0.707, 0.0], 0.1),
    ]
    .to_vec(),
  );
  let (node_index_to_position, graph, _) =
    create_hexasphere_graph([0., 0., 0.], 1.0, 5, &obs_func, 0.01);

  /*
   * Note that having start and goal at antipodal will cause nag to fail to find distinct paths
   * This is because all paths topo-geometrically equivalent
   */
  let mut start = HashSet::new();
  start.insert(find_nearest_node_index(
    &node_index_to_position,
    [-10., 10., 0.],
  ));
  let mut goal = HashSet::new();
  goal.insert(find_nearest_node_index(
    &node_index_to_position,
    [10., 10., 0.],
  ));

  let mut visualizer = Visualizer::new();

  visualizer.visualize_graph(
    graph.clone(),
    [start.clone(), goal.clone()].iter().flatten(),
  );

  let now = Instant::now();
  let (nag, goal_indices) = nag_dijkstra::nag_dijkstra(
    &graph,
    &start,
    &goal,
    4,
    |nag| {
      // visualizer.visualize_nag(nag.clone(), graph.clone());
    },
    100,
  );

  println!("Done! Took {}s", now.elapsed().as_secs());

  print_path(&nag, &goal_indices);

  visualizer.visualize_goals(&nag, &graph, &goal_indices);
}
