use std::{
  fs::{self, File},
  io::Write,
  rc::Rc,
  time::Instant,
};

use crate::{
  graph_utils::{
    find_nearest_node_index, generate_node_index_to_position_map, output_trajectory, print_path,
  },
  link_obstacles::{
    combine_link_obstacles, link_aabb_obstacle, link_cylinder_obstacle, link_sphere_obstacle,
  },
  mm_8dof_graph::create_mm_6dof_arm_graph,
  nag_dijkstra,
  obstacles::{aabb_obstacle, combine_obstacles, cylinder_obstacle, sphere_obstacle},
  planning_setting::PlanningSetting,
  visualization::Visualizer,
};
use ahash::{HashSet, HashSetExt};
use itertools::Itertools;

pub fn find_distinct_paths(planning_setting: PlanningSetting, mut visualizer: Option<Visualizer>) {
  // let mut rng = rand::thread_rng();

  let link_radius = planning_setting.link_radius;
  let link_obs_func = combine_link_obstacles(
    planning_setting
      .sphere_obstacle_specs
      .clone()
      .into_iter()
      .map(|(center, radius)| {
        Rc::new(link_sphere_obstacle(center, radius + link_radius))
          as Rc<dyn Fn([f32; 3], [f32; 3]) -> bool>
      })
      .chain(
        planning_setting
          .aabb_obstacle_specs
          .clone()
          .into_iter()
          .map(|(center, dim)| {
            Rc::new(link_aabb_obstacle(
              center,
              [
                dim[0] + 2.0 * link_radius,
                dim[1] + 2.0 * link_radius,
                dim[2] + 2.0 * link_radius,
              ],
            )) as Rc<dyn Fn([f32; 3], [f32; 3]) -> bool>
          }),
      )
      .chain(
        planning_setting
          .cylinder_obstacle_specs
          .clone()
          .into_iter()
          .map(|(center, radius_height)| {
            Rc::new(link_cylinder_obstacle(
              center,
              [
                radius_height[0] + link_radius,
                radius_height[1] + 2.0 * link_radius,
              ],
            )) as Rc<dyn Fn([f32; 3], [f32; 3]) -> bool>
          }),
      )
      .collect_vec(),
  );

  let base_radius = planning_setting.base_radius;
  let mobile_base_height = planning_setting.mobile_base_height;
  /* Shift obstacles by mobile_base_height */
  let base_obs_func = combine_obstacles(
    planning_setting
      .sphere_obstacle_specs
      .clone()
      .into_iter()
      .map(|(center, radius)| {
        Rc::new(sphere_obstacle(
          [center[0], center[1], center[2] + mobile_base_height],
          radius + base_radius,
        )) as Rc<dyn Fn([f32; 3]) -> bool>
      })
      .chain(
        planning_setting
          .aabb_obstacle_specs
          .clone()
          .into_iter()
          .map(|(center, dim)| {
            Rc::new(aabb_obstacle(
              [center[0], center[1], center[2] + mobile_base_height],
              [
                dim[0] + 2.0 * base_radius,
                dim[1] + 2.0 * base_radius,
                dim[2],
              ],
            )) as Rc<dyn Fn([f32; 3]) -> bool>
          }),
      )
      .chain(
        planning_setting
          .cylinder_obstacle_specs
          .clone()
          .into_iter()
          .map(|(center, radius_height)| {
            Rc::new(cylinder_obstacle(
              [center[0], center[1], center[2] + mobile_base_height],
              [radius_height[0] + base_radius, radius_height[1]],
            )) as Rc<dyn Fn([f32; 3]) -> bool>
          }),
      )
      .collect_vec(),
  );

  if let Some(visualizer) = &mut visualizer {
    for (center, radius) in planning_setting.sphere_obstacle_specs {
      visualizer.add_sphere_obstacle(center, radius, [0, 0, 0])
    }
    for (center, dims) in planning_setting.aabb_obstacle_specs {
      visualizer.add_aabb_obstacle(center, dims, [0, 0, 0])
    }
    for (center, radius_height) in planning_setting.cylinder_obstacle_specs {
      visualizer.add_cylinder_obstacle(center, radius_height, [0, 0, 0])
    }
  }

  let num_discretization = 40;
  let now = Instant::now();
  let eef_points = &(0..num_discretization + 1)
    .map(|t| {
      (planning_setting.eef_traj_func)(1.0 / (num_discretization as f32) * (t as f32))
      // .outer_iter()
      // .map(|center| center.to_vec().try_into().unwrap())
      // .collect_vec()
    })
    .collect_vec();
  let graph = create_mm_6dof_arm_graph(
    planning_setting.forearm_length,
    planning_setting.upperarm_length,
    planning_setting.arm_base_height,
    [-2.0, -2.0],
    [2.0, 2.0],
    0.1,
    eef_points,
    &link_obs_func,
    &base_obs_func,
  );
  println!(
    "Graph has {} nodes and {} edges, took {}ms to create",
    graph.node_count(),
    graph.edge_count(),
    now.elapsed().as_millis()
  );

  let node_index_to_position = generate_node_index_to_position_map(&graph);

  // fs::write(
  //   "two_link_arm_graph.dot",
  //   format!("{:?}", Dot::with_config(&graph, &[Config::EdgeNoLabel])),
  // )
  // .unwrap();

  let mut start = HashSet::new();
  start.insert(find_nearest_node_index(
    &node_index_to_position,
    [
      planning_setting.initial_x_w[0],
      planning_setting.initial_x_w[1],
      planning_setting.initial_x_w[2], // elbow position
      planning_setting.initial_x_e[0],
      planning_setting.initial_x_e[1],
      planning_setting.initial_x_e[2], // end effector position
      planning_setting.initial_x_b[0],
      planning_setting.initial_x_b[1],
      planning_setting.initial_x_b[2], // base position
    ],
  ));
  let mut goal = HashSet::new();
  goal.insert(find_nearest_node_index(
    &node_index_to_position,
    [
      planning_setting.final_x_w[0],
      planning_setting.final_x_w[1],
      planning_setting.final_x_w[2], // elbow position
      planning_setting.final_x_e[0],
      planning_setting.final_x_e[1],
      planning_setting.final_x_e[2], // end effector position
      planning_setting.final_x_b[0],
      planning_setting.final_x_b[1],
      planning_setting.final_x_b[2], // base position
    ],
  ));

  println!("Start: {:?}, Goal: {:?}", start, goal);

  if let Some(visualizer) = &mut visualizer {
    visualizer.visualize_arm_graph(graph.clone());
  }

  // let now = Instant::now();
  let (nag, goal_node_indices) = nag_dijkstra::nag_dijkstra(
    &graph,
    &start,
    &goal,
    planning_setting.num_paths,
    |nag| {
      if let Some(visualizer) = &mut visualizer {
        // visualizer.visualize_arm_nag(nag.clone(), graph.clone());
      }
    },
    1000,
  );

  println!(
    "Creating {}/{} trajectories took {}ms",
    goal_node_indices.len(),
    planning_setting.num_paths,
    now.elapsed().as_millis()
  );

  if goal_node_indices.is_empty() {
    println!("No paths found!");
    return;
  }

  // print_path(&nag, &goal_node_indices);
  for (idx, goal_node_index) in goal_node_indices.iter().enumerate() {
    let mut trajectory_file = File::create(format!("trajectory{}.txt", idx)).unwrap();
    trajectory_file
      .write_all(output_trajectory(&graph, &nag, &goal_node_index).as_bytes())
      .unwrap();
  }

  if let Some(visualizer) = &mut visualizer {
    visualizer.visualize_arm_path(&nag, &graph, &goal_node_indices);
  }
}
