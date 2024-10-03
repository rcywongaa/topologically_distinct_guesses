// mod array_utils;
// mod colors;
// mod find_distinct_paths;
// mod graph_utils;
// mod grid_graph;
// mod hexasphere_graph;
// mod nag_dijkstra;
// mod not_nan_util;
// mod planning_setting;
// mod projected_obstacles;
// mod swept_sphere_graph;
// mod two_link_arm_graph;
// mod visualization;

use topo_geo_paths::{
  find_distinct_paths::find_distinct_paths, planning_setting::get_planning_setting,
  visualization::Visualizer,
};

// use find_distinct_paths::find_distinct_paths;
// use planning_setting::get_planning_setting;
// use visualization::Visualizer;

fn main() {
  env_logger::init();

  // find_distinct_paths(get_planning_setting(), Some(Visualizer::new()));
  find_distinct_paths(get_planning_setting(), None);
}
