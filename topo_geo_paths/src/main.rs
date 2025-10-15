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
use clap::Parser;

use topo_geo_paths::{
  find_distinct_paths::find_distinct_paths, planning_setting::get_planning_setting,
  visualization::Visualizer,
};

// use find_distinct_paths::find_distinct_paths;
// use planning_setting::get_planning_setting;
// use visualization::Visualizer;

#[derive(Parser, Debug)]
struct Cli {
  // cargo run -r -- -v
  #[clap(short, long, default_value = "false")]
  visualize: bool,
}

fn main() {
  env_logger::init();
  let args = Cli::parse();

  if args.visualize {
    find_distinct_paths(get_planning_setting(), Some(Visualizer::new()));
  } else {
    find_distinct_paths(get_planning_setting(), None);
  }
}
