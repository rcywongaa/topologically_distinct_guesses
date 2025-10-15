'''
For 1 .. N
1. Generate randomized simple scene obstacle specifications.
2. Run NAGS for X seconds to generate distinct paths
3. Optimize NAGS paths and record final costs.
4. Run OMPL for X seconds to generate distinct paths
5. Optimize OMPL paths and record final costs.
'''

'''
This should be run in the `code` directory!
'''
import subprocess
import os
import time
import glob
import argparse

IS_OPTIMIZE = False

parser = argparse.ArgumentParser()
parser.add_argument(
    "num_trials",
    type=int,
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="./randomized_trials",
    help="Directory to save output files"
)

print("Make sure this is run in the code directory!")
time.sleep(3)

args = parser.parse_args()

for i in range(args.num_trials):
    working_dir = f"{args.output_dir}/trial{i}"
    if not os.path.exists(working_dir):
        print("Creating working directory:", working_dir)
        os.makedirs(working_dir)
    nag_results_dir = os.path.join(working_dir, "nags_results")
    ompl_results_dir = os.path.join(working_dir, "ompl_results")
    if not os.path.exists(nag_results_dir):
        os.makedirs(nag_results_dir)
        print("Creating NAGS results directory:", nag_results_dir)
    if not os.path.exists(ompl_results_dir):
        os.makedirs(ompl_results_dir)
        print("Creating OMPL results directory:", ompl_results_dir)

    print("Generating randomized scene...")
    subprocess.run(["python3", "generate_randomized_scene.py"], cwd="./config", check=True)
    print("Moving randomized scene data to working directory...")
    subprocess.run(["cp", "config/randomized_scene.pkl", working_dir + "/"], check=True)
    print("Setting up planning settings...")
    subprocess.run(["python3", "planning_setting.py"], cwd="./config", check=True)

    print("Running NAGS...")
    env = os.environ.copy()
    env["RUST_BACKTRACE"] = "1"
    env["RUST_LOG"] = "info"
    subprocess.run(["cargo", "run", "--release"], cwd="./topo_geo_paths", env=env, check=True)

    print("Copying NAGS results to working directory...")
    nag_trajectory_files = glob.glob("topo_geo_paths/trajectory*.txt")
    if nag_trajectory_files:
        subprocess.run(["mv", *nag_trajectory_files, nag_results_dir], check=True)
        if IS_OPTIMIZE:
            print("Optimizing NAGS results...")
            subprocess.run(["python3", "optimize_feasible_batch.py", *glob.glob(nag_results_dir + "/trajectory*.txt")], check=True)

    ### Ensure no nodes from last run
    time.sleep(1)
    if subprocess.run(["ros2", "node", "list"], capture_output=True).stdout != b'':
        subprocess.run(["killall", "ompl_constrained_planning"])
    while subprocess.run(["ros2", "node", "list"], capture_output=True).stdout != b'':
        time.sleep(2)
        print("Waiting for ROS2 nodes to be killed...")

    print("Running OMPL...")
    subprocess.run(["ros2", "launch", "moveit2_tutorials", "ompl_constrained_planning.launch.py"], cwd=ompl_results_dir, env=os.environ.copy(), check=True)
    print("Optimizing OMPL results...")
    ompl_trajectory_files = glob.glob(ompl_results_dir + "/trajectory*.txt")
    if ompl_trajectory_files:
        if IS_OPTIMIZE:
            subprocess.run(["python3", "optimize_feasible_batch.py", *ompl_trajectory_files], check=True)

