# Generating and Optimizing Topologically Distinct Guesses for Mobile Manipulator Path Planning with Path Constraints ([arxiv](https://arxiv.org/abs/2410.20635))
![sequential workflow](https://github.com/rcywongaa/topologically_distinct_guesses/actions/workflows/build_run_publish_combined.yml/badge.svg)
![parallel workflow](https://github.com/rcywongaa/topologically_distinct_guesses/actions/workflows/build_run_publish.yml/badge.svg)

 Go [here](https://rcywongaa.github.io/topologically_distinct_guesses/) for interactive visualization of the paths.
 Alternatively, use the following links
<!-- ## Results -->
<!-- Using anonymous.4open.science links here seems to mess up the README -->
[Path 1](https://refined-github-html-preview.kidonng.workers.dev/rcywongaa/topologically_distinct_guesses/raw/refs/heads/master/outputs/trajectory0_opt.html)
[Path 2](https://refined-github-html-preview.kidonng.workers.dev/rcywongaa/topologically_distinct_guesses/raw/refs/heads/master/outputs/trajectory1_opt.html)
[Path 3](https://refined-github-html-preview.kidonng.workers.dev/rcywongaa/topologically_distinct_guesses/raw/refs/heads/master/outputs/trajectory2_opt.html)

(May show "No connection to server", please wait a few seconds for the simulation to load)

Animation controls are under "Open Controls/Animation/defualt"

## Setup
1. Install [Drake](https://drake.mit.edu/apt.html#stable-releases)
1. Install Python dependencies (recommended via virtual environment [`venv`](https://realpython.com/python-virtual-environments-a-primer/))
    ```
    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install pybullet --upgrade
    pip3 install playwright
    ```
1. Set up playwright: `playwright install`
1. Install ROS2 dependencies `ros2_ws`
    ```
    cd ros2_ws && rosdep install -y --ignore-src --rosdistro jazzy --from-paths .
    ```
1. Build
   ```
   colcon build --symlink-install
   ```
   Remember to use `colcon build --symlink-install` to properly handle symlinks in `ompl_constrained_planning/config`

## Run
### Set up planning scene (no need to recompile)
1. Edit `config/planning_setting.py` to set up planning parameters
1. `pushd config && python3 planning_setting.py; popd`
### Run NAGS algorithm
1. `pushd topo_geo_paths && RUST_BACKTRACE=1 RUST_LOG=info cargo run -r; popd`
### Run OMPL
1. `ros2 launch moveit2_tutorials ompl_constrained_planning.launch.py`
### Optimize
1. `python3 optimize_feasible.py --filename topo_geo_paths/trajectory0.txt --output trajectory0_opt.txt 2>/dev/null`
### Visualize
1. `source ros2_ws/install/setup.zsh; python3 drake_viz.py --filename=trajectory0_opt.txt`

