# Generating and Optimizing Topologically Distinct Guesses for Mobile Manipulator Path Planning

<!-- Go to the [Project Page](https://rcywongaa.github.io/topologically_distinct_guesses/) for interactive visualization of the paths. -->
<!-- ## Results -->
<!-- Using anonymous.4open.science links here seems to mess up the README -->
<!-- [Path 1](https://anonymous.4open.science/w/topologically_distinct_guesses-C0AA/trajectory0_opt.html)
[Path 2](https://anonymous.4open.science/w/topologically_distinct_guesses-C0AA/trajectory1_opt.html)
[Path 3](https://anonymous.4open.science/w/topologically_distinct_guesses-C0AA/trajectory2_opt.html)

(May show "No connection to server", please wait a few seconds for the simulation to load)

Animation controls are under "Open Controls/Animation/defualt"; -->

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

## Notes
### Randomized experiments
In the current implementation, `find_distinct_paths.rs` manually calls `planning_setting.py` to load values.

How do we ensure that the randomly generated values stay the same between NAGS and OMPL runs?
Separate the randomization from the `planning_setting.py` call.



### MoveIt2
The moveit binaries have an issue where planning reports success even though the trajectory doesn't actually reach the goal.
This PR is needed: https://github.com/moveit/moveit2/pull/2455
which requires building from source.

Whenever editing `config/planning_setting.py`, remember to run `python3 planning_setting.py` within the `config/` directory.
The MoveIt experiments rely on some auto-generated files sym-linked to the `config/` directory.

### Sharing `planning_setting.py`
`ros2-rust` did not support params at the time this project started...
Also, params/yaml does not allow passing eef path which is a python function.

How to get planning_setting.py stuff into ompl planning
- initial position (in joint angles)
- final position (in joint angles)
- eef path
- obstacles

c++ & pybind11
- can express eef path
- dealing with file paths is icky
- symlinks/hardlinks are finicky

python + param
- no support for nested types
- might need to use this eventually when moving to ROS

How to get `planning_setting.py` stuff into NAGS
- Remember that `planning_setting.rs` makes a copy of `planning_setting.py` to execute.
  Hence all relative paths from `planning_setting.py` is lost
- In order to preserve relative paths, we should use modules within `config/`.
- Modules are imported by `planning_setting.py` and hence `planning_setting.rs` with the correct directory, at `config/`.
