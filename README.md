# Generating and Optimizing Topologically Distinct Guesses for Mobile Manipulator Path Planning
![Build, Run, Upload (Parallel)](https://github.com/rcywongaa/topologically_distinct_guesses/actions/workflows/build_run_publish.yml/badge.svg)
![Build, Run, Upload (Sequential)](https://github.com/rcywongaa/topologically_distinct_guesses/actions/workflows/build_run_publish_combined.yml/badge.svg)

Go to the [Project Page](https://rcywongaa.github.io/topologically_distinct_guesses/) for interactive visualization of the paths.

## Build
1. Install [Drake](https://drake.mit.edu/apt.html#stable-releases)
1. Install pybullet (recommended via virtual environment [`venv`](https://realpython.com/python-virtual-environments-a-primer/))
    ```
    python3 -m venv .
    source bin/activate
    pip3 install pybullet --upgrade
    ```
1. Build `ros2_ws`
    ```
    cd ros2_ws && rosdep install -y --ignore-src --rosdistro jazzy --from-paths .
    ```

## Run
1. Edit `config/planning_setting.py` to set up planning parameters
1. `pushd config && python3 planning_setting.py; popd`
1. `pushd topo_geo_paths && RUST_BACKTRACE=1 RUST_LOG=info cargo run -r; popd`
1. `python3 optimize_feasible.py --filename topo_geo_paths/trajectory0.txt --output trajectory0_opt.txt 2>/dev/null`
1. `source ros2_ws/install/setup.zsh; python3 drake_viz.py --filename=trajectory0_opt.txt`

## MoveIt2
Building `moveit2` from source is required due to some bugs on constraint planning in the release version.

Whenever editing `config/planning_setting.py`, remember to run `python3 planning_setting.py` within the `config/` directory.
The MoveIt experiments rely on some auto-generated files sym-linked to the `config/` directory.
