Remember to use `colcon build --symlink-install` to properly handle symlinks in `ompl_constrained_planning/config`

The moveit binaries have an issue where planning reports success even though the trajectory doesn't actually reach the goal.
This PR is needed: https://github.com/moveit/moveit2/pull/2455
which requires building from source.

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
