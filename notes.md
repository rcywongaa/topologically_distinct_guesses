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

### Collision Handling
```
"""
Trick from Equation 6
Finding Locally Optimal, Collision-Free Trajectories with Sequential Convex Optimization

Doesn't solve unless very few obstacles.
Likely due to large number of variables t created
"""
# t = prog.NewContinuousVariables(1)
# prog.SetInitialGuess(t, [0])
# prog.AddCost(t[0] * soft_constraint_scaling)
# prog.AddConstraint(t[0] >= 0)
# prog.AddConstraint(-(l2_norm_sq_expr(sample - center) - radius**2) <= t[0])

```
```
def add_link_not_in_obstacle_constraint(self, prog, sample):
    """distance_3d attempt (doesn't support pyobject)"""
    # sphere = colliders.Sphere(sample, scene.link_radius)
    # box_transform = np.eye(4)
    # box_transform[:3, 3] = self.center
    # print(box_transform)
    # box = colliders.Box(box_transform, self.dims)
    # prog.AddConstraint(gjk.gjk_distance(sphere, box)[0])
    """ sphere decomposition attempt (too slow) """
    # for sphixel in self.sphixels:
    #     sphixel.add_link_not_in_obstacle_constraint(prog, sample)
    """ pybullet attempt (ignores Expressions) """
    # sphere = p.createCollisionShape(p.GEOM_SPHERE, radius=scene.link_radius)
    # box = p.createCollisionShape(
    #     p.GEOM_BOX,
    #     halfExtents=[self.dims[0] / 2, self.dims[1] / 2, self.dims[2] / 2],
    # )
    # pts = p.getClosestPoints(
    #     bodyA=-1,
    #     bodyB=-1,
    #     distance=100,
    #     collisionShapeA=sphere,
    #     collisionShapePositionA=sample,
    #     collisionShapeB=box,
    #     collisionShapePositionB=self.center,
    # )
    # print(sample)
    # print(pts)
    """ FCL attempt (doesn't support Expression) """
    # sphere = fcl.CollisionObject(fcl.Sphere(scene.link_radius), fcl.Transform())
    # sphere.setTranslation(sample)
    # box = fcl.CollisionObject(fcl.Box(*self.dims), fcl.Transform())
    # box.setTranslation(self.center)
    # req = fcl.DistanceRequest()
    # res = fcl.DistanceResult()
    # ret = fcl.distance(sphere, box, req, res)
    # print(ret)
    """ pydrake ComputeSignedDistanceToPoint """

```

```
def dist(self, sample, col):
    """distance_3d (too slow) """
    # sphere = colliders.Sphere(sample, scene.link_radius)
    # box_transform = np.eye(4)
    # box_transform[:3, 3] = self.center
    # box = colliders.Box(box_transform, self.dims)
    # return np.array([gjk.gjk_distance(sphere, box)[0]])
    """FCL does not report signed distance"""
    # sphere = fcl.CollisionObject(fcl.Sphere(scene.link_radius), fcl.Transform())
    # sphere.setTranslation(sample)
    # box = fcl.CollisionObject(fcl.Box(*self.dims), fcl.Transform())
    # box.setTranslation(self.center)
    # req = fcl.DistanceRequest()
    # res = fcl.DistanceResult()
    # ret = fcl.distance(sphere, box, req, res)
    # if ret < self.min_dist:
    #     self.min_dist = ret
    #     print(f"min dist = {self.min_dist}")
    #     print(f"sample = {sample}")
    #     print(
    #         f"res b1: {res.b1}, b2: {res.b2}, nearest point: {res.nearest_points}, o1: {res.o1}, o2: {res.o2}"
    #     )
    # return np.array([ret])
    """ pybullet attempt """
    pts = p.getClosestPoints(
        bodyA=-1,
        bodyB=-1,
        distance=100,
        collisionShapeA=col,
        collisionShapePositionA=sample,
        collisionShapeB=self.box_col,
        collisionShapePositionB=self.center,
    )

```
