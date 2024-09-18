import json
import math
from math import floor, isnan, atan2
import numpy as np
import time
from pydrake.solvers import (
    MathematicalProgram,
    Solve,
    SolverOptions,
    CommonSolverOption,
    SnoptSolver,
)
from pydrake.math import sin, cos
from pydrake.math import (
    ComputeNumericalGradient,
    NumericalGradientMethod,
    NumericalGradientOption,
)
from typing import NamedTuple
from pydrake.geometry import (
    Rgba,
    Sphere,
    StartMeshcat,
    SceneGraph,
    GeometryInstance,
    Cylinder,
    Box,
    QueryObject,
)
from pydrake.autodiffutils import (
    AutoDiffXd,
    ExtractGradient,
    ExtractValue,
    InitializeAutoDiff,
)
from pydrake.math import RigidTransform, RollPitchYaw
from config import planning_setting
from dataformat import get_x
import uuid

WHEEL_BASE_WIDTH = 0.5  # m
MAX_WHEEL_LINEAR_VELOCITY = 2.0  # m/s
MAX_ELBOW_VELOCITY = 2 * MAX_WHEEL_LINEAR_VELOCITY  # should > MAX_WHEEL_LINEAR_VELOCITY
MAX_ANGULAR_VELOCITY = 4 * MAX_WHEEL_LINEAR_VELOCITY / WHEEL_BASE_WIDTH

LOOP_PLAYBACK = False
# LOOP_PLAYBACK = True
# VISUALIZE = False
VISUALIZE = True


def exponentially_smoothed_hinge_loss_function(x):
    if x.dtype == float:
        if x >= 0.0:
            return 0.0
        else:
            return -x * math.exp(1.0 / x)
    else:
        x_value = ExtractValue(x).flatten()
        x_grad = ExtractGradient(x)

        if x_value >= 0.0:
            y_value = 0.0
            return AutoDiffXd(y_value)
        else:
            exp_one_over_x = math.exp(1.0 / x_value)
            y_value = -x_value * exp_one_over_x
            y_grad = (-exp_one_over_x + exp_one_over_x / x_value) @ x_grad
            return AutoDiffXd(y_value, y_grad)


def quadratically_smoothed_hinge_loss_function(x):
    if x.dtype == float:
        if x >= 0.0:
            return 0.0
        else:
            if x > -1:
                return x**2 / 2
            else:
                return -0.5 - x
    else:
        x_value = ExtractValue(x).flatten()
        x_grad = ExtractGradient(x)

        if x_value >= 0.0:
            y_value = 0.0
            return AutoDiffXd(y_value)
        else:
            if x_value > -1:
                y_value = x_value**2 / 2
                y_grad = x_value @ x_grad
                return AutoDiffXd(y_value, y_grad)
            else:
                y_value = -0.5 - x_value
                y_grad = -x_grad.flatten()
                return AutoDiffXd(y_value, y_grad)


def dist_to_obstacle_expr(x, obstacle, sample_func, col):
    if x.dtype == float:
        sample = sample_func(x)
        return obstacle.dist(sample, col)
    else:
        x_value = ExtractValue(x).flatten()
        x_grad = ExtractGradient(x)

        sample = sample_func(x_value)
        # Compute the constraint and approximate gradient
        # We are using central differencing here as the `option` argument.
        y_value = obstacle.dist(sample, col)
        y_grad = (
            ComputeNumericalGradient(
                lambda val: obstacle.dist(sample_func(val), col),
                x_value,
                option=NumericalGradientOption(NumericalGradientMethod.kCentral),
            )
            @ x_grad
        )

        # Return the new AutoDiffXd object with the constraint and gradient
        return InitializeAutoDiff(y_value, y_grad)


c = 1.0


def gamma_fn(x, exprs, x_indices):
    if x.dtype == float:
        ret = 0.0
    else:
        ret = AutoDiffXd(0.0)
    for expr, (start_index, end_index) in zip(exprs, x_indices):
        ret += exponentially_smoothed_hinge_loss_function(
            10.0 * expr(x[start_index:end_index])
        )
        # ret += quadratically_smoothed_hinge_loss_function(
        #     1.0 * expr(x[start_index:end_index])
        # )
    return np.array([ret])


class Gamma:

    def __init__(self):
        self.exprs = []
        self.xs = []
        self.x_indices = []
        self.x_start_index = 0

    def add_expr(self, expr, x):
        self.exprs.append(expr)
        self.xs.append(x)
        end_index = self.x_start_index + len(x)
        self.x_indices.append((self.x_start_index, end_index))
        self.x_start_index = end_index
        # print(f"x: {x}, xs: {self.xs[-1]}, indices: {self.x_indices[-1]}")

    def create_constraint(self, prog):
        prog.AddConstraint(
            lambda x: gamma_fn(x, self.exprs, self.x_indices),
            lb=np.array([0.0]),
            ub=np.array([0.0]),
            vars=np.hstack(self.xs),
        )


def to_upperarm_sample(x, alpha):
    return (1 - alpha) * x[0:3] + alpha * np.array(
        [x[3], x[4], planning_setting.arm_base_height], dtype=x[0].dtype
    )


def to_forearm_sample(x, alpha, t):
    return (1 - alpha) * x[0:3] + alpha * np.array(
        planning_setting.get_eef_position(t), dtype=x[0].dtype
    )


def to_base_sample(x):
    # Center of mobile base cylinder
    return np.array(
        [x[0], x[1], planning_setting.mobile_base_height / 2], dtype=x[0].dtype
    )


def l2_norm_sq_expr(expr):
    return expr.T @ expr


def show_pose(meshcat, x_b, x_w, x_e, theta, is_persist=False):
    radius = 0.025
    link_color = Rgba(0.0, 0.0, 0.0, 0.25)
    link_width = 1  # pixels

    x_b_name = "x_b"
    x_w_name = "x_w"
    x_e_name = "x_e"
    heading_indicator_name = "heading_indicator"
    upperarm_name = "upperarm"
    forearm_name = "forearm"
    if is_persist:
        x_b_name = str(uuid.uuid4())
        x_w_name = str(uuid.uuid4())
        x_e_name = str(uuid.uuid4())
        heading_indicator_name = str(uuid.uuid4())
        upperarm_name = str(uuid.uuid4())
        forearm_name = str(uuid.uuid4())

    meshcat.SetTransform(x_b_name, RigidTransform(x_b))
    meshcat.SetObject(x_b_name, Sphere(radius), Rgba(1, 0, 0, 0.5))
    heading_indicator_position = (
        RigidTransform(x_b)
        .multiply(RigidTransform(rpy=RollPitchYaw([0, 0, theta]), p=[0, 0, 0]))
        .multiply(RigidTransform([radius, 0, 0]))
    )

    meshcat.SetTransform(
        heading_indicator_name, RigidTransform(heading_indicator_position)
    )
    meshcat.SetObject(heading_indicator_name, Sphere(0.5 * radius), Rgba(1, 1, 1, 0.5))

    meshcat.SetTransform(x_w_name, RigidTransform(x_w))
    meshcat.SetObject(x_w_name, Sphere(radius), Rgba(0, 1, 0, 0.5))

    meshcat.SetTransform(x_e_name, RigidTransform(x_e))
    meshcat.SetObject(x_e_name, Sphere(radius), Rgba(0, 0, 1, 0.5))

    meshcat.SetLine(
        upperarm_name,
        np.hstack(
            [x_b.reshape((-1, 1)), x_w.reshape((-1, 1))],
        ),
        link_width,
        link_color,
    )
    meshcat.SetLine(
        forearm_name,
        np.hstack(
            [x_w.reshape((-1, 1)), x_e.reshape((-1, 1))],
        ),
        link_width,
        link_color,
    )


import pdb


def visualization_callback(meshcat, prog, x, x_b_var, x_w_var, theta_var):
    if visualization_callback.count % 5 != 0:
        print(f"Num iterations: {visualization_callback.count}, skipping...")
    else:
        print(f"Num iterations: {visualization_callback.count}, visualizing...")
        meshcat.Delete()
        planning_setting.show_obstacles(meshcat)

        for k in range(N):
            x_b_val = x[prog.FindDecisionVariableIndices(x_b_var[k])]
            x_w_val = x[prog.FindDecisionVariableIndices(x_w_var[k])]
            theta_val = x[prog.FindDecisionVariableIndices(theta_var[k])]
            show_pose(
                meshcat,
                np.append(x_b_val, planning_setting.arm_base_height),
                x_w_val,
                x_e[k],
                theta_val,
                is_persist=True,
            )
        pdb.set_trace()
    visualization_callback.count += 1


visualization_callback.count = 0


def show_obstacles(meshcat, obstacles):
    for idx, obs in enumerate(obstacles):
        name = f"obstacle{idx}"
        meshcat.SetTransform(name, RigidTransform(obs.center))
        meshcat.SetObject(name, Sphere(obs.radius), Rgba(0.2, 0.2, 0.2, 1))


initial_x_b = planning_setting.initial_x_b
final_x_b = planning_setting.final_x_b

start_buffer = planning_setting.buffer
end_buffer = planning_setting.buffer
N = planning_setting.T
dt = planning_setting.dt

x_e = np.array(
    [planning_setting.get_eef_position(0)] * start_buffer
    + [
        planning_setting.get_eef_position(t)
        for t in np.linspace(0.0, 1.0, N - start_buffer - end_buffer)
    ]
    + [planning_setting.get_eef_position(1.0)] * end_buffer
)
link_discretization = 5
alphas = np.linspace(0, 1, link_discretization)


def optimize(
    x_b_guess=None,
    x_w_guess=None,
    a_guess=None,
    theta_guess=None,
    v_guess=None,
    w_guess=None,
):
    """Attempt to use drake's ComputeSignedDistanceToPoint (doesn't work with Expression)"""
    # scene_graph = SceneGraph()
    # source_id = scene_graph.RegisterSource()
    # for center, radius in planning_setting.sphere_obstacle_specs:
    #     scene_graph.RegisterAnchoredGeometry(
    #         source_id,
    #         GeometryInstance(shape=Sphere(radius), X_PG=RigidTransform(center)),
    #     )
    # for center, (radius, height) in planning_setting.cylinder_obstacle_specs:
    #     scene_graph.RegisterAnchoredGeometry(
    #         source_id,
    #         GeometryInstance(
    #             shape=Cylinder(radius, height), X_PG=RigidTransform(center)
    #         ),
    #     )
    # for center, dims in planning_setting.sphere_obstacle_specs:
    #     scene_graph.RegisterAnchoredGeometry(
    #         source_id,
    #         GeometryInstance(shape=Box(*dims), X_PG=RigidTransform(center)),
    #     )
    # query_object = scene_graph.get_query_output_port().Eval(
    #     scene_graph.CreateDefaultContext()
    # )
    """"""

    prog = MathematicalProgram()
    v = prog.NewContinuousVariables(N - 1, 1, name="v")
    w = prog.NewContinuousVariables(N - 1, 1, name="w")
    delta_x_w = prog.NewContinuousVariables(N - 1, 3, name="delta_x_w")
    x_w = prog.NewContinuousVariables(N, 3, name="x_w")
    x_b = prog.NewContinuousVariables(N, 2, name="x_b")
    x_b_aug = np.concatenate(
        [x_b, np.ones((x_b.shape[0], 1)) * planning_setting.arm_base_height], axis=1
    )
    theta = prog.NewContinuousVariables(N, 1, name="theta")

    a = prog.NewContinuousVariables(N, 1, name="a")

    # Costs
    for k in range(N - 1):
        prog.Add2NormSquaredCost(np.eye(1), np.zeros((1, 1)), v[k])
        # FIXME: Angular distance should be weighted differently from linear distance
        prog.Add2NormSquaredCost(np.eye(1), np.zeros((1, 1)), w[k])
        prog.Add2NormSquaredCost(np.eye(3), np.zeros((3, 1)), delta_x_w[k])

    # Dynamic constraints
    for k in range(N - 1):
        for i in range(x_b.shape[1]):
            prog.AddConstraint(
                x_b[k + 1, i]
                == x_b[k, i]
                + (np.array([[cos(theta[k, 0])], [sin(theta[k, 0])]]) @ v[k])[i] * dt
            )
        for i in range(theta.shape[1]):
            prog.AddConstraint(theta[k + 1, i] == theta[k, i] + w[k, i] * dt)
        for i in range(x_w.shape[1]):
            prog.AddConstraint(x_w[k + 1, i] == x_w[k, i] + delta_x_w[k, i] * dt)

        # Max velocity constraint
        prog.AddConstraint(l2_norm_sq_expr(v[k]) <= MAX_WHEEL_LINEAR_VELOCITY**2)
        prog.AddConstraint(l2_norm_sq_expr(delta_x_w[k]) <= MAX_ELBOW_VELOCITY**2)
        prog.AddConstraint(l2_norm_sq_expr(w[k]) <= MAX_ANGULAR_VELOCITY**2)
        # Add combined linear and angular velocity constraint (to mimic max wheel velocity)
        prog.AddConstraint(
            l2_norm_sq_expr(v[k]) + WHEEL_BASE_WIDTH**2 * l2_norm_sq_expr(w[k])
            <= MAX_WHEEL_LINEAR_VELOCITY**2
        )

    # Link length constraints
    for k in range(N):
        prog.AddConstraint(
            l2_norm_sq_expr(x_w[k] - x_e[k]) == planning_setting.forearm_length**2
        )
        prog.AddConstraint(
            l2_norm_sq_expr(x_w[k] - x_b_aug[k]) == planning_setting.upperarm_length**2
        )

    """
    Elbow position constraint
    This causes the warning
    WARNING:drake:UpdateHessianType(): Unable to determine Hessian type of the Quadratic Constraint. Falling back to indefinite Hessian type.
    """
    for k in range(N):
        required_projected_x_w = x_b[k] + a[k, 0] * (x_e[k, 0:2] - x_b[k])
        for i in range(2):
            prog.AddConstraint(x_w[k, i] == required_projected_x_w[i])

    # Not in obstacle constraint
    for k in range(N):
        gamma = Gamma()
        for obs in planning_setting.obstacles:
            for alpha in alphas:
                # Be careful about lambdas in loops
                # https://stackoverflow.com/questions/2295290/what-do-lambda-function-closures-capture
                gamma.add_expr(
                    lambda x, obs=obs, alpha=alpha: dist_to_obstacle_expr(
                        x,
                        obs,
                        lambda x, alpha=alpha: to_upperarm_sample(x, alpha),
                        planning_setting.link_col,
                    ),
                    np.hstack([x_w[k], x_b[k]]),
                )
                gamma.add_expr(
                    lambda x, obs=obs, alpha=alpha: dist_to_obstacle_expr(
                        x,
                        obs,
                        lambda x, alpha=alpha: to_forearm_sample(x, alpha, k / (N - 1)),
                        planning_setting.link_col,
                    ),
                    x_w[k],
                )
            gamma.add_expr(
                lambda x, obs=obs: dist_to_obstacle_expr(
                    x, obs, to_base_sample, planning_setting.base_col
                ),
                x_b[k],
            )
        gamma.create_constraint(prog)

    # initial and final positions
    for i in range(x_b.shape[1]):
        prog.AddConstraint(x_b[0, i] == planning_setting.initial_x_b[i])
        # prog.AddConstraint(theta[0, 0] == planning_setting.initial_theta)
        prog.AddConstraint(x_b[-1, i] == planning_setting.final_x_b[i])
        # prog.AddConstraint(theta[-1, 0] == planning_setting.final_theta)

    # Set initial guess
    if x_b_guess is not None:
        prog.SetInitialGuess(x_b, x_b_guess)
    if x_w_guess is not None:
        prog.SetInitialGuess(x_w, x_w_guess)
    if a_guess is not None:
        prog.SetInitialGuess(a, a_guess)
    if theta_guess is not None:
        prog.SetInitialGuess(theta, theta_guess)
    if v_guess is not None:
        prog.SetInitialGuess(v, v_guess)
    if w_guess is not None:
        prog.SetInitialGuess(w, w_guess)

    # for constraint in prog.GetAllConstraints():
    #     if not prog.CheckSatisfiedAtInitialGuess(constraint):
    #         print(f"Initial guess violates constraint: {constraint}")

    options = SolverOptions()
    # options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    options.SetOption(SnoptSolver.id(), "Major optimality tolerance", 5e-5)
    prog.SetSolverOptions(options)
    if VISUALIZE:
        prog.AddVisualizationCallback(
            lambda x: visualization_callback(meshcat, prog, x, x_b, x_w, theta),
            prog.decision_variables(),
        )
    start_time = time.time()
    print(f"Beginning to solve...")
    result = Solve(prog)
    print("Solver: ", result.get_solver_id().name())
    print(f"Solve time: {time.time() - start_time}s")
    print("Success? ", result.is_success())
    print(f"Cost: {result.get_optimal_cost()}")
    x_w_star = result.GetSolution(x_w)
    x_b_star = result.GetSolution(x_b)
    theta_star = result.GetSolution(theta)
    # print('x_e = ', x_e)
    # print('x_w* = ', x_w_star)
    # print('x_b* = ', x_b_star)

    return x_w_star, x_b_star, theta_star


def calc_a(x_w, x_b, x_e):
    a = (x_w[0:2] - x_b) / (x_e[0:2] - x_b)
    if isnan(a[0]) or isnan(a[1]):
        return 0.0
    assert abs(a[0] - a[1]) < 1e-3, f"a[0] - a[1] = {a[0]-a[1]}"
    return a[0]


def read_guess_from_file(filename):
    with open(filename) as f:
        x_w_guess = np.zeros((N, 3))
        x_b_guess = np.zeros((N, 2))
        a_guess = np.zeros((N, 1))
        theta_guess = np.zeros((N, 1))
        v_guess = np.zeros((N - 1, 1))
        w_guess = np.zeros((N - 1, 1))
        lines = f.readlines()
        num_lines = len(lines)
        interval = N / num_lines
        t = 0
        # x_b_guess[0] = planning_setting.initial_x_b[0:2]
        for i in range(1, num_lines):
            T = floor(interval * i)
            last_pose = json.loads(lines[i - 1])
            (last_x_w, last_x_e, last_x_b) = get_x(last_pose)
            pose = json.loads(lines[i])
            (x_w, x_e, x_b) = get_x(pose)

            x_w_guess[t:T, :] = np.linspace(
                last_x_w,
                x_w,
                T - t,
                endpoint=False,
            )
            x_b_guess[t:T, :] = np.linspace(
                last_x_b,
                x_b,
                T - t,
                endpoint=False,
            )

            last_a = calc_a(last_x_w, last_x_b, last_x_e)
            a = calc_a(x_w, x_b, x_e)
            a_guess[t:T, 0] = np.linspace(last_a, a, T - t, endpoint=False)
            t = T
        x_w_guess[T:, :] = x_w
        # x_b_guess[T:, :] = planning_setting.final_x_b[0:2]
        x_b_guess[T:, :] = x_b_guess[T - 1, :]
        a_guess[T:, :] = calc_a(x_w, x_b, x_e)

        """ We don't enforce initial and final theta anymore """
        # theta_guess[0] = planning_setting.initial_theta
        # theta_guess[-1] = planning_setting.final_theta
        for t in range(0, T - 1):
            d_xb = x_b_guess[t + 1] - x_b_guess[t]
            d_xb_norm = np.linalg.norm(d_xb)
            if d_xb_norm < 1e-6:
                theta_guess[t] = theta_guess[t - 1]
                v_guess[t] = 0.0
            else:
                theta_guess[t] = atan2(d_xb[1], d_xb[0])
                """
                Setting v_guess to 0.0 seems to do better
                """
                v_guess[t] = 0.0
                # v_guess[t] = d_xb_norm / dt
        theta_guess[-1] = theta_guess[T - 2]

        for t in range(0, T - 1):
            w_guess[t] = (theta_guess[t + 1] - theta_guess[t]) / dt

        return x_w_guess, x_b_guess, a_guess, theta_guess, v_guess, w_guess


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize a trajectory given a feasible guess"
    )
    parser.add_argument(
        "--filename",
        help="File containing the trajectory guess",
        type=str,
        required=False,
    )
    parser.add_argument("--output", help="Output filename", type=str, required=False)
    args = parser.parse_args()

    meshcat = StartMeshcat()
    meshcat.SetCameraPose(
        camera_in_world=[-1.5, -1.0, 2.5], target_in_world=[0.0, 0.0, 0.0]
    )
    planning_setting.show_obstacles(meshcat)

    x_w_guess = None
    x_b_guess = None
    a_guess = None
    theta_guess = None
    v_guess = None
    w_guess = None

    if args.filename:
        print(f"Using {args.filename} as guess")
        x_w_guess, x_b_guess, a_guess, theta_guess, v_guess, w_guess = (
            read_guess_from_file(args.filename)
        )

    """ Show the guess"""
    if x_w_guess is not None:
        x_b_guess_aug = np.concatenate(
            [
                x_b_guess,
                np.ones((x_b_guess.shape[0], 1)) * planning_setting.arm_base_height,
            ],
            axis=1,
        )

        for k in range(N):
            # show_pose(meshcat, x_b_star_aug[k], x_w_star[k], x_e[k], theta_star[k])
            show_pose(
                meshcat,
                x_b_guess_aug[k],
                x_w_guess[k],
                x_e[k],
                theta_guess[k],
                is_persist=True,
            )

    # planning_setting.connect_pybullet()

    x_w_star, x_b_star, theta_star = optimize(
        x_w_guess=x_w_guess,
        x_b_guess=x_b_guess,
        a_guess=a_guess,
        theta_guess=theta_guess,
        v_guess=v_guess,
        w_guess=w_guess,
    )

    # planning_setting.disconnect_pybullet()

    x_b_star_aug = np.concatenate(
        [x_b_star, np.ones((x_b_star.shape[0], 1)) * planning_setting.arm_base_height],
        axis=1,
    )

    output_file = None
    if args.output:
        output_file = open(args.output, "w")

    meshcat.Delete()
    planning_setting.show_obstacles(meshcat)
    for k in range(N):
        # print(f"x_b: {x_b_star_aug[k]}, x_w: {x_w_star[k]}, x_e: {x_e[k]}")
        if output_file is not None:
            output = json.dumps(
                np.hstack(
                    [x_w_star[k], x_e[k], x_b_star_aug[k], [theta_star[k]]]
                ).tolist()
            )
            output_file.write(f"{output}\n")
        if not LOOP_PLAYBACK:
            show_pose(
                meshcat,
                x_b_star_aug[k],
                x_w_star[k],
                x_e[k],
                theta_star[k],
                is_persist=True,
            )

    if LOOP_PLAYBACK:
        while True:
            for k in range(N):
                # print(f"x_b: {x_b_star_aug[k]}, x_w: {x_w_star[k]}, x_e: {x_e[k]}")
                show_pose(meshcat, x_b_star_aug[k], x_w_star[k], x_e[k], theta_star[k])
                time.sleep(dt)
