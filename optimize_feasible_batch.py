import optimize_feasible
import concurrent.futures


def process(filename):
    print(f"Using {filename} as guess")
    (
        x_w_guess,
        x_b_guess,
        a_guess,
        theta_guess,
        v_guess,
        w_guess,
        delta_x_w_guess,
    ) = optimize_feasible.read_guess_from_file(filename)

    optimize_feasible.save_image(
        filename.replace(".txt", "_init.png"),
        x_w_guess,
        x_b_guess,
        theta_guess,
    )

    x_w_star, x_b_star, theta_star, (result, solve_time) = optimize_feasible.optimize(
        x_w_guess=x_w_guess,
        x_b_guess=x_b_guess,
        a_guess=a_guess,
        theta_guess=theta_guess,
        v_guess=v_guess,
        w_guess=w_guess,
        delta_x_w_guess=delta_x_w_guess,
    )

    optimize_feasible.output_to_file(
        filename.replace(".txt", "_opt.txt"),
        x_w_star,
        x_b_star,
        theta_star,
    )

    optimize_feasible.save_image(
        filename.replace(".txt", "_opt.png"),
        x_w_star,
        x_b_star,
        theta_star,
    )

    with open(filename.replace(".txt", "_opt_results.txt"), "w") as f:
        f.write(f"Success: {result.is_success()}\n")
        f.write(f"Optimal cost: {result.get_optimal_cost()}\n")
        f.write(f"Solve time: {solve_time}s\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize a trajectory given a feasible guess"
    )
    parser.add_argument(
        "filenames", help="File containing the trajectory guess", type=str, nargs="+"
    )
    args = parser.parse_args()

    # with concurrent.futures.ThreadPoolExecutor(10) as executor:
    #     futures = [executor.submit(process, filename) for filename in args.filenames]
    #     concurrent.futures.wait(futures)

    for filename in args.filenames:
        process(filename)
