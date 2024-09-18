import optimize_feasible

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize a trajectory given a feasible guess"
    )
    parser.add_argument(
        "filenames",
        help="Files containing the trajectory guess",
        type=str,
        nargs="+",
    )
    args = parser.parse_args()

    for filename in args.filenames:
        print(f"Using {filename} as guess")
        x_w_guess, x_b_guess, a_guess = optimize_feasible.read_guess_from_file(filename)
        x_w_star, x_b_star, theta_star = optimize_feasible.optimize(
            x_w_guess=x_w_guess, x_b_guess=x_b_guess, a_guess=a_guess
        )
