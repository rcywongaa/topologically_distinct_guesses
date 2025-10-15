import glob
import re
import argparse
import yaml
from statistics import fmean
from natsort import natsorted

parser = argparse.ArgumentParser()
parser.add_argument(
    "directory",
    type=str,
    default="randomized_trials",
)

args = parser.parse_args()

directory = args.directory

def regex_glob(directory, pattern):
    files = glob.glob(directory + "/*")
    return [f for f in files if re.search(pattern, f)]

def analyze_initial_guesses(num_trials):
    num_nags_trajectories = 0
    num_ompl_trajectories = 0
    for i in range(num_trials):
        # nags_trajectory_files = natsorted(glob.glob(f"{directory}/trial{i}/nags_results/trajectory[0-9]*.txt"))
        nags_trajectory_files = natsorted(regex_glob(f"{directory}/trial{i}/nags_results", r"trajectory[0-9]*\.txt"))
        ompl_trajectory_files = natsorted(regex_glob(f"{directory}/trial{i}/ompl_results", r"trajectory[0-9]*\.txt"))
        num_nags_trajectories += len(nags_trajectory_files)
        num_ompl_trajectories += len(ompl_trajectory_files)
    print("Average number of NAGS trajectories:", num_nags_trajectories / num_trials)
    print("Average number of OMPL trajectories:", num_ompl_trajectories / num_trials)

'''
Returns the (result, optimal_cost)
where result is True if any trajectory was successful, and optimal_cost is the minimum cost of the successful trajectories.
'''
def analyze_opt_results(trajectory_files):
    result = False
    optimal_cost = float('inf')
    optimal_cost_not_first = False
    first_optimal_cost = None
    for idx, file in enumerate(trajectory_files):
        with open(file, 'r') as f:
            data = yaml.safe_load(f)
            if data["Success"]:
                result = True
                cost = data["Optimal cost"]
                if cost < optimal_cost:
                    optimal_cost = cost
                    if idx != 0:
                        optimal_cost_not_first = True
                    else:
                        first_optimal_cost = cost
    return (result, optimal_cost, optimal_cost_not_first, first_optimal_cost)

nags_success_rate_per_trial = []
ompl_success_rate_per_trial = []
nags_improvement_over_ompl_per_trial = []
nags_optimal_not_first_per_trial = []
nags_improvement_over_dijkstra_per_trial = []
nags_better_count = 0
num_trials = len(glob.glob(f"{directory}/trial*"))
for i in range(num_trials):
    nag_trajectory_files = natsorted(glob.glob(f"{directory}/trial{i}/nags_results/trajectory*_opt_results.txt"))
    ompl_trajectory_files = natsorted(glob.glob(f"{directory}/trial{i}/ompl_results/trajectory*_opt_results.txt"))

    if not nag_trajectory_files and not ompl_trajectory_files:
        print(f"No trajectory data found for trial {i}. Skipping...")
        continue

    nag_success, nag_optimal_cost, nag_optimal_cost_not_first, nag_first_optimal_cost = analyze_opt_results(nag_trajectory_files)
    nags_success_rate_per_trial.append(nag_success)
    nags_optimal_not_first_per_trial.append(nag_optimal_cost_not_first)

    ompl_success, ompl_optimal_cost, _, _ = analyze_opt_results(ompl_trajectory_files)
    ompl_success_rate_per_trial.append(ompl_success)

    if nag_optimal_cost < ompl_optimal_cost:
        nags_better_count += 1
    improvement_over_ompl = None
    if nag_success and ompl_success:
        improvement_over_ompl = nag_optimal_cost/ompl_optimal_cost
    elif ompl_success:
        # print(f"Trial {i}: Only OMPL succeeded with cost {ompl_optimal_cost}.")
        improvement_over_ompl = 2
    elif nag_success:
        # print(f"Trial {i}: Only NAGS succeeded with cost {nag_optimal_cost}.")
        improvement_over_ompl = 0
    else:
        # print(f"Trial {i}: Neither NAGS nor OMPL succeeded.")
        improvement_over_ompl = 1
    nags_improvement_over_ompl_per_trial.append(improvement_over_ompl)

    improvement_over_dijkstra = None
    if nag_first_optimal_cost is not None:
        improvement_over_dijkstra = nag_optimal_cost / nag_first_optimal_cost
        nags_improvement_over_dijkstra_per_trial.append(improvement_over_dijkstra)
    # else:
    #     improvement_over_dijkstra = 0
        # print(f"Trial {i}: NAGS first guess failed to optimize.")

    # print(f"Trial {i}: NAGS Success: {nag_success}, NAGS Optimal Cost: {nag_optimal_cost}, OMPL Success: {ompl_success}, OMPL Optimal Cost: {ompl_optimal_cost}, NAGS Improvement: {improvement_over_ompl}")

analyze_initial_guesses(num_trials)
print("Total number of trials: ", num_trials)
print("NAGS Overall Success Rate:", fmean(nags_success_rate_per_trial) if nags_success_rate_per_trial else None)
print("OMPL Overall Success Rate:", fmean(ompl_success_rate_per_trial) if ompl_success_rate_per_trial else None)
print("NAGS Improvement per trial:", nags_improvement_over_ompl_per_trial)
print("NAGS Overall Improvement:", fmean(nags_improvement_over_ompl_per_trial) if nags_improvement_over_ompl_per_trial else None)
print("NAGS Optimal Cost Not First:", fmean(nags_optimal_not_first_per_trial) if nags_optimal_not_first_per_trial else None)
# print("NAGS Improvement Over Dijkstra:", fmean(nags_improvement_over_dijkstra_per_trial) if nags_improvement_over_dijkstra_per_trial else None)
print("NAGS Better %:", nags_better_count/num_trials)
