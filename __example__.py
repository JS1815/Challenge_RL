from simulator import Simulator
from planners import Planner
from problems import HealthcareProblem, ResourceType
from reporter import EventLogReporter, ResourceScheduleReporter
from rl_planner import RLPlanner
import torch

BASE = {
    'wta' : 19.52,
    'wth' : 322.84,
    'nerv': 199.30,
    'cost': 49.42
}
# function that compares my result over the naive planner 
def compare(result, N):
    if N == 0:
        print("No cases finalized. Cannot compare.")
        return
    avgs = {
        'wta' : result['waiting_time_for_admission'] / N,
        'wth' : result['waiting_time_in_hospital']  / N,
        'nerv': result['nervousness']               / N,
        'cost': result['personnel_cost']            / N
    }
    pcts = {k: 100*(avgs[k] - BASE[k]) / BASE[k] for k in BASE}
    final = (pcts['wta'] + pcts['wth'] + pcts['nerv'] + 3*pcts['cost']) / 6

    print("\n=== Comparison to baseline ===")
    print(f"New avgs:  {avgs}")
    print(f"% improve: {pcts}")
    print(f"Final score: {final:.2f}")

if __name__ == '__main__':
    problem = HealthcareProblem()
    planner = RLPlanner(problem, training=False)
    try:
        # Load the weights of the best model found during training
        planner.policy_net.load_state_dict(torch.load('dqn_model.pth'))
        # Set the model to evaluation mode
        planner.policy_net.eval()
        print("Successfully loaded trained model: dqn_model.pth")
    except FileNotFoundError:
        print("ERROR: dqn_model.pth not found!")
        exit() 
    simulator = Simulator(planner, problem)
    result = simulator.run(365 * 24)
    print (result)
    # N = planner.completed_cases_count
    # print(f"Processed Cases: {N} " )
    # compare(result, N)