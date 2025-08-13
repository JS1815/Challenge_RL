import torch
import numpy as np
from simulator import Simulator
from problems import HealthcareProblem
from rl_planner import RLPlanner  

BASE = { 'wta' : 19.52, 'wth' : 322.84, 'nerv': 199.30, 'cost': 49.42 }

NUM_EPISODES = 100 
SIMULATION_DAYS = 60

def main():
    best_score = float('inf')
    
    print(f"--- Starting DRL Training: {NUM_EPISODES} episodes, {SIMULATION_DAYS} days each ---")
    
    problem = None
    planner = RLPlanner(problem, training=True)

    for i_episode in range(NUM_EPISODES):

        problem = HealthcareProblem()
        simulator = Simulator(planner, problem)
        
        print(f"\n--- Episode {i_episode + 1}/{NUM_EPISODES} ---")
        
        result = simulator.run(SIMULATION_DAYS * 24)

        # Evaluate the performance of this episode

        N = planner.completed_cases_count
        if N > 10: # Ensure enough cases were processed for a stable score
            avgs = {
                'wta': result['waiting_time_for_admission'] / N,
                'wth': result['waiting_time_in_hospital'] / N,
                'nerv': result['nervousness'] / N,
                'cost': result['personnel_cost'] / N
            }
            pcts = {k: 100 * (avgs[k] - BASE[k]) / BASE[k] for k in BASE}
            final_score = (pcts['wta'] + pcts['wth'] + pcts['nerv'] + 3 * pcts['cost']) / 6
            
            print(f"Episode {i_episode + 1} complete. Final Score: {final_score:.2f}")

            # Save the model if it's the best one so far
            if final_score < best_score:
                best_score = final_score
                torch.save(planner.policy_net.state_dict(), 'dqn_model.pth')
                print(f"  -> New best score! Model saved to dqn_model.pth")
        else:
            print(f"Episode complete. Only {N} cases finalized, cannot score reliably.")

    print("\n DRL Training Complete ")
    print(f"Best score achieved during training: {best_score:.2f}")
    print(" best model has been saved as 'dqn_model.pth'.")
if __name__ == "__main__":
    main()
