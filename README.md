# Challenge_RL

# Reinforcement Learning in combination with constraint programming solver

The second solution I implemented is based on a Deep Reinforcement Learning (DRL) agent. The agent is built on top of a Constraint Programming (CP) solver. This combination allows us to minimize across all four KPIs defined in the competition: Waiting time for admission (WTA), Waiting time in hospital (WTH), Nervousness (NERV) and Personnel cost (COST). 

The system is divided into two main layers.

The first layer is the strategic DRL agent that is built on a Deep Q-Network (DQN). It learns an optimal policy for weekly resource allocation. The main difference in comparison to the hybrid Genetic Algorithm in my first submission is that the DRL agent can make dynamic decisions based on the current hospital’s performance (measured via the KPIs) during the simulation. It is able to allocate resources dynamically because of these factors:

State Representation: Once a week, the agent looks at the performance of the hospital. This performance is measured by considering trends such as the length of internal queues, and the change in waiting times and backlog over the previous two weeks. This gives the agent the possibility to evaluate the current state of the hospital and, depending on the situation, increase or decrease resources for the next week. 

Action Space: The agent selects one of 6 pre-defined resource strategies from an action map. The strategies range from a “cheap” mode with minimal resources and staffing to a “max resources” mode to handle spikes in patients. The agent has to learn which strategy is the most effective depending on the current state of the hospital. 

Reward Function: The agent learns how to select a good strategy with a reward signal calculated from the weekly change in performance. It is rewarded for reducing the average waiting times, admission backlog and operational costs. This allows the agent to directly connect its weekly choices to a measurable outcome. 

The second layer is the CP-Solver, focusing on admitting incoming patients. This means that once the DRL agent has chosen a strategy for the week, the actual patient scheduling is handled by the CP-solver from the first submission. 

The main features of the planner using the CP-solver, built using Google-OR Tools, include:

Treatment pathways derived from process mining, a work in progress (WIP) limit to prevent internal bottlenecks building up and controlling the WTH score, ignoring the cases_to_replan list during planning to keep the NERV score down, and a weighted objective function that prioritizes patients who have been waiting for a longer time and thus directly reducing the WTA score. 

By combining the dynamic DRL agent for a weekly scheduling decision with the proven CP-solver for regular patient planning, this solution is designed to minimize the final score across all 4 KPIs. 
