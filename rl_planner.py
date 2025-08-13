import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque, defaultdict

from my_planner import GeneticHybridPlanner
from problems import ResourceType
from reporter import EventLogReporter
from simulator import EventType

# Hyperparams
BATCH_SIZE = 64
GAMMA = 0.98
EPS_START = 0.8
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 4e-4
# the states we can choose dynamically depending on trends to schedule resources accordingly
STATE_SIZE = 9
ACTION_MAP = {
    0: {'or_day': 3, 'or_night': 1, 'intake_day': 2, 'intake_night': 1, 'er_day': 6, 'er_night': 3}, # cheap
    1: {'or_day': 4, 'or_night': 2, 'intake_day': 3, 'intake_night': 2, 'er_day': 8, 'er_night': 4}, # standard
    2: {'or_day': 5, 'or_night': 2, 'intake_day': 4, 'intake_night': 2, 'er_day': 9, 'er_night': 5}, # more resources
    3: {'or_day': 5, 'or_night': 3, 'intake_day': 4, 'intake_night': 3, 'er_day': 9, 'er_night': 6}, # max resources
    4: {'or_day': 4, 'or_night': 2, 'intake_day': 3, 'intake_night': 2, 'er_day': 9, 'er_night': 7, 'er_weekend': 6}, # er strong weekends
    5: {'or_day': 5, 'or_night': 2, 'intake_day': 3, 'intake_night': 2, 'er_day': 8, 'er_night': 4},   # more OR day, low intake
    6: {'or_day': 4, 'or_night': 2, 'intake_day': 3, 'intake_night': 2, 'er_day': 9, 'er_night': 6, 'er_weekend': 6}, # stronger ER nights
}
ACTION_SIZE = len(ACTION_MAP)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity): self.memory = deque([], maxlen=capacity)
    def push(self, *args): self.memory.append(Transition(*args))
    def sample(self, batch_size): return random.sample(self.memory, batch_size)
    def __len__(self): return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.l1 = nn.Linear(n_obs, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, n_actions)
    def forward(self, x):
        x = F.relu(self.l1(x)); x = F.relu(self.l2(x)); return self.l3(x)

class RLPlanner(GeneticHybridPlanner):
    def __init__(self, problem_ref, training=True):
        dummy_chromosome = [4, 2, 2, 3, 2, 1, 9, 5, 4, 24, 2, 5, 190]
        super().__init__(dummy_chromosome, problem_ref)

        self.unassigned_task_queues = defaultdict(int)


        self.training = training
        self.eventlog_reporter = EventLogReporter("./temp/rl_event_log.csv", ['diagnosis'])

        self.policy_net = DQN(STATE_SIZE, ACTION_SIZE)
        self.target_net = DQN(STATE_SIZE, ACTION_SIZE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0

        # 2-week history for state trends and reward
        self.history = deque(maxlen=2)
        self.history.append({'wta': 20.0, 'backlog': 0, 'cost': 50000.0, 'reward': 0.0})
        self.history.append({'wta': 20.0, 'backlog': 0, 'cost': 50000.0, 'reward': 0.0})

        # KPI aggregation for current week
        self.kpis_this_week = {'wta_total': 0.0, 'wta_count': 0, 'cost': 0.0}

        # trackers
        self.case_first_seen_time = {}   # A/B: arrival time captured via events
        self.backlog = 0                 # admissions backlog (A/B only)

        # help compute daily cost exactly once per day
        self._last_cost_day = None

        self.last_state = None
        self.last_action = None
        self.first_week = True

    def get_weekly_state(self, sim_time):
        last_week, two_weeks_ago = self.history[-1], self.history[-2]

        week_of_year = ((sim_time // 24) % 365) / 52.0

        # queues for the resources 
        q_intake = self.unassigned_task_queues[ResourceType.INTAKE] / 50.0
        q_or     = self.unassigned_task_queues[ResourceType.OR]     / 50.0

        avg_wta   = last_week['wta'] / 24.0
        backlog   = last_week['backlog'] / 100.0
        wta_trend = (last_week['wta'] - two_weeks_ago['wta']) / 24.0
        back_tr   = (last_week['backlog'] - two_weeks_ago['backlog']) / 100.0
        rew_trend = last_week['reward'] - two_weeks_ago['reward']

        last_act = ACTION_MAP[self.last_action.item()] if self.last_action is not None else ACTION_MAP[0]
        or_day   = last_act['or_day'] / 5.0

        vec = [week_of_year, avg_wta, backlog, q_intake, q_or, wta_trend, back_tr, rew_trend, or_day]
        return torch.tensor([np.clip(v, -5, 5) for v in vec], dtype=torch.float32).unsqueeze(0)

    def schedule(self, simulation_time):
        # this was originally once a week
        is_decision_time = (simulation_time % 24 ==18) 

        if is_decision_time and self.training:
            current_state = self.get_weekly_state(simulation_time)

            if not self.first_week:
                # compute weekly reward from deltas
                wta_this_week  = self.kpis_this_week['wta_total'] / max(1, self.kpis_this_week['wta_count'])
                cost_this_week = self.kpis_this_week['cost']
                last_week = self.history[-1]

                wta_change     = last_week['wta']     - wta_this_week
                backlog_change = last_week['backlog'] - self.backlog
                cost_change    = (last_week['cost']   - cost_this_week) / 5000.0

                reward_val = (0.5 * wta_change) + (1.0 * backlog_change) + (0.75 * cost_change)
                reward = torch.tensor([reward_val], dtype=torch.float32)

                self.memory.push(self.last_state, self.last_action, current_state, reward)

                self.history.append({'wta': wta_this_week, 'backlog': self.backlog,
                                     'cost': cost_this_week, 'reward': reward_val})

            # choose action
            action_tensor = self.select_action(current_state)
            weekly_policy = ACTION_MAP[action_tensor.item()]
            self.policy.update(weekly_policy)
            self.policy['or_weekend']     = weekly_policy.get('or_weekend',     max(1, weekly_policy['or_night'] - 1))
            self.policy['intake_weekend'] = weekly_policy.get('intake_weekend', 1)
            self.policy['er_weekend']     = weekly_policy.get('er_weekend',     max(2, weekly_policy['er_night'] - 1))

            self.last_state  = current_state
            self.last_action = action_tensor
            self.first_week  = False
            self.kpis_this_week = {'wta_total': 0.0, 'wta_count': 0, 'cost': 0.0}

            self.optimize_model()

        #  Cost accounting: add once per day (e.g., at 00:00) 
        day = simulation_time // 24
        if self._last_cost_day is None or day > self._last_cost_day:
            todays_schedule = super().schedule(simulation_time)
            staffed = 0
            for res_type, _, count in todays_schedule:
                if res_type not in (ResourceType.A_BED, ResourceType.B_BED):
                    staffed += count
            # bill 24 hours for the staffed headcount once per day
            self.kpis_this_week['cost'] += staffed * 24
            self._last_cost_day = day
            return todays_schedule

        # otherwise, just return the parent schedule without re-billing
        return super().schedule(simulation_time)

    def report(self, case_id, element, timestamp, resource, lifecycle_state, data=None):
        # keep parent logic (queue tracking, etc.) and event logging
        super().report(case_id, element, timestamp, resource, lifecycle_state, data)
        self.eventlog_reporter.callback(case_id, element, timestamp, resource, lifecycle_state, data)

        if element is not None:
            lbl = getattr(element, "label", None)
            # map labels to resource types we care about for the state
            if lbl == 'intake':
                rtype = ResourceType.INTAKE
            elif lbl == 'surgery':
                rtype = ResourceType.OR
            elif lbl == 'ER_treatment':
                rtype = ResourceType.ER_PRACTITIONER
            else:
                rtype = None  

            if rtype is not None:
                if lifecycle_state == EventType.ACTIVATE_TASK:
                    self.unassigned_task_queues[rtype] += 1
                elif lifecycle_state == EventType.START_TASK:
                    # guard against negative
                    if self.unassigned_task_queues[rtype] > 0:
                        self.unassigned_task_queues[rtype] -= 1

        # A/B admissions backlog bookkeeping using only events:
        if lifecycle_state == EventType.CASE_ARRIVAL and element and element.case_type in ('A', 'B'):
            # first time we see the case -> save arrival
            if case_id not in self.case_first_seen_time:
                self.case_first_seen_time[case_id] = timestamp
            # this is a new referral case -> backlog++
            self.backlog += 1

        # When intake starts, compute WTA and backlog
        if lifecycle_state == EventType.START_TASK and element and element.label == 'intake':
            start = timestamp
            arrive = self.case_first_seen_time.get(case_id, start)
            self.kpis_this_week['wta_total'] += (start - arrive)
            self.kpis_this_week['wta_count'] += 1
            # remove from backlog if it was a referral case
            if case_id in self.case_first_seen_time:
                self.backlog = max(0, self.backlog - 1)

    # RL helpers 
    def select_action(self, state):
        sample = random.random()
        eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1.0 * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if (sample > eps) or (not self.training):
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1,1)
        else:
            return torch.tensor([[random.randrange(ACTION_SIZE)]], dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE: return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(s is not None for s in batch.next_state), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch  = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        q_sa = self.policy_net(state_batch).gather(1, action_batch)

        next_vals = torch.zeros(BATCH_SIZE)
        with torch.no_grad():
            next_vals[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        target = reward_batch + GAMMA * next_vals
        loss = F.smooth_l1_loss(q_sa, target.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        td, pd = self.target_net.state_dict(), self.policy_net.state_dict()
        for k in pd: td[k] = pd[k] * TAU + td[k] * (1 - TAU)
        self.target_net.load_state_dict(td)
