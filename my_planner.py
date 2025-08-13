from ortools.sat.python import cp_model
from planners import Planner
from problems import ResourceType
from simulator import EventType
from collections import defaultdict
from reporter import EventLogReporter
import random;
import math;

MAX_OR = 5
MAX_INTAKE = 4
MAX_A_BEDS = 30
MAX_B_BEDS = 40
MAX_ER_PRACTITIONERS = 9

TREATMENT_PATHWAYS = {
    'A_surgery': [
        {'label': 'intake', 'resource': ResourceType.INTAKE, 'duration': 1.0},
        {'label': 'surgery', 'resource': ResourceType.OR, 'duration': 2.0},      
        {'label': 'nursing', 'resource': ResourceType.A_BED, 'duration': 39.0}   
    ],
    'A_simple': [
        {'label': 'intake', 'resource': ResourceType.INTAKE, 'duration': 1.0},
        {'label': 'nursing', 'resource': ResourceType.A_BED, 'duration': 39.0}  
    ],
    'B_surgery': [
        {'label': 'intake', 'resource': ResourceType.INTAKE, 'duration': 1.0},
        {'label': 'surgery', 'resource': ResourceType.OR, 'duration': 4.0},   
        {'label': 'nursing', 'resource': ResourceType.B_BED, 'duration': 60.0}  
    ],
    'B_simple': [
        {'label': 'intake', 'resource': ResourceType.INTAKE, 'duration': 1.0},
        {'label': 'nursing', 'resource': ResourceType.B_BED, 'duration': 60.0}  
    ]
}

class GeneticHybridPlanner(Planner, EventLogReporter):
    def __init__(self, chromosome, problem_ref):
        super().__init__("./temp/current_best_run.csv", data_types=['diagnosis'])

        if not hasattr(chromosome, '__len__'):
            raise TypeError(
                f"The 'chromosome' passed to the planner must be a list or tuple, "
                f"but its type was {type(chromosome)}. Value: {chromosome}"
            )
        
        self.patient_data = {}  

        self.policy = {
                'or_wd_day':         int(chromosome[0]), 'or_wd_night':       int(chromosome[1]),
                'or_weekend':        int(chromosome[2]), 'intake_wd_day':     int(chromosome[3]),
                'intake_wd_night':   int(chromosome[4]), 'intake_weekend':    int(chromosome[5]),
                'er_wd_day':         int(chromosome[6]), 'er_wd_night':       int(chromosome[7]),
                'er_weekend':        int(chromosome[8]), 'fallback_plan_hours': int(chromosome[9]),
                'cp_lookahead_weeks':int(chromosome[10]), 'bed_buffer_percentage': int(chromosome[11]),
                'wip_limit':         int (chromosome[12])
            }
        
        self.case_arrival_time = {}
        self.backlog = 0
        self.unprocessed_patients = 0
        self.completed_cases_count = 0
        self.active_cases_count = 0

    def report(self, case_id, element, timestamp, resource, lifecycle_state, data=None):

        super().callback(case_id, element, timestamp, resource, lifecycle_state, data)
        # backlog counter when patient arrives
        # only increase for A and B patients 
        if lifecycle_state == EventType.CASE_ARRIVAL and element and element.case_type in ('A','B'):
            self.case_arrival_time[case_id] = timestamp
            self.backlog += 1
        elif lifecycle_state == EventType.COMPLETE_CASE:
            self.completed_cases_count +=1
            self.active_cases_count = max(0, self.active_cases_count -1)
        elif lifecycle_state== EventType.START_TASK and element.label == 'intake' :
            self.active_cases_count += 1

            
    def schedule(self, simulation_time):

        hour_of_day = simulation_time % 24
        day_of_week = (simulation_time // 24) % 7 
        is_weekday = day_of_week < 5

        if is_weekday and (8 <= hour_of_day < 18):
            base_or, base_intake, base_er = self.policy['or_wd_day'], self.policy['intake_wd_day'], self.policy['er_wd_day']
        elif is_weekday:
            base_or, base_intake, base_er = self.policy['or_wd_night'], self.policy['intake_wd_night'], self.policy['er_wd_night']
        else: # Weekend
            base_or, base_intake, base_er = self.policy['or_weekend'], self.policy['intake_weekend'], self.policy['er_weekend']

        # reactive backlog logic
        # old piece of code that was crucial before the calculation of backlog was corrected
        # usually never reach backlog values > 20 so dead code
        final_or, final_intake, final_er = base_or, base_intake, base_er
        if self.backlog > 50:
            final_or, final_intake = math.ceil(base_or * 2), math.ceil(base_intake * 2)
            print(f"We have reached a backlog of over 50 setting resources to max to handle it!")
        elif self.backlog > 20:
            final_or, final_intake = math.ceil(base_or * 1.2), math.ceil(base_intake * 1.2)
            print(f"We have reached a backlog of over 20, increase resources to handle!")

        # ensure we dont go over physical limit
        final_or = min(final_or, MAX_OR); final_intake = min(final_intake, MAX_INTAKE); final_er = min(final_er, MAX_ER_PRACTITIONERS)

        schedule_time_tomorrow = simulation_time + 158

        schedule = [
            (ResourceType.OR,               schedule_time_tomorrow, final_or),
            (ResourceType.INTAKE,           schedule_time_tomorrow, final_intake),
            (ResourceType.ER_PRACTITIONER,  schedule_time_tomorrow, final_er),
            (ResourceType.A_BED,            schedule_time_tomorrow, MAX_A_BEDS),
            (ResourceType.B_BED,            schedule_time_tomorrow, MAX_B_BEDS),
        ]

        return schedule


    def plan(self, cases_to_plan, cases_to_replan, simulation_time):
        # We ignore cases_to_replan to keep stability
        # otherwise CPSolver fails and also it is discouraged to replan because of nervousness
        
        if not cases_to_plan:
            return []

        # Prioritize the new arrivals by who has been waiting longest.
        sorted_cases = sorted(cases_to_plan, key=lambda c: self.case_arrival_time.get(c, simulation_time))

        planned_admissions = self.run_cp_planner(sorted_cases, simulation_time)
        self.backlog = max (0, self.backlog - len(planned_admissions))
        return planned_admissions

    def run_cp_planner(self, cases, now):

        wip_limit = self.policy['wip_limit'] # made as chromosome to test best value
        # 2. Calculate how many new patients we can safely admit.
        available_slots = max(0, wip_limit - self.active_cases_count)
        
        # 3. Only send a batch to the solver that fits within our capacity.
        cases_to_solve = cases[:available_slots]

        if not cases_to_solve:
            return [] # Return early if there are no slots to fill.
        buffer_multiplier = 1.0 - (self.policy['bed_buffer_percentage']/ 100.0)
        max_resources = {
            ResourceType.OR: MAX_OR, ResourceType.A_BED: math.floor(MAX_A_BEDS* buffer_multiplier),
            ResourceType.B_BED: math.floor(MAX_B_BEDS* buffer_multiplier), ResourceType.INTAKE: MAX_INTAKE,
            ResourceType.ER_PRACTITIONER: MAX_ER_PRACTITIONERS,
        }
        model = cp_model.CpModel()
        horizon_end = int(now + self.policy['cp_lookahead_weeks'] * 7 * 24)
        min_admission_time = math.ceil(now + 24.001)

        admission_vars = {} # Use dictionary for mapping
        all_intervals = defaultdict(list)
        weighted_admission_times = [] # To prioritize older patients

        for case_id in cases_to_solve:
            diagnosis = self.patient_data.get(case_id, {}).get('diagnosis', 'A')
            pathway_key = None
            if diagnosis.startswith('A'):
                # 48.8% chance of surgery for 'A' patients
                pathway_key = 'A_surgery' if random.random() < 0.49 else 'A_simple'
            elif diagnosis.startswith('B'):
                # 23.5% chance of surgery for 'B' patients
                pathway_key = 'B_surgery' if random.random() < 0.24 else 'B_simple'
            
            pathway = TREATMENT_PATHWAYS.get(pathway_key)
            if not pathway: continue # Skip if no valid pathway found

            admission_time = model.NewIntVar(min_admission_time, horizon_end, f'admission_{case_id}')
            admission_vars[case_id] = admission_time

            # The weight is based on how many days a patient has already waited
            # more important to schedule this patient early than a patient who just arrived
            wait_time = now - self.case_arrival_time.get(case_id, now)
            weight = 1 + int(wait_time / 24) # 1 point for each day of waiting
            weighted_admission_times.append(admission_time * weight)
            # constraints for the solver
            # intake must be finished before surgery starts etc
            previous_task_end = admission_time
            for i, task_info in enumerate(pathway):
                duration = max(1, int(task_info.get('duration', 1)))
                start_var = model.NewIntVar(min_admission_time, horizon_end, f'start_{case_id}_{i}')
                end_var = model.NewIntVar(min_admission_time, horizon_end, f'end_{case_id}_{i}')
                interval_var = model.NewIntervalVar(start_var, duration, end_var, f'interval_{case_id}_{i}')
                all_intervals[task_info['resource']].append(interval_var)
                model.Add(start_var >= previous_task_end)
                previous_task_end = end_var

        for resource_type, intervals in all_intervals.items():
            model.AddCumulative(intervals, [1] * len(intervals), max_resources.get(resource_type, 1))

        model.Minimize(sum(weighted_admission_times))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0

        status = solver.Solve(model)
        
        admissions = []

        fallback_hours = self.policy.get('fallback_plan_hours', 72)
        safe_fallback_time = now + max(25, fallback_hours)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for case_id in cases:
                if case_id in admission_vars and solver.Value(admission_vars[case_id]) is not None:
                    admissions.append((case_id, solver.Value(admission_vars[case_id])))
                else:
                    admissions.append((case_id, safe_fallback_time))
        else:
            # this should usually never happen
            print(f"WARN: CP Solver failed at time {now}. Using safe fallback.")
            for case_id in cases:
                admissions.append((case_id, safe_fallback_time))
        
        return admissions