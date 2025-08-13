from enum import Enum, auto
from sortedcontainers import SortedList
from collections import defaultdict


class EventType(Enum):
	CASE_ARRIVAL = auto()
	ACTIVATE_TASK = auto()
	ACTIVATE_EVENT = auto()
	START_TASK = auto()
	COMPLETE_TASK = auto()
	COMPLETE_EVENT = auto()
	PLAN_EVENTS = auto()
	COMPLETE_CASE = auto()
	SCHEDULE_RESOURCES = auto()
	ASSIGN_RESOURCES = auto()
	REGULAR_PLANNING_MOMENT = auto()


class SimulationEvent:
	def __init__(self, event_type, moment, element, resource=None):
		self.event_type = event_type
		self.moment = moment
		self.element = element
		self.resource = resource

	def __lt__(self, other):
		return self.moment < other.moment

	def __str__(self):
		return str(self.event_type) + "\t(" + str(round(self.moment, 2)) + ")\t" + str(self.element) + "," + str(self.resource)


class ResourceSchedule:
	'''
	Contains the schedule of the resources as the number of resources of each type that is available at a moment.
	The schedule is a dict of resource_type -> list of (hour, number of resources available from that hour).
	Resources of a specific type are available with in a specific number from the hour registered in the dictionary until, but not including, the next hour registered in the schedule.
	I.e., for a specific resource type t, there is a (virtual) list [..., (h_i, n_i), (h_i+1, n_i+1), ...], with h_i strictly increasing. The number of resources of type t available at hour h is n, where h_i <= h < h_i_1 and n = n_i.
	Initially all resources are available.

	We encode the schedule as a dict of resource_type -> (scheduling_moments, number_of_resources_available_from_each_scheduling_moment):
	- scheduling_moments is an ordered list of moments at which the number of resources available changes
	- number_of_resources_available_from_each_scheduling_moment is dict of scheduling_moment -> number of resources available from that moment

	We keep track of the resource cost at each simulation hour, by recording:
	- planned_ahead_count is a list of dicts with the number of resources of each type that were scheduled for now one week ago, for now+1, for now+2 one week ago, ... up until now+157
	- resource_cost is a list of the resource costs at each simulation hour h
	'''
	def __init__(self):
		self.resource_types = defaultdict(lambda : 0)
		self.schedule = dict()
		# kpi variables for the resource cost
		self.planned_ahead_count = []
		self.cost_measurements = []

	def init_schedule(self, resources):
		'''
		Initializes the schedule with all resource types maximally available.
		:param resources: list of Resource objects that are available
		'''		
		for resource in resources:
			self.resource_types[resource.type] += 1
		for resource_type in self.resource_types:
			scheduling_moments = SortedList([0])
			number_of_resources_available_from_each_scheduling_moment = {0: self.resource_types[resource_type]}
			self.schedule[resource_type] = (scheduling_moments, number_of_resources_available_from_each_scheduling_moment)
		self.planned_ahead_count = [self.resource_types.copy() for _ in range(158)]

	def get_number_of_resources(self, resource_type, moment):
		'''
		Returns the number of resources of the given type that are available at the given moment.
		Precondition: moment >= 0, resource_type in self.resource_types
		:param resource_type: the type of resource for which the number of resources available is requested
		:param moment: the moment at which the number of resources available is requested
		:return: the number of resources of the given type that are available at the given moment		
		'''
		scheduling_moments, number_of_resources_available_from_each_scheduling_moment = self.schedule[resource_type]
		scheduling_index = scheduling_moments.bisect_right(moment) - 1
		return number_of_resources_available_from_each_scheduling_moment[scheduling_moments[scheduling_index]]

	def get_current_resources(self, moment):
		'''
		Returns the number of resources of each type that are available at the given moment.
		Precondition: moment >= 0
		:param moment: the moment at which the number of resources available is requested
		:return: a dict of resource_type -> number of resources of that type that are available at the given moment
		'''
		current_resources = dict()
		for resource_type in self.schedule:
			current_resources[resource_type] = self.get_number_of_resources(resource_type, moment)
		return current_resources

	def add_scheduling_moment(self, resource_type, moment, nr_available):
		'''
		Adds a scheduling moment to the schedule for the given resource type.
		If the scheduling moment is already in the schedule, the number of resources available at that moment is updated.
		:param resource_type: the type of resource for which the scheduling moment is added
		:param moment: the moment at which the number of resources available changes
		:param nr_available: the number of resources of the given type that are available from the given moment
		'''
		scheduling_moments, number_of_resources_available_from_each_scheduling_moment = self.schedule[resource_type]
		scheduling_index = scheduling_moments.bisect_right(moment) - 1
		if scheduling_moments[scheduling_index] == moment:
			number_of_resources_available_from_each_scheduling_moment[moment] = nr_available
		else:
			scheduling_moments.add(moment)
			number_of_resources_available_from_each_scheduling_moment[moment] = nr_available

	def add_cost_measurement(self, now, busy_resources):
		'''
		Adds a cost measurement. This method must be called each (simulation) hour, i.e. at SCHEDULE_RESOURCES.
		The class keeps a list of resources that were planned one week ahead for the current hour h, for h+1, for h+1, ... up until h+157.
		It calculates the cost of the resources that are scheduled at that moment as follows:
		- get the number of resources that were scheduled for now one week ago, they cost 1
		- get the number of resources that are scheduled for now, they cost 2 insofar there are more than the number of resources that were scheduled for now one week ago
		- the number of busy_resources cost 3 insofar they are more than the number of resources that are scheduled for now
		Finally, we rotate the list of resources that were planned one week ahead, i.e. we remove the one planned for hour h and add the one planned for h+158 (which will be h+157 in the next invocation).
		'''
		# get the number of resources that were scheduled for now one week ago
		planned_ahead = self.planned_ahead_count[0]
		cost = 0
		for resource_type in planned_ahead:
			cost += planned_ahead[resource_type]
		# get the number of resources that are scheduled for now
		current_resources = self.get_current_resources(now)
		for resource_type in current_resources:
			cost += 2 * (current_resources[resource_type] - planned_ahead[resource_type])
		# get the number of busy resources
		busy_resource_count = dict()
		for resource in busy_resources:
			if resource.type not in busy_resource_count:
				busy_resource_count[resource.type] = 1
			else:
				busy_resource_count[resource.type] += 1
		for resource_type in busy_resource_count:
			cost += 3 * (max(0, busy_resource_count[resource_type] - current_resources[resource_type]))
		# add the cost to the list of costs
		self.cost_measurements.append(cost)
		# rotate the list of resources that were planned one week ahead
		self.planned_ahead_count.pop(0)
		self.planned_ahead_count.append(self.get_current_resources(now + 158))
	
	def get_total_cost(self):
		'''
		Returns the total cost of the resources that were scheduled.
		:return: the total cost of the resources that were scheduled
		'''
		return sum(self.cost_measurements)


class Simulator:
	def __init__(self, planner, problem):
		self.events = []  # list of tuples (planned moment, simulationevent)
		self.planned_events = []  # list of tuples (time of planning, simulationevent)
		self.unassigned_tasks = dict()  # dictionary of unassigned tasks id -> task
		self.assigned_tasks = dict()  # dictionary of assigned tasks id -> (task, resource, moment of assignment)
		self.available_resources = set()  # set of available resources
		self.away_resources = []  # list of resources that are unavailable, because they are away
		self.busy_resources = dict()  # dictionary of busy resources resource -> (task they are busy on, moment they started on the task)
		self.busy_cases = dict()  # dictionary of busy cases case_id -> list of ids of elements that are planned in self.events for the case
		self.now = 0  # current moment in the simulation
		self.finalized_cases = 0  # number of cases that have been finalized
		self.total_cycle_time = 0  # sum of cycle times of finalized cases
		self.case_start_times = dict()  # dictionary of case_id -> moment the case started
		self.task_start_end_times = dict()
		self.event_times = dict()
		self.problem = problem  # the problem to be simulated
		self.planner = planner  # the planner to be used for planning events
		self.running_time = 0  # the time the simulation is run for
		self.schedule = ResourceSchedule()

		problem.set_simulator(self)
		self.init_simulation()

	def restart(self):
		self.events = []
		self.planned_events = []
		self.unassigned_tasks = dict()
		self.assigned_tasks = dict()
		self.available_resources = set()
		self.away_resources = []
		self.busy_resources = dict()
		self.busy_cases = dict()
		self.now = 0
		self.finalized_cases = 0
		self.total_cycle_time = 0
		self.case_start_times = dict()
		self.task_start_end_times = dict()
		self.event_times = dict()
		self.running_time = 0
		self.schedule = ResourceSchedule()

		self.problem.restart()
		self.init_simulation()

	def sort_events(self):
		"""
		First start tasks (i.e. use resources) before another COMPLETE_EVENT comes into action
		"""
		self.events.sort(key = lambda k : (k[0], # time
									 1 if k[1].event_type == EventType.COMPLETE_EVENT else
									 0)
		)

	def init_simulation(self):
		"""
		Initializes the simulation by:
		- setting the available resources to all resources in the problem
		- adding the first case arrival event to the events list
		- setting the first regular planning moment
		- restarting the problem
		"""
		for r in self.problem.resources:
			self.available_resources.add(r)
		self.schedule.init_schedule(self.problem.resources)
		self.problem.restart()
		(t, task) = self.problem.next_case()
		self.events.append((t, SimulationEvent(EventType.CASE_ARRIVAL, t, task)))
		next_planning_moment = self.problem.next_regular_planning_moment(0)
		self.events.append((next_planning_moment, SimulationEvent(EventType.REGULAR_PLANNING_MOMENT, next_planning_moment, None)))
		self.events.append((0, SimulationEvent(EventType.SCHEDULE_RESOURCES, 0, None)))
		self.sort_events()

	def cancel(self, case_id, event_label):
		"""
		Cancels an event for a case with a certain label by removing it from the events list.
		"""
		found_index = None
		for i in range(len(self.events)):
			if self.events[i][1].element is not None and self.events[i][1].element.case_id == case_id and self.events[i][1].element.label == event_label:
				found_index = i
		if found_index is not None:
			event = self.events.pop(found_index)
			# also remove the element from the busy case
			self.busy_cases[case_id] = list(filter(
				lambda k : k.id != event[1].element.id,
				self.busy_cases[case_id]
			))

	def is_planning_slot(self, time):
		"""
		Returns whether the given time is the time of a planning slot.
		There are planning slots every half hour between 8:00 and 15:00 (inclusive) on weekdays.
		"""
		time_in_week = time % (24 * 7)
		hour_of_day_x_10 = round((time_in_week % 24)*10)  # we multiply by 10 to avoid floating point errors
		#day_of_week = round(time_in_week // 24)  # 0 is Monday, 1 is Tuesday, ..., 6 is Sunday
		day_of_week = True
		return hour_of_day_x_10 >= 80 and hour_of_day_x_10 <= 150 and day_of_week < 5 and hour_of_day_x_10 % 5 == 0

	def replan(self, element, time):
		"""
		replans the specified element at the specified time
		"""
		self.cancel(element.case_id, element.label)
		element.occurrence_time = time
		self.events.append((time, SimulationEvent(EventType.COMPLETE_EVENT, time, element)))
		self.sort_events()

	def activate(self, element, time):
		"""
		Activates an element.
		For an event that means scheduling the completion of the event for the moment at which it happens.
		For a task that means scheduling the assignment of resources immediately.
		"""
		element.activation_time = time
		self.busy_cases[element.case_id].append(element)
		if element.is_event():
			self.planner.report(element.case_id, element, self.now, None, EventType.ACTIVATE_EVENT)
			self.events.append((element.occurrence_time, SimulationEvent(EventType.COMPLETE_EVENT, element.occurrence_time, element)))
		elif element.is_task():
			self.planner.report(element.case_id, element, self.now, None, EventType.ACTIVATE_TASK)
			self.unassigned_tasks[element.id] = element
			self.events.append((self.now, SimulationEvent(EventType.ASSIGN_RESOURCES, self.now, None)))
		self.events.append((self.now, SimulationEvent(EventType.PLAN_EVENTS, self.now, None)))

	def run(self, running_time=24*365):
		"""
		Runs the simulation for the specified amount of time.
		"""
		nr_resources_scheduled_now = dict()
		nr_resources_working_now = dict()
		self.running_time = running_time
		while self.now <= running_time:
			(self.now, event) = self.events.pop(0)

			if event.event_type == EventType.CASE_ARRIVAL:				
				self.planner.report(event.element.case_id, None, self.now, None, EventType.CASE_ARRIVAL)  # report CASE_ARRIVAL
				# create the case
				self.case_start_times[event.element.case_id] = self.now
				self.busy_cases[event.element.case_id] = []
				# activate the first element
				self.activate(event.element, self.now)
				# schedule the next case arrival
				(t, task) = self.problem.next_case()
				self.events.append((t, SimulationEvent(EventType.CASE_ARRIVAL, t, task)))

			elif event.event_type == EventType.START_TASK:
				self.task_start_end_times[event.element] = [self.now, 0]
				self.planner.report(event.element.case_id, event.element, self.now, event.resource, EventType.START_TASK) # report START_TASK
				self.problem.start_task(event.element)
				# start the task
				self.busy_resources[event.resource] = (event.element, self.now)
				# schedule the completion of the task
				t = self.now + self.problem.processing_time_sample(event.resource, event.element, self.now)
				self.events.append((t, SimulationEvent(EventType.COMPLETE_TASK, t, event.element, event.resource)))				

			elif event.event_type == EventType.COMPLETE_EVENT \
			  		or event.event_type == EventType.COMPLETE_TASK:
				self.planner.report(event.element.case_id, event.element, self.now, event.resource, event.event_type) # report COMPLETE_EVENT or COMPLETE_TASK
				# for tasks, process the resource that performed the task
				if event.event_type == EventType.COMPLETE_TASK:
					self.task_start_end_times[event.element][1] = self.now
					del self.busy_resources[event.resource]
					if nr_resources_working_now[event.resource.type] > nr_resources_scheduled_now[event.resource.type]: # if there are too many resources of that type, send it away
						self.away_resources.append(event.resource)
					else:
						self.available_resources.add(event.resource)
						self.events.append((self.now, SimulationEvent(EventType.ASSIGN_RESOURCES, self.now, None)))  # if a resource becomes available, it can be assigned, so we schedule the assignment of resources
					del self.assigned_tasks[event.element.id]
				else:
					self.event_times[event.element] = self.now

				# complete the element
				self.busy_cases[event.element.case_id] = list(filter(
					lambda k : k.id != event.element.id,
					self.busy_cases[event.element.case_id]
				))
				next_elements = self.problem.complete_element(event.element)
				# activate the next elements
				for next_element in next_elements:  
					self.activate(next_element, self.now)
				# if the case is done, complete the case
				if len(self.busy_cases[event.element.case_id]) == 0:					
					self.events.append((self.now, SimulationEvent(EventType.COMPLETE_CASE, self.now, event.element)))

			elif event.event_type == EventType.SCHEDULE_RESOURCES:
				# check how many resources of each type there are supposed to be now
				nr_resources_scheduled_now = self.schedule.get_current_resources(self.now)
				# check how many resources of each type there are now (this is the maximum number of resources according to the schedule, minus the away resources - it is not the available resources, because those do not include the busy resources)
				nr_resources_working_now = self.schedule.resource_types.copy()
				for resource in self.away_resources:
					nr_resources_working_now[resource.type] -= 1
				# for each away resource: if there are too few of that type (i.e. nr_resources_working_now[type] < nr_resources_scheduled_now[type]), add it to the available resources.
				resources_to_add = []
				for resource in self.away_resources:
					if nr_resources_working_now[resource.type] < nr_resources_scheduled_now[resource.type]:						
						resources_to_add.append(resource)
						nr_resources_working_now[resource.type] += 1
				for resource in resources_to_add:
					self.away_resources.remove(resource)
					self.available_resources.add(resource)
				if len(resources_to_add) > 0:
					self.events.append((self.now, SimulationEvent(EventType.ASSIGN_RESOURCES, self.now, None)))  # if a resource becomes available, it can be assigned, so we schedule the assignment of resources
				# for each available resource: if there are too many of that type (i.e. nr_resources_working_now[type] > nr_resources_scheduled_now[type]), remove it from the available resources.
				resources_to_remove = []
				for resource in self.available_resources:
					if nr_resources_working_now[resource.type] > nr_resources_scheduled_now[resource.type]:
						resources_to_remove.append(resource)
						nr_resources_working_now[resource.type] -= 1
				for resource in resources_to_remove:
					self.available_resources.remove(resource)
					self.away_resources.append(resource)
				# report the resource cost
				self.schedule.add_cost_measurement(self.now, self.busy_resources.keys())
				# schedule the next resource check
				self.events.append((self.now + 1, SimulationEvent(EventType.SCHEDULE_RESOURCES, self.now + 1, None)))
				# report the number of resources of each type
				data = dict()
				data['available_resources'] = len(self.available_resources)
				data['busy_resources'] = len(self.busy_resources)
				data['away_resources'] = len(self.away_resources)
				self.planner.report(None, None, self.now, None, EventType.SCHEDULE_RESOURCES, data)  # report SCHEDULE_RESOURCES

			elif event.event_type == EventType.ASSIGN_RESOURCES:
				# assign resources to tasks
				if len(self.unassigned_tasks) > 0 and len(self.available_resources) > 0:
					assignments = self.problem.assign_resources(self.unassigned_tasks, self.available_resources)
					for (task, resource) in assignments:
						self.events.append((self.now, SimulationEvent(EventType.START_TASK, self.now, task, resource)))
						del self.unassigned_tasks[task.id]
						self.assigned_tasks[task.id] = (task, resource, self.now)
						self.available_resources.remove(resource)

			elif event.event_type == EventType.REGULAR_PLANNING_MOMENT:
				# schedule event planning for now
				self.events.append((self.now, SimulationEvent(EventType.PLAN_EVENTS, self.now, None)))
				# we also schedule the resources at the regular planning moment
				scheduled_resources = self.planner.schedule(self.now)
				for (resource_type, t, nr) in scheduled_resources:
					self.problem.check_resource_schedule(resource_type, t, nr)
					self.schedule.add_scheduling_moment(resource_type, t, nr)					
				# schedule the next regular planning moment
				next_planning_moment = self.problem.next_regular_planning_moment(self.now)
				self.events.append((next_planning_moment, SimulationEvent(EventType.REGULAR_PLANNING_MOMENT, next_planning_moment, None)))

			elif event.event_type == EventType.PLAN_EVENTS:				
				# plan events
				# is done each time an element is activated and there are events to plan
				if len(self.problem.can_plan)>0:
					plannable_elements = self.problem.can_plan
					replannable_elements = self.problem.can_replan
					planned_events = self.planner.plan(list(plannable_elements.keys()), list(replannable_elements.keys()), self.now)
					for planned_element in planned_events:
						if planned_element[0] in plannable_elements:
							if len(plannable_elements[planned_element[0]]) != 1:
								raise ValueError("At this stage, we only allow for planning of one event at a time.")						
							planned_element = self.problem.plan(planned_element[0], plannable_elements[planned_element[0]][0], planned_element[1])
							self.planned_events.append([self.now,planned_element])
							if not planned_element.is_event():
								raise ValueError("At this stage, we only allow for planning of events.")
							self.activate(planned_element, self.now)
						elif planned_element[0] in replannable_elements:
							case_id = planned_element[0]
							time = planned_element[1]
							if len(replannable_elements[case_id]) != 1:
								raise ValueError("At this stage, we only allow for planning of one event at a time.")
							element_label = next(iter(replannable_elements[case_id]))
							replanned_event = self.problem.plan(case_id, element_label, time)
							self.planned_events.append([self.now,replanned_event])
							if not replanned_event.is_event():
								raise ValueError("At this stage, we only allow for planning of events.")
							self.replan(replanned_event, time)
						else:
							raise ValueError("Element not in plannable or replannable elements.")

			elif event.event_type == EventType.COMPLETE_CASE:
				self.planner.report(event.element.case_id, None, self.now, None, EventType.COMPLETE_CASE)  # report COMPLETE_CASE
				self.total_cycle_time += self.now - self.case_start_times[event.element.case_id]
				self.finalized_cases += 1
				del self.busy_cases[event.element.case_id]
			self.sort_events()

		score = self.problem.evaluate()
		return score
