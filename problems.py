from enum import Enum, auto, StrEnum
import random
import pickle
from abc import ABC, abstractmethod
import collections
from simulator import EventType
from functools import reduce
import statistics
import numpy as np

import arrival_times

class ElementType(Enum):
	TASK = auto()
	EVENT = auto()


class Element:
    """
    An element is either a task or an event of a process.
    An element is part of a case.
    An element has a:
    - case_id: the id of the case to which the element belongs.
    - case_type: the type of the case to which the element belongs.
    - element_id: the id of the element.
    - label: the label of the element.
    - element_type: the type of the element, which is either a task or an event.
    - data: a dictionary of data that is associated with the element; the dictionary keys are the data types and the values are the data values.
    - occurrence_time: the time when the event should occur in absolute simulation time; used for events only and must be set for each event.
    """
    def __init__(self, case_id, case_type, element_id, label, element_type, occurrence_time=None):
        self.id = element_id
        self.case_id = case_id
        self.label = label
        self.case_type = case_type
        self.element_type = element_type
        self.data = dict()
        self.occurrence_time = occurrence_time  # used for time-based events only, represents the time when the event should occur
        if self.is_event() and self.occurrence_time is None:
            raise ValueError("The occurrence time of an event must be set.")
        if self.is_task() and self.occurrence_time is not None:
            raise ValueError("The occurrence time of a task must not be set.")

    def is_event(self):
        return self.element_type == ElementType.EVENT
    
    def is_task(self):
        return self.element_type == ElementType.TASK
    
    def __str__(self):
        return self.label + "_" + self.case_type + "(" + str(self.case_id) + ")_" + str(self.id) + (str(self.data) if len(self.data) > 0 else "")

class ResourceType(StrEnum):
    OR = "OR"
    A_BED = "A_BED"
    B_BED = "B_BED"
    INTAKE = "INTAKE"
    ER_PRACTITIONER = "ER_PRACTITIONER"


class Resource:
    """
    """
    def __init__(self, type, id):
        self.type = type
        self.id = id

    def __str__(self):
        return str(self.id)

class Problem(ABC):

    def __init__(self):
        self.simulator = None
        self.resources = []  # a list of resources that can be used in the problem.
        self.case_types = []  # a list of case types that can be generated in the problem.
        
        self.can_plan = dict()  # a dictionary of case_id to list of element labels that can be planned for that case.
        self.can_replan = dict()  # a dictionary of dictionaries: case_id -> element_label -> element that can be replanned.
        self.next_case_id = 0  # the id of the next case to be generated.
        self.next_case_arrival_time = dict()  # a dictionary of case type to the time of the next case arrival of that type.
        self.next_element_id = 0  # the id of the next element to be generated.
        self.case_type = dict()  # a dictionary of case_id to case_type
        self.case_data = dict()  # a dictionary of case_id to data, where data is a dictionary of data types to data values

    def set_simulator(self, simulator):
        self.simulator = simulator

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'rb') as handle:
            instance = pickle.load(handle)
        return instance

    def save(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @abstractmethod
    def resource_pool(self, task):
        """
        For a given task, which is an instance of Element, returns a list of resources that can be used to complete the task.
        """
        raise NotImplementedError
    
    def restart(self):
        """
        Resets the problem to its initial state.
        """
        self.can_plan = dict()
        self.can_replan = dict()
        self.next_case_id = 0
        self.next_case_arrival_time = dict()
        for ct in self.case_types:
            self.next_case_arrival_time[ct] = self.interarrival_time_sample(ct, is_first_arrival=True)
        self.next_element_id = 0
        self.case_type = dict()
        self.case_data = dict()

    @abstractmethod
    def next_case(self):
        """
        Returns the next case to be generated, which is a tuple of arrival time and the initial Element of the case.
        """
        raise NotImplementedError

    @abstractmethod
    def next_regular_planning_moment(self, time):
        """
        Returns the next regular planning moment after the given time.
        """
        raise NotImplementedError

    @abstractmethod
    def processing_time_sample(self, resource, task, simulation_time):
        """
        Returns a sample of the processing time for a resource/task combination, where task is an instance of Element.
        """
        raise NotImplementedError

    @abstractmethod
    def complete_element(self, element):
        """
        Completes the given element and returns a list of new elements that are generated as a result of the completion.
        """
        raise NotImplementedError
    
    @abstractmethod
    def start_task(self, element):
        """
        Reports when a task has started
        """
        raise NotImplementedError

    @abstractmethod
    def data_sample(self, element):
        """
        Returns a sample of data for a given element.
        """
        raise NotImplementedError

    def get_unique_element_id(self):
        unique_id = self.next_element_id
        self.next_element_id += 1

        return unique_id
    
    def assign_resources(self, unassigned_tasks, available_resources):
        """
        Assigns tasks to resources
        """
        raise NotImplementedError

    def plan(self, case_id, element_label, time):
        if case_id not in self.simulator.case_start_times:
            raise ValueError("An event is planned for case " + str(case_id) + " that has not yet started. This typically happens when an event is planned as part of the next_case method. Avoid that.")

        new_element = Element(case_id, self.case_type[case_id], self.get_unique_element_id(), element_label, ElementType.EVENT, occurrence_time=time)
        self.add_data(new_element, self.data_sample(new_element))
        self.remove_can_plan(case_id, element_label)
        self.add_can_replan(case_id, element_label, new_element)
        return new_element

    def add_can_plan(self, case_id, element_label):
        if case_id not in self.can_plan:
            self.can_plan[case_id] = []
        self.can_plan[case_id].append(element_label)
    
    def remove_can_plan(self, case_id, element_label):
        self.can_plan[case_id].remove(element_label)
        if len(self.can_plan[case_id]) == 0:
            del self.can_plan[case_id]

    def replan(self, case_id, element_label, time):
        if case_id not in self.can_replan:
            raise ValueError("The case " + str(case_id) + " cannot be replanned.")
        if element_label not in self.can_replan[case_id]:
            raise ValueError("The element " + element_label + " of case " + str(case_id) + " cannot be replanned.")
        element_to_replan = self.can_replan[case_id][element_label]
        return element_to_replan

    def add_can_replan(self, case_id, element_label, element):
        if case_id not in self.can_replan:
            self.can_replan[case_id] = dict()
        self.can_replan[case_id][element_label] = element
    
    def remove_can_replan(self, case_id, element_label):
        del self.can_replan[case_id][element_label]
        if len(self.can_replan[case_id]) == 0:
            del self.can_replan[case_id]

    def end_case(self, case_id):
        self.simulator.busy_cases[case_id] = []


    @abstractmethod
    def interarrival_time_sample(self, case_type, is_first_arrival=False):
        """
        Returns a sample of the interarrival time for a given case type.
        """
        raise NotImplementedError

    def add_data(self, element, data):
        """
        Adds the given data to the given element and updates the case data as well.
        This method should be used to add data to the element. Data should not be added directly to the element, otherwise the case data will not be updated.
        """
        element.data = data
        if element.case_id not in self.case_data:
            self.case_data[element.case_id] = data
        else:
            self.case_data[element.case_id].update(data)

    def get_case_type(self, case_id):
        """
        Returns the type of the case with the given id.
        :param case_id: the id of the case.
        :return: the type of the case.
        """
        return self.case_type[case_id]
    
    def get_case_data(self, case_id):
        """
        Returns the data of the case with the given id.
        :param case_id: the id of the case.
        :return: the data of the case, which is a dictionary of data types to data values.
        """
        return self.case_data[case_id]

    def next_case_type(self):
        next_arrival = None  # type of next arriving case
        for case_type in self.case_types:
            if next_arrival is None or self.next_case_arrival_time[case_type] < self.next_case_arrival_time[next_arrival]:
                next_arrival = case_type
                arrival_time = self.next_case_arrival_time[case_type]
        case_id = self.next_case_id
        self.case_type[case_id] = next_arrival

        self.next_case_arrival_time[next_arrival] += self.interarrival_time_sample(next_arrival)
        self.next_case_id += 1

        return next_arrival, arrival_time, case_id
    
    @abstractmethod
    def evaluate(self):
        """
        Evaluate solution method
        :return: float score
        """
        raise NotImplementedError

class HealthcareElements(StrEnum):
    PATIENT_REFERAL = "patient_referal"
    EMERGENCY_PATIENT = "emergency_patient"
    TIME_FOR_INTAKE = "time_for_intake"
    INTAKE = "intake"
    PATIENT_LEFT_DUE_TO_LONG_WAIT = "patient_left_due_to_long_wait"
    SURGERY = "surgery"
    NURSING = "nursing"
    ER_TREATMENT = "ER_treatment"
    ER_TO_SURGERY = "ER_to_surgery"
    RELEASING = "releasing"


class HealthcareProblem(Problem):
    
    def __init__(self):
        super().__init__()
        self.case_types = ["A", "B", "EM"]
        self.__create_resources()
        self.planning_slot_usage = dict()  # (time, resource_type) -> list of planned element ids; time is in hours from Monday 2018-01-01 00:00 multiplied by 10 to avoid floating point errors
        self.planned_in_slot = dict()  # (case_id, element_label) -> (time, resource_type)
        self.patients_after_intake = [] # list of patients  having completed intake but no surgery / nursing has yet started
        self.er_treatment_finished = dict()
        self.er_surgery_nursing_started = dict()
        self.interarrival = arrival_times.InterArrivalTimes()
        self.restart()

    def __create_resources(self):
        self.__ORs = [Resource(ResourceType.OR, "OR" + str(i)) for i in range(1, 6)]
        self.__A_BEDs = [Resource(ResourceType.A_BED, "A_BED" + str(i)) for i in range(1, 31)]
        self.__B_BEDs = [Resource(ResourceType.B_BED, "B_BED" + str(i)) for i in range(1, 41)]
        self.__INTAKEs = [Resource(ResourceType.INTAKE, "INTAKE" + str(i)) for i in range(1, 5)]
        self.__ER_PRACTITIONERs = [Resource(ResourceType.ER_PRACTITIONER, "ER_PRACTITIONER" + str(i)) for i in range(1,10)]
        self.resources = self.__ORs + self.__A_BEDs + self.__B_BEDs + self.__INTAKEs + self.__ER_PRACTITIONERs

    def restart(self):
        super().restart()
        self.planning_slot_usage = dict()
        self.planned_in_slot = dict()

    def resource_pool(self, element):
        if element.label == HealthcareElements.SURGERY:
            return self.__ORs
        elif element.label == HealthcareElements.NURSING and self.get_case_data(element.case_id)['diagnosis'].startswith("A"):
            return self.__A_BEDs
        elif element.label == HealthcareElements.NURSING and self.get_case_data(element.case_id)['diagnosis'].startswith("B"):
            return self.__B_BEDs
        elif element.label == HealthcareElements.INTAKE:
            return self.__INTAKEs
        elif element.label == HealthcareElements.ER_TREATMENT:
            return self.__ER_PRACTITIONERs
        else:
            raise ValueError("Unknown task label", element.label)
    
    def is_working_time(self, simulator_time):
        week_day = simulator_time // 24 % 7
        time_of_day = simulator_time % 24
        working_day = week_day < 5 #monday-friday
        working_time = time_of_day >= 8 and time_of_day <= 17
        if working_day and working_time:
            return True
        else:
            return False

    def next_regular_planning_moment(self, previous_planning_moment):
        if previous_planning_moment == 0:
            return 18  # plan each day at 18:00
        return previous_planning_moment + 24

    def resources_idle_now(self, resource_type):
        """
        Returns if any resource of resource_type is idle at this moment in simulation time
        """
        # Return false if resources are not available        
        for resource in self.simulator.available_resources:
            if resource.type == resource_type:
                return True
        return False


    def assign_resources(self, unassigned_tasks, available_resources):
        """
        assign emergency tasks first, i.e., prioritize them
        """
        assignments = []
        resources_to_use = set(available_resources)
        emergency_tasks = filter(lambda k : k.case_type == 'EM', unassigned_tasks.values())
        for emergency_task in emergency_tasks:
            valid_resources = list(resources_to_use & set(self.resource_pool(emergency_task)))
            if valid_resources:
                assignments.append((emergency_task, valid_resources[0]))
                resources_to_use.remove(valid_resources[0])
        
        non_emergency_tasks = filter(lambda k : k.case_type != 'EM', unassigned_tasks.values())
        for non_emergency_task in non_emergency_tasks:
            valid_resources = list(resources_to_use & set(self.resource_pool(non_emergency_task)))
            if valid_resources:
                assignments.append((non_emergency_task, valid_resources[0]))
                resources_to_use.remove(valid_resources[0])
        return assignments

    def plan(self, case_id, element_label, time):
        if time < self.simulator.now:
            raise ValueError("The planned time of the element is in the past. The current simulation time is " + str(self.simulator.now) + " and the planned time is " + str(time) + ".")
        if element_label is not HealthcareElements.TIME_FOR_INTAKE:
            raise ValueError("The element " + element_label + " cannot be planned.")
        if case_id not in self.can_plan and case_id not in self.can_replan:
            raise ValueError("The element " + element_label + " cannot be (re)planned for case " + str(case_id) + ".")
        if case_id in self.can_plan and case_id in self.can_replan:
            raise ValueError("The element " + element_label + " can both be planned and replanned for case " + str(case_id) + ". This should not happen.")
        if element_label == HealthcareElements.TIME_FOR_INTAKE and time < self.simulator.now + 24:
            raise ValueError("The time for intake must be at least one day (24 hours) after the current time.")
        if case_id in self.can_plan:
            e = super().plan(case_id, element_label, time)
            return e
        elif case_id in self.can_replan:
            e = super().replan(case_id, element_label, time)
            return e
    
    def check_resource_schedule(self, resource_type, time, nr):
        if resource_type not in [ResourceType.OR, ResourceType.A_BED, ResourceType.B_BED, ResourceType.INTAKE, ResourceType.ER_PRACTITIONER]:
            raise ValueError("You are trying to schedule an unknown resource type", resource_type)
        if time < self.simulator.now+14:
            raise ValueError("Resources must be scheduled for tomorrow or later. However, you are trying to schedule resources of type " + resource_type + " for time " + str(time) + " which is less than 14 hours from now ("+str(self.simulator.now)+").")
        if nr > 5 and resource_type == ResourceType.OR:
            raise ValueError("You are trying to schedule " + str(nr) + " ORs, but there are only 5 ORs available.")
        if nr > 30 and resource_type == ResourceType.A_BED:
            raise ValueError("You are trying to schedule " + str(nr) + " A beds, but there are only 30 A beds available.")
        if nr > 40 and resource_type == ResourceType.B_BED:
            raise ValueError("You are trying to schedule " + str(nr) + " B beds, but there are only 40 B beds available.")
        if nr > 4 and resource_type == ResourceType.INTAKE:
            raise ValueError("You are trying to schedule " + str(nr) + " intake resources, but there are only 4 intake resources available.")
        if nr > 9 and resource_type == ResourceType.ER_PRACTITIONER:
            raise ValueError("You are trying to schedule " + str(nr) + " ER practitioners, but there are only 9 ER practitioners available.")
        if time < self.simulator.now + 158:  # when scheduling resources less than a week ahead, we can only increase their number
            if nr < self.simulator.schedule.get_number_of_resources(resource_type, time):
                raise ValueError("You are trying to schedule " + str(nr) + " " + resource_type + "s in less than a week from today. However, there are already " + str(self.simulator.schedule.get_number_of_resources(resource_type, time)) + " scheduled for that time. When scheduling less than a week ahead, you can only increase the number of resources scheduled, not decrease.")

    def data_sample(self, element):
        if element.label == HealthcareElements.PATIENT_REFERAL or element.label == HealthcareElements.EMERGENCY_PATIENT:
            if element.case_type == "A":
                return {"diagnosis": random.choices(["A1", "A2", "A3", "A4"], weights=[50, 25, 12.5, 12.5], k=1)[0]}
            elif element.case_type == "B":
                return {"diagnosis": random.choices(["B1", "B2", "B3", "B4"], weights=[50, 25, 12.5, 12.5], k=1)[0]}
            elif element.case_type == "EM":
                if random.random() > 0.5:
                    return {"diagnosis": random.choices(["B1", "B2", "B3", "B4"], weights=[50, 25, 12.5, 12.5], k=1)[0]}
                else:
                    return {"diagnosis" : None}
        return dict()

    def interarrival_time_sample(self, case_type, is_first_arrival=False):
        current_time = self.next_case_arrival_time[case_type] if not is_first_arrival else 0
        return self.interarrival.sample(current_time, case_type)

    def processing_time_sample(self, resource, task, simulation_time):
        if task.label == HealthcareElements.INTAKE:
            return max(0, random.normalvariate(1, 1/8))
        elif task.label == HealthcareElements.ER_TREATMENT:
            return max(0, random.normalvariate(2, 1/2))
        elif task.label == HealthcareElements.SURGERY:
            diagnosis = self.get_case_data(task.case_id)["diagnosis"]
            if diagnosis == "A2":
                return max(0, random.normalvariate(1, 1/4))
            elif diagnosis == "A3":
                return max(0, random.normalvariate(2, 1/2))
            elif diagnosis == "A4":
                return max(0, random.normalvariate(4, 1/2))
            elif diagnosis == "B3":
                return max(0, random.normalvariate(4, 1/2))
            elif diagnosis == "B4":
                return max(0, random.normalvariate(4, 1))
        elif task.label == HealthcareElements.NURSING:
            diagnosis = self.get_case_data(task.case_id)["diagnosis"]
            if diagnosis == "A1":
                duration = random.normalvariate(4, 1/2)
            elif diagnosis == "A2":
                duration = random.normalvariate(8, 2)
            elif diagnosis == "A3":
                duration = random.normalvariate(16, 2)
            elif diagnosis == "A4":
                duration = random.normalvariate(16, 2)
            elif diagnosis == "B1":
                duration = random.normalvariate(8, 2)
            elif diagnosis == "B2":
                duration = random.normalvariate(16, 2)
            elif diagnosis == "B3":
                duration = random.normalvariate(16, 4)
            elif diagnosis == "B4":
                duration = random.normalvariate(16, 4)
            nursing_finish_time = simulation_time + max(0, duration)
            release_time = self.next_release_time(nursing_finish_time)
            duration_until_release = release_time - simulation_time
            return duration_until_release
        else:
            raise ValueError("Unknown task label", task.element_label)
        
    def complication(self, task):
        diagnosis = self.get_case_data(task.case_id)["diagnosis"]
        r = random.random()
        if diagnosis in ["A1", "A2", "B2"]:
            return r < 0.01
        elif diagnosis in ["A3", "A4", "B3", "B4"]:
            return r < 0.02
        elif diagnosis in ["B1"]:
            return r < 0.001
        elif diagnosis == "EM":
            return False
        else:
            raise ValueError("Unknown Diagnosis", diagnosis)
        
    def next_release_time(self, current_time):
        release_times = [8, 13, 18]
        if current_time % 72 <= max(release_times):
            # release today
            valid_release_offsets = filter(lambda k : k >= 0,
                                   [release_time - (current_time % 72) for release_time in release_times]
            )
            return current_time + min(valid_release_offsets)
        else:
            # release next day
            return current_time + (72 - current_time % 72) + min(release_times)
        
    def next_case(self):
        case_type, arrival_time, case_id = self.next_case_type()

        if case_type == "EM":
            next_label = HealthcareElements.EMERGENCY_PATIENT
            next_element_type = ElementType.EVENT
        elif case_type == "A" or case_type == "B":
            next_label = HealthcareElements.PATIENT_REFERAL
            next_element_type = ElementType.EVENT
        
        initial_element = Element(case_id, case_type, self.get_unique_element_id(), next_label, next_element_type, occurrence_time=arrival_time)
        self.add_data(initial_element, self.data_sample(initial_element))

        return arrival_time, initial_element
    
    def complete_element(self, element):
        simulator_time = self.simulator.now
        next_label = None
        next_element_occurrence_time = None
        if element.label == HealthcareElements.PATIENT_REFERAL:
            # Afer referral, (1) the TIME_FOR_INTAKE can be planned
            # and (2) the patient can leave due to long wait
            self.add_can_plan(element.case_id, HealthcareElements.TIME_FOR_INTAKE)
            # We include the patient leaving due to long wait as an event, even though it never happens in the simulation
            # However, we need to keep it, because we always need a next task or event; otherwise the case completes
            # TODO: this is a workaround; it would be better to have a way to indicate that a case is completed
            next_label = HealthcareElements.PATIENT_LEFT_DUE_TO_LONG_WAIT
            next_element_type = ElementType.EVENT
            next_element_occurrence_time = simulator_time + self.simulator.running_time + 1

        elif element.label == HealthcareElements.TIME_FOR_INTAKE:
            self.remove_can_replan(element.case_id, HealthcareElements.TIME_FOR_INTAKE)
            next_label = HealthcareElements.INTAKE
            next_element_type = ElementType.TASK
            self.simulator.cancel(element.case_id, HealthcareElements.PATIENT_LEFT_DUE_TO_LONG_WAIT)

        elif element.label == HealthcareElements.PATIENT_LEFT_DUE_TO_LONG_WAIT:
            raise ValueError("This event should never happen in the simulation. It is only used to keep the case alive.")

        elif element.label == HealthcareElements.INTAKE:
            # After intake, the surgery / nursing happens
            self.simulator.cancel(element.case_id, HealthcareElements.PATIENT_LEFT_DUE_TO_LONG_WAIT)
            self.simulator.cancel(element.case_id, HealthcareElements.TIME_FOR_INTAKE)

            self.patients_after_intake.append(element.case_id)
            diagnosis = self.get_case_data(element.case_id)["diagnosis"]
            if diagnosis in ["A2", "A3", "A4", "B3", "B4"]:
                next_label = HealthcareElements.SURGERY
                next_element_type = ElementType.TASK
            elif diagnosis in ["A1", "B1", "B2"]:
                next_label = HealthcareElements.NURSING
                next_element_type = ElementType.TASK
            else:
                raise ValueError("Unknown diagnosis", diagnosis)
            
        elif element.label == HealthcareElements.SURGERY:
            # afer surgery the nursing happens
            next_label = HealthcareElements.NURSING
            next_element_type = ElementType.TASK


        elif element.label == HealthcareElements.EMERGENCY_PATIENT:
            # after an ER patient has arrived he will receive ER treatment
            next_label = HealthcareElements.ER_TREATMENT
            next_element_type = ElementType.TASK

        elif element.label == HealthcareElements.ER_TREATMENT:
            # after the patient has received ER treatment he will either be
            # sent home, or will further be processed
            diagnosis = self.get_case_data(element.case_id)["diagnosis"]
            if diagnosis == None:
                # release patient right away
                next_label = HealthcareElements.RELEASING
                next_element_type = ElementType.EVENT
                next_element_occurrence_time = simulator_time
            else:
                self.er_treatment_finished[element.case_id] = simulator_time
                if diagnosis in ["A2", "A3", "A4", "B3", "B4"]:
                    next_label = HealthcareElements.SURGERY
                    next_element_type = ElementType.TASK
                elif diagnosis in ["A1", "B1", "B2"]:
                    next_label = HealthcareElements.NURSING
                    next_element_type = ElementType.TASK

        elif element.label == HealthcareElements.NURSING:
            # after nursing has been completed the patient will
            # either be released, or get another treatment when
            # a complication has arised
            if self.complication(element):
                diagnosis = self.get_case_data(element.case_id)["diagnosis"]
                # If diagnosis is A1, B1, B2 patients do not need new surgery
                if diagnosis in ["A1", "B1", "B2"]:
                    next_label = HealthcareElements.NURSING
                else:
                    next_label = HealthcareElements.SURGERY
                next_element_type = ElementType.TASK
            else:
                # release patient at batch release time
                next_label = HealthcareElements.RELEASING
                next_element_type = ElementType.EVENT
                next_element_occurrence_time = self.next_release_time(simulator_time)

        elif element.label == HealthcareElements.RELEASING:
            next_label = None

        if next_label is not None:
            new_element = Element(element.case_id, element.case_type, self.get_unique_element_id(), next_label, next_element_type, occurrence_time=next_element_occurrence_time)
            #self.add_data(new_element, self.get_case_data(element.case_id))
            #self.add_data(new_element,  self.data_sample(new_element))
            return [new_element]
        
        return []

    def start_task(self, element):
        # keeps track of the patients after intake, and
        # the times when surgery/nurisng of er patients has started (used for evaluation)
        if element.label == HealthcareElements.SURGERY or \
            element.label == HealthcareElements.NURSING:
            if element.case_id in self.patients_after_intake:
                self.patients_after_intake.remove(element.case_id)
            if element.case_type == 'EM' and element.case_id not in self.er_surgery_nursing_started:
                self.er_surgery_nursing_started[element.case_id] = self.simulator.now

    def evaluate(self):
        """
        Evaluates the performance of the planning algorithm
        """
        er_treatment_duration_factor = 20
        sent_home_factor = 500
        processed_factor = 5000

        unfinished_cases = 0
        for busy_case in self.simulator.busy_cases:  
            if busy_case in self.simulator.case_start_times:
                start_time = self.simulator.case_start_times[busy_case]
                if start_time <= self.simulator.now:
                    unfinished_cases += 1
        
        cases_started = self.simulator.finalized_cases + unfinished_cases
        if cases_started == 0:  # This is just for debugging, should never happen under normal circumstances
            return

        # Penality for patients that wait longer than 5 hours after ER treatment
        # until they get processed with Nursing/Surgery
        er_treatment_2_processing = [
            (self.er_surgery_nursing_started[case_id] if case_id in self.er_surgery_nursing_started else self.simulator.now)
               - er_treatment_finished
               for case_id, er_treatment_finished in self.er_treatment_finished.items()]
        er_treatment_2_processing_excessive = [0 if t < 4 else (t-4)**2 for t in er_treatment_2_processing]

        er_treatment_score = sum(er_treatment_2_processing_excessive) / cases_started * er_treatment_duration_factor

        # patients sent home
        time_for_intake = dict(filter(lambda k : k[0].label == HealthcareElements.TIME_FOR_INTAKE,
                                self.simulator.event_times.items()))
        time_for_intake_cases = [t[0].case_id for t in time_for_intake.items()]
        intake_count = collections.Counter(time_for_intake_cases)
        sent_home_count = sum(filter(lambda k : k > 1, intake_count.values()))
        sent_home_score = sent_home_count / cases_started * sent_home_factor

        # patients processed
        released = len(list(filter(lambda k : k[0].label == HealthcareElements.RELEASING,
                                self.simulator.event_times.items())))
        processed_score = (cases_started - released) * processed_factor / cases_started


        # Measure_1: Throughput-Time: case_start -> HealthcareElements.INTAKE 
        tpt_dict = {}
        i = 0 
        for key, value in self.simulator.task_start_end_times.items(): 
            if key.label == HealthcareElements.INTAKE:
                case_start = self.simulator.case_start_times[key.case_id]
                tpt_dict[key] = (value[0]- case_start) 

        for key, value in self.simulator.busy_cases.items():
            if value[-1].label == HealthcareElements.INTAKE:
                case_start = self.simulator.case_start_times[key]
                tpt_dict[key] = (self.simulator.running_time - case_start)
                                        

        # Measure_2: Waiting-Time: start_time of task - end_time of following task
        wait_dict = {}           
        for key,value in self.simulator.task_start_end_times.items():
            if key.case_type != "EM":
                wait_dict.setdefault(key.case_id, {})
                wait_dict[key.case_id][key.label] = value

        for key, value in wait_dict.items():
            starts = [item[0] for key, item in wait_dict[key].items()]
            ends = [item[1] for key, item in wait_dict[key].items()]
            ends_starts_zip = [list(x) for x in zip(starts, ends)]
            intervals =  np.roll(np.array(ends_starts_zip), 1)
            intervals[0][0] = intervals[0][1] # First task "INTAKE" cannot have a waiting time
            intervals = np.diff(intervals)
            wait_dict[key]["waiting_time"] = sum(intervals)
        
        #busy_cases and wait_dict are mutually exclusive
        #add wait time of busy case task activation time till now
        for key, value in self.simulator.busy_cases.items():
            activation_time = value[-1].activation_time
            wait_dict[key] = dict()
            wait_dict[key]['waiting_time'] = self.simulator.running_time - activation_time 

        # Nervousness
        replans_per_trace = {} # array with time_of_replan,newly_planned_event,time_of_last_planning
        planned_events = self.simulator.planned_events
        for planned_event in planned_events:
          #print(f"{planned_event[0]} - {planned_event[1]} - {planned_event[1].occurrence_time}")
          case_id = planned_event[1].case_id
          if not(planned_event[1].case_id in replans_per_trace):
            replans_per_trace[case_id] = []
          planned_event_with_previous_time = planned_event.copy()
          to_add = None if(len(replans_per_trace[case_id])==0) else replans_per_trace[case_id][-1][1].occurrence_time
          planned_event_with_previous_time.append(to_add)
          replans_per_trace[case_id].append(planned_event_with_previous_time)
        
        nervousness_score = 0.0
        for trace,replans in replans_per_trace.items():
          #print(f"---------------------------- {trace}")
          for x in replans:
            pass
            #if(x[2]!= None and x[0]<x[2]):
            #if(x[2]!= None):
              #print(f"{x[0]} - {x[1].occurrence_time} ({x[1].label}) - {x[2]}")
          if len(replans) > 1:
            # evlt todo: auf 0/14 Tage normalisieren
            nervousness_score += reduce(lambda nervousness_increase,replan: nervousness_increase + (0 if(replan[2] is None) else (max(168*2.0-(replan[2]-replan[0]),0))), replans, 0.0)
        for key, value in self.simulator.busy_cases.items():
          if value[-1].label in [HealthcareElements.TIME_FOR_INTAKE, HealthcareElements.PATIENT_LEFT_DUE_TO_LONG_WAIT]:
            nervousness_score += 168*2.0*3
        #print(replans_per_trace)

        # Personnel Cost
        personnel_costs = self.simulator.schedule.get_total_cost()

        return {
            'waiting_time_for_admission' : sum(tpt_dict.values()),
            "waiting_time_in_hospital" : float(sum([element["waiting_time"] for element in wait_dict.values()])),
            'nervousness' : nervousness_score,
            'personnel_cost' : personnel_costs
        }
