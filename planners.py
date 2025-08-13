from abc import ABC, abstractmethod


class Planner(ABC):
    """
    The class that must be implemented to create a planner.
    The class must implement the plan method.
    The class must not use the simulator or the problem directly. Information from those classes is not available to it.
    Note that once an event is planned, it can still show up as possible event to (re)plan.
    """
        
    @abstractmethod
    def plan(self, plannable_elements, simulation_time):
        '''
        The method that must be implemented for planning.
        :param plannable_elements: A dictionary with case_id as key and a list of element_labels that can be planned or re-planned.
        :param simulation_time: The current simulation time.
        :return: A list of tuples of how the elements are planned. Each tuple must have the following format: (case_id, element_label, timestamp).
        '''
        
        pass


    def report(self, case_id, element, timestamp, resource, lifecycle_state):
        '''
        The method that can be implemented for reporting.
        It is called by the simulator upon each simulation event.
        '''
        pass