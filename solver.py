"""
PraVaahan AI Scheduler - Core CP-SAT Solver Module

This module contains the TrainScheduler class that uses Google OR-Tools CP-SAT
to solve train scheduling optimization problems.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from ortools.sat.python import cp_model

# Configure logging
logger = logging.getLogger(__name__)


class TrainScheduler:
    """
    A CP-SAT based train scheduler that optimizes train scheduling to minimize
    total penalty based on delays and priorities.
    
    Attributes:
        trains_data (List[Dict]): List of train data dictionaries
        model (cp_model.CpModel): The CP-SAT model instance
        train_tasks (Dict): Dictionary mapping train names to their task variables
        horizon (int): The time horizon for scheduling
    """
    
    def __init__(self, trains_data: List[Dict[str, Any]]):
        """
        Initialize the TrainScheduler with train data.
        
        Args:
            trains_data (List[Dict]): List of dictionaries containing train information
                Each dictionary should have keys: 'name', 'arrival_time', 'duration', 'priority'
        
        Raises:
            ValueError: If trains_data is empty or contains invalid data
        """
        if not trains_data:
            raise ValueError("trains_data cannot be empty")
        
        # Validate all train data before proceeding
        for train in trains_data:
            self._validate_train_data(train)
        
        self.trains_data = trains_data
        self.model = cp_model.CpModel()
        self.train_tasks: Dict[str, Dict[str, Any]] = {}
        self.horizon = self._calculate_horizon()
        
        logger.info(f"Initialized TrainScheduler with {len(trains_data)} trains")
        logger.debug(f"Calculated horizon: {self.horizon}")
    
    def _calculate_horizon(self) -> int:
        """
        Calculate the time horizon for the scheduling problem.
        
        Returns:
            int: The calculated time horizon
        """
        total_duration = sum(train['duration'] for train in self.trains_data)
        total_arrival = sum(train['arrival_time'] for train in self.trains_data)
        return total_duration + total_arrival
    
    def _validate_train_data(self, train: Dict[str, Any]) -> None:
        """
        Validate individual train data.
        
        Args:
            train (Dict): Train data dictionary
            
        Raises:
            ValueError: If train data is invalid
        """
        required_fields = ['name', 'arrival_time', 'duration', 'priority']
        for field in required_fields:
            if field not in train:
                raise ValueError(f"Missing required field: {field}")
        
        if not isinstance(train['name'], str) or not train['name'].strip():
            raise ValueError("Train name must be a non-empty string")
        
        if not isinstance(train['arrival_time'], (int, float)) or train['arrival_time'] < 0:
            raise ValueError("Arrival time must be a non-negative number")
        
        if not isinstance(train['duration'], (int, float)) or train['duration'] <= 0:
            raise ValueError("Duration must be a positive number")
        
        if not isinstance(train['priority'], (int, float)) or train['priority'] < 1:
            raise ValueError("Priority must be a number >= 1")
        
        # Convert to integers for OR-Tools compatibility
        train['arrival_time'] = int(train['arrival_time'])
        train['duration'] = int(train['duration'])
        train['priority'] = int(train['priority'])
    
    def _create_variables(self) -> Tuple[List[Any], List[Any]]:
        """
        Create CP-SAT variables for all trains.
        
        Returns:
            Tuple[List[Any], List[Any]]: Tuple of (all_end_times, total_penalty_vars)
        """
        all_end_times = []
        total_penalty_vars = []
        
        for train in self.trains_data:
            train_name = train['name']
            arrival = train['arrival_time']
            duration = train['duration']
            priority = train['priority']
            
            # Create variables
            start_var = self.model.NewIntVar(
                arrival, self.horizon, f'start_{train_name}'
            )
            end_var = self.model.NewIntVar(
                0, self.horizon, f'end_{train_name}'
            )
            interval_var = self.model.NewIntervalVar(
                start_var, duration, end_var, f'interval_{train_name}'
            )
            
            # Store task information
            self.train_tasks[train_name] = {
                'start': start_var,
                'end': end_var,
                'interval': interval_var,
                'arrival': arrival,
                'duration': duration,
                'priority': priority
            }
            
            all_end_times.append(end_var)
            
            # Create delay and penalty variables
            delay_var = self.model.NewIntVar(0, self.horizon, f'delay_{train_name}')
            self.model.Add(delay_var == start_var - arrival)
            
            penalty_var = self.model.NewIntVar(
                0, self.horizon * priority, f'penalty_{train_name}'
            )
            self.model.AddMultiplicationEquality(penalty_var, [delay_var, priority])
            total_penalty_vars.append(penalty_var)
            
            logger.debug(f"Created variables for train {train_name}")
        
        return all_end_times, total_penalty_vars
    
    def _add_constraints(self, all_end_times: List[Any], total_penalty_vars: List[Any]) -> None:
        """
        Add constraints to the CP-SAT model.
        
        Args:
            all_end_times (List[Any]): List of end time variables
            total_penalty_vars (List[Any]): List of penalty variables
        """
        # No overlap constraint
        intervals = [task['interval'] for task in self.train_tasks.values()]
        self.model.AddNoOverlap(intervals)
        
        # Total penalty constraint
        total_penalty = self.model.NewIntVar(
            0, self.horizon * sum(t['priority'] for t in self.trains_data),
            'total_penalty'
        )
        self.model.Add(total_penalty == sum(total_penalty_vars))
        self.model.Minimize(total_penalty)
        
        logger.debug("Added constraints to the model")
    
    def _extract_solution(self, solver: cp_model.CpSolver) -> Dict[str, Any]:
        """
        Extract solution from the solver.
        
        Args:
            solver (cp_model.CpSolver): The solved CP-SAT solver
            
        Returns:
            Dict[str, Any]: Solution dictionary containing schedule and metadata
        """
        schedule = []
        total_penalty = 0
        
        for train in self.trains_data:
            train_name = train['name']
            start_time = solver.Value(self.train_tasks[train_name]['start'])
            end_time = solver.Value(self.train_tasks[train_name]['end'])
            delay = start_time - train['arrival_time']
            
            schedule.append({
                'name': train_name,
                'start_time': start_time,
                'end_time': end_time,
                'delay': delay
            })
            
            # Calculate penalty for this train
            penalty = delay * train['priority']
            total_penalty += penalty
            
            logger.debug(f"Train {train_name}: start={start_time}, end={end_time}, delay={delay}")
        
        return {
            'status': 'Optimal',
            'total_penalty': total_penalty,
            'schedule': schedule
        }
    
    def solve(self) -> Dict[str, Any]:
        """
        Solve the train scheduling optimization problem.
        
        Returns:
            Dict[str, Any]: Solution dictionary containing:
                - status: 'Optimal', 'Infeasible', or 'Error'
                - total_penalty: Total penalty value (if optimal)
                - schedule: List of scheduled trains (if optimal)
                - error_message: Error details (if error)
        
        Raises:
            Exception: If an unexpected error occurs during solving
        """
        try:
            logger.info("Starting train scheduling optimization")
            
            # Create variables
            all_end_times, total_penalty_vars = self._create_variables()
            
            # Add constraints
            self._add_constraints(all_end_times, total_penalty_vars)
            
            # Solve the model
            solver = cp_model.CpSolver()
            logger.info("Solving CP-SAT model...")
            
            status = solver.Solve(self.model)
            
            if status == cp_model.OPTIMAL:
                logger.info("Found optimal solution")
                return self._extract_solution(solver)
            elif status == cp_model.FEASIBLE:
                logger.warning("Found feasible but not optimal solution")
                return self._extract_solution(solver)
            elif status == cp_model.INFEASIBLE:
                logger.warning("Problem is infeasible - no valid schedule exists")
                return {
                    'status': 'Infeasible',
                    'total_penalty': None,
                    'schedule': [],
                    'error_message': 'No valid schedule exists for the given constraints'
                }
            else:
                error_msg = f"Unexpected solver status: {status}"
                logger.error(error_msg)
                return {
                    'status': 'Error',
                    'total_penalty': None,
                    'schedule': [],
                    'error_message': error_msg
                }
                
        except Exception as e:
            error_msg = f"Unexpected error during solving: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                'status': 'Error',
                'total_penalty': None,
                'schedule': [],
                'error_message': error_msg
            }


def solve_schedule(trains_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Legacy function for backward compatibility.
    
    Args:
        trains_data (List[Dict]): List of train data dictionaries
        
    Returns:
        Dict[str, Any]: Solution dictionary
    """
    logger.warning("Using legacy solve_schedule function. Consider using TrainScheduler class directly.")
    scheduler = TrainScheduler(trains_data)
    return scheduler.solve()
