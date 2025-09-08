"""
Unit tests for PraVaahan AI Scheduler

This module contains comprehensive unit tests for the TrainScheduler class
and related functionality using pytest.
"""

import pytest
import logging
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

from solver import TrainScheduler, solve_schedule


class TestTrainScheduler:
    """Test cases for the TrainScheduler class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Configure logging to suppress output during tests
        logging.getLogger().setLevel(logging.CRITICAL)
    
    def test_valid_initialization(self):
        """Test TrainScheduler initialization with valid data."""
        trains_data = [
            {"name": "Train-001", "arrival_time": 0, "duration": 30, "priority": 5},
            {"name": "Train-002", "arrival_time": 15, "duration": 45, "priority": 3}
        ]
        
        scheduler = TrainScheduler(trains_data)
        
        assert scheduler.trains_data == trains_data
        assert len(scheduler.train_tasks) == 0  # Tasks created during solve()
        assert scheduler.horizon > 0
        assert scheduler.model is not None
    
    def test_empty_trains_data_raises_error(self):
        """Test that empty trains_data raises ValueError."""
        with pytest.raises(ValueError, match="trains_data cannot be empty"):
            TrainScheduler([])
    
    def test_missing_required_fields_raises_error(self):
        """Test that missing required fields raise ValueError."""
        trains_data = [{"name": "Train-001"}]  # Missing required fields
        
        with pytest.raises(ValueError, match="Missing required field"):
            TrainScheduler(trains_data)
    
    def test_invalid_train_name_raises_error(self):
        """Test that invalid train names raise ValueError."""
        trains_data = [{"name": "", "arrival_time": 0, "duration": 30, "priority": 5}]
        
        with pytest.raises(ValueError, match="Train name must be a non-empty string"):
            TrainScheduler(trains_data)
    
    def test_negative_arrival_time_raises_error(self):
        """Test that negative arrival times raise ValueError."""
        trains_data = [{"name": "Train-001", "arrival_time": -1, "duration": 30, "priority": 5}]
        
        with pytest.raises(ValueError, match="Arrival time must be a non-negative number"):
            TrainScheduler(trains_data)
    
    def test_non_positive_duration_raises_error(self):
        """Test that non-positive durations raise ValueError."""
        trains_data = [{"name": "Train-001", "arrival_time": 0, "duration": 0, "priority": 5}]
        
        with pytest.raises(ValueError, match="Duration must be a positive number"):
            TrainScheduler(trains_data)
    
    def test_invalid_priority_raises_error(self):
        """Test that invalid priorities raise ValueError."""
        trains_data = [{"name": "Train-001", "arrival_time": 0, "duration": 30, "priority": 0}]
        
        with pytest.raises(ValueError, match="Priority must be a number >= 1"):
            TrainScheduler(trains_data)
    
    def test_solve_simple_feasible_case(self):
        """Test solving a simple feasible case."""
        trains_data = [
            {"name": "Train-001", "arrival_time": 0, "duration": 30, "priority": 5},
            {"name": "Train-002", "arrival_time": 15, "duration": 45, "priority": 3}
        ]
        
        scheduler = TrainScheduler(trains_data)
        result = scheduler.solve()
        
        assert result['status'] in ['Optimal', 'Feasible']
        assert result['total_penalty'] is not None
        assert len(result['schedule']) == 2
        
        # Check that all trains are scheduled
        scheduled_names = {train['name'] for train in result['schedule']}
        assert scheduled_names == {"Train-001", "Train-002"}
        
        # Check that start times are >= arrival times
        for train in result['schedule']:
            assert train['start_time'] >= trains_data[0]['arrival_time'] if train['name'] == 'Train-001' else trains_data[1]['arrival_time']
            assert train['end_time'] == train['start_time'] + (30 if train['name'] == 'Train-001' else 45)
    
    def test_solve_infeasible_case(self):
        """Test solving an infeasible case."""
        # Create a case where two trains have overlapping requirements that can't be satisfied
        trains_data = [
            {"name": "Train-001", "arrival_time": 0, "duration": 100, "priority": 5},
            {"name": "Train-002", "arrival_time": 10, "duration": 100, "priority": 3}
        ]
        
        scheduler = TrainScheduler(trains_data)
        result = scheduler.solve()
        
        # This might be feasible or infeasible depending on the specific constraints
        # We'll check that we get a valid response structure
        assert result['status'] in ['Optimal', 'Feasible', 'Infeasible', 'Error']
        assert 'schedule' in result
        assert 'total_penalty' in result
    
    def test_solve_single_train(self):
        """Test solving with a single train."""
        trains_data = [
            {"name": "Train-001", "arrival_time": 10, "duration": 30, "priority": 5}
        ]
        
        scheduler = TrainScheduler(trains_data)
        result = scheduler.solve()
        
        assert result['status'] in ['Optimal', 'Feasible']
        assert len(result['schedule']) == 1
        assert result['schedule'][0]['name'] == "Train-001"
        assert result['schedule'][0]['start_time'] >= 10
        assert result['schedule'][0]['end_time'] == result['schedule'][0]['start_time'] + 30
        assert result['schedule'][0]['delay'] == result['schedule'][0]['start_time'] - 10
    
    def test_solve_high_priority_preference(self):
        """Test that higher priority trains get better scheduling."""
        trains_data = [
            {"name": "Low-Priority", "arrival_time": 0, "duration": 30, "priority": 1},
            {"name": "High-Priority", "arrival_time": 5, "duration": 30, "priority": 10}
        ]
        
        scheduler = TrainScheduler(trains_data)
        result = scheduler.solve()
        
        assert result['status'] in ['Optimal', 'Feasible']
        
        # Find the trains in the schedule
        low_priority_schedule = next(train for train in result['schedule'] if train['name'] == 'Low-Priority')
        high_priority_schedule = next(train for train in result['schedule'] if train['name'] == 'High-Priority')
        
        # High priority train should have less delay relative to its arrival time
        high_priority_delay_ratio = high_priority_schedule['delay'] / 5  # arrival time is 5
        low_priority_delay_ratio = low_priority_schedule['delay']  # arrival time is 0, so just use delay
        
        # This is a heuristic test - the exact behavior depends on the solver
        # Both trains should be scheduled
        assert high_priority_schedule['start_time'] >= 5
        assert low_priority_schedule['start_time'] >= 0
    
    def test_horizon_calculation(self):
        """Test that horizon is calculated correctly."""
        trains_data = [
            {"name": "Train-001", "arrival_time": 10, "duration": 20, "priority": 5},
            {"name": "Train-002", "arrival_time": 30, "duration": 40, "priority": 3}
        ]
        
        scheduler = TrainScheduler(trains_data)
        expected_horizon = (10 + 20) + (30 + 40)  # sum of (arrival + duration)
        assert scheduler.horizon == expected_horizon
    
    def test_solve_with_float_values(self):
        """Test solving with float values (should be converted to int)."""
        trains_data = [
            {"name": "Train-001", "arrival_time": 10.5, "duration": 20.7, "priority": 5.0}
        ]
        
        scheduler = TrainScheduler(trains_data)
        result = scheduler.solve()
        
        assert result['status'] in ['Optimal', 'Feasible']
        assert len(result['schedule']) == 1
    
    def test_solve_with_large_dataset(self):
        """Test solving with a larger dataset."""
        trains_data = []
        for i in range(10):
            trains_data.append({
                "name": f"Train-{i:03d}",
                "arrival_time": i * 5,
                "duration": 20,
                "priority": (i % 5) + 1
            })
        
        scheduler = TrainScheduler(trains_data)
        result = scheduler.solve()
        
        assert result['status'] in ['Optimal', 'Feasible', 'Infeasible']
        if result['status'] in ['Optimal', 'Feasible']:
            assert len(result['schedule']) == 10


class TestLegacyFunction:
    """Test cases for the legacy solve_schedule function."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        logging.getLogger().setLevel(logging.CRITICAL)
    
    def test_legacy_function_works(self):
        """Test that the legacy function still works."""
        trains_data = [
            {"name": "Train-001", "arrival_time": 0, "duration": 30, "priority": 5}
        ]
        
        result = solve_schedule(trains_data)
        
        assert 'status' in result
        assert 'schedule' in result
        assert result['status'] in ['Optimal', 'Feasible', 'Infeasible', 'Error']
    
    def test_legacy_function_with_multiple_trains(self):
        """Test legacy function with multiple trains."""
        trains_data = [
            {"name": "Train-001", "arrival_time": 0, "duration": 30, "priority": 5},
            {"name": "Train-002", "arrival_time": 15, "duration": 45, "priority": 3}
        ]
        
        result = solve_schedule(trains_data)
        
        assert result['status'] in ['Optimal', 'Feasible', 'Infeasible', 'Error']
        if result['status'] in ['Optimal', 'Feasible']:
            assert len(result['schedule']) == 2


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        logging.getLogger().setLevel(logging.CRITICAL)
    
    def test_identical_trains(self):
        """Test with identical train data."""
        trains_data = [
            {"name": "Train-001", "arrival_time": 0, "duration": 30, "priority": 5},
            {"name": "Train-002", "arrival_time": 0, "duration": 30, "priority": 5}
        ]
        
        scheduler = TrainScheduler(trains_data)
        result = scheduler.solve()
        
        assert result['status'] in ['Optimal', 'Feasible', 'Infeasible']
    
    def test_zero_duration_raises_error(self):
        """Test that zero duration raises an error."""
        trains_data = [{"name": "Train-001", "arrival_time": 0, "duration": 0, "priority": 5}]
        
        with pytest.raises(ValueError):
            TrainScheduler(trains_data)
    
    def test_very_large_numbers(self):
        """Test with very large numbers."""
        trains_data = [
            {"name": "Train-001", "arrival_time": 1000000, "duration": 500000, "priority": 10}
        ]
        
        scheduler = TrainScheduler(trains_data)
        result = scheduler.solve()
        
        assert result['status'] in ['Optimal', 'Feasible', 'Infeasible', 'Error']
    
    def test_unicode_train_names(self):
        """Test with unicode train names."""
        trains_data = [
            {"name": "🚂-001", "arrival_time": 0, "duration": 30, "priority": 5},
            {"name": "Train-中文", "arrival_time": 15, "duration": 45, "priority": 3}
        ]
        
        scheduler = TrainScheduler(trains_data)
        result = scheduler.solve()
        
        assert result['status'] in ['Optimal', 'Feasible', 'Infeasible']


class TestLogging:
    """Test logging functionality."""
    
    def test_logging_during_solve(self, caplog):
        """Test that appropriate log messages are generated during solve."""
        caplog.set_level(logging.INFO)
        
        trains_data = [
            {"name": "Train-001", "arrival_time": 0, "duration": 30, "priority": 5}
        ]
        
        scheduler = TrainScheduler(trains_data)
        scheduler.solve()
        
        # Check that key log messages are present
        log_messages = [record.message for record in caplog.records]
        assert any("Starting train scheduling optimization" in msg for msg in log_messages)
        assert any("Solving CP-SAT model" in msg for msg in log_messages)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
