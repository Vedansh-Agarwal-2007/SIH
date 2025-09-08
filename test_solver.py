import pytest
from solver import TrainScheduler, SolverConfig, TrainData, ValidationError


# --- Test Suite for the Advanced TrainScheduler ---

def test_successful_initialization():
    """Tests that the scheduler initializes correctly with valid data."""
    trains = [TrainData(name="T1", arrival_time=0, duration=10, priority=5)]
    scheduler = TrainScheduler(trains_data=trains)
    assert len(scheduler.trains_data) == 1
    assert scheduler.config.track_length == 20  # Check default config


def test_initialization_with_custom_config():
    """Tests that a custom configuration is applied correctly."""
    trains = [TrainData(name="T1", arrival_time=0, duration=10, priority=5)]
    config = SolverConfig(track_length=50, headway_time=5)
    scheduler = TrainScheduler(trains_data=trains, config=config)
    assert scheduler.config.track_length == 50
    assert scheduler.config.headway_time == 5


def test_validation_error_on_empty_train_list():
    """Tests that an error is raised for empty train data."""
    with pytest.raises(ValidationError, match="trains_data cannot be empty"):
        TrainScheduler(trains_data=[])


def test_validation_error_on_bad_train_data():
    """Tests that an error is raised for invalid data within a train object."""
    with pytest.raises(ValidationError, match="Duration must be a positive number"):
        trains_data = [{"name": "BadTrain", "arrival_time": 0, "duration": 0, "priority": 1}]
        TrainScheduler(trains_data=trains_data)


def test_config_validation_error():
    """Tests that an error is raised for invalid config data."""
    with pytest.raises(ValueError, match="track_length must be positive"):
        SolverConfig(track_length=0)


def test_simple_schedule_solves_optimally():
    """Tests a simple, solvable two-train scenario."""
    trains = [
        TrainData(name="T1", arrival_time=0, duration=10, priority=1),
        TrainData(name="T2", arrival_time=5, duration=10, priority=5)
    ]
    scheduler = TrainScheduler(trains_data=trains)
    result = scheduler.solve()

    assert result['status'] in ['Optimal', 'Feasible']
    assert len(result['schedule']) == 2
    assert result['solver_metrics']['total_penalty'] is not None
    assert 'moving_block_metrics' in result


def test_moving_block_headway_constraint():
    """
    Specifically tests the 'Moving Block' headway logic. Two trains arriving at
    the same time must be scheduled with a gap at least equal to the headway time.
    """
    trains = [
        TrainData(name="T1", arrival_time=0, duration=10, priority=1),
        TrainData(name="T2", arrival_time=0, duration=10, priority=1)
    ]
    config = SolverConfig(headway_time=3)  # Set a specific headway of 3 minutes
    scheduler = TrainScheduler(trains_data=trains, config=config)
    result = scheduler.solve()

    assert result['status'] in ['Optimal', 'Feasible']

    schedule = result['schedule']
    t1_start = next(t['start_time'] for t in schedule if t['name'] == 'T1')
    t2_start = next(t['start_time'] for t in schedule if t['name'] == 'T2')

    # Verify that the start times are separated by at least the headway time
    assert abs(t1_start - t2_start) >= config.headway_time