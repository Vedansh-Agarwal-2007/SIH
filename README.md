# PraVaahan AI Scheduler

A production-grade train scheduling optimization service using Google OR-Tools CP-SAT solver and FastAPI.

## Features

- **Intelligent Scheduling**: Uses CP-SAT optimization to minimize total penalty based on train delays and priorities
- **RESTful API**: Clean FastAPI-based REST API with comprehensive documentation
- **Production Ready**: Includes logging, error handling, validation, and testing
- **Scalable**: Designed to handle multiple trains with complex scheduling constraints
- **Well Tested**: Comprehensive unit test suite with pytest

## Quick Start

### Installation

#### Quick Setup (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd pravaahan-scheduler

# Run the installation script
chmod +x install.sh
./install.sh
```

#### Manual Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd pravaahan-scheduler
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
# For development (includes testing and dev tools)
pip install -r requirements.txt

# For production only (minimal dependencies)
pip install -r requirements-minimal.txt
```

#### Requirements
- Python 3.8 or higher
- pip (Python package installer)

### Running the Service

#### Development Mode
```bash
python main.py
```

#### Production Mode
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000`

### API Documentation

- Interactive API docs: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc`

## API Usage

### Schedule Optimization

**Endpoint**: `POST /solve`

**Request Body**:
```json
{
  "trains": [
    {
      "name": "Train-001",
      "arrival_time": 0,
      "duration": 30,
      "priority": 5
    },
    {
      "name": "Train-002", 
      "arrival_time": 15,
      "duration": 45,
      "priority": 3
    }
  ]
}
```

**Response**:
```json
{
  "status": "Optimal",
  "total_penalty": 25,
  "schedule": [
    {
      "name": "Train-001",
      "start_time": 0,
      "end_time": 30,
      "delay": 0
    },
    {
      "name": "Train-002",
      "start_time": 30,
      "end_time": 75,
      "delay": 15
    }
  ]
}
```

### Health Check

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "service": "PraVaahan AI Scheduler"
}
```

## Testing

Run the test suite:
```bash
pytest test_solver.py -v
```

Run tests with coverage:
```bash
pytest test_solver.py --cov=solver --cov-report=html
```

## Architecture

### Core Components

1. **TrainScheduler Class** (`solver.py`): Core optimization logic using CP-SAT
2. **FastAPI Application** (`main.py`): REST API with validation and error handling
3. **Test Suite** (`test_solver.py`): Comprehensive unit tests

### Key Features

- **Input Validation**: Pydantic models ensure data integrity
- **Error Handling**: Graceful error handling with appropriate HTTP status codes
- **Logging**: Structured logging throughout the application
- **Type Hints**: Full type annotation for better code maintainability
- **Documentation**: Comprehensive docstrings and API documentation

## Configuration

### Environment Variables

- `LOG_LEVEL`: Set logging level (default: INFO)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)

### Logging

Logs are written to both console and `pravaahan_scheduler.log` file. Log levels:
- `DEBUG`: Detailed debugging information
- `INFO`: General information about operations
- `WARNING`: Warning messages for non-critical issues
- `ERROR`: Error messages for failures

## Production Deployment

### Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment

The service is designed to be stateless and can be easily deployed on Kubernetes with horizontal pod autoscaling.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

[Add your license information here]

## Troubleshooting

### Common Issues

#### OR-Tools Installation Issues
If you encounter issues installing OR-Tools:
```bash
# Try installing with specific version
pip install ortools==9.14.6206

# Or install from conda-forge
conda install -c conda-forge ortools
```

#### Python Version Issues
Ensure you're using Python 3.8 or higher:
```bash
python3 --version
# Should show Python 3.8.x or higher
```

#### Virtual Environment Issues
If you have issues with the virtual environment:
```bash
# Remove and recreate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Permission Issues on macOS/Linux
If you get permission errors:
```bash
chmod +x install.sh
./install.sh
```

### Getting Help

For issues and questions, please create an issue in the repository or contact the development team.

## Support

For issues and questions, please create an issue in the repository or contact the development team.
