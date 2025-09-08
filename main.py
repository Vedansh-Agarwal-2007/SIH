"""
PraVaahan AI Scheduler - FastAPI Application

This module provides a REST API for the train scheduling optimization service
using FastAPI and the TrainScheduler class.
"""

import logging
import sys
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

from solver import TrainScheduler, solve_schedule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pravaahan_scheduler.log')
    ]
)

logger = logging.getLogger(__name__)


class Train(BaseModel):
    """Pydantic model for train data validation."""
    name: str = Field(..., min_length=1, max_length=100, description="Unique train identifier")
    arrival_time: int = Field(..., ge=0, description="Arrival time (non-negative integer)")
    duration: int = Field(..., gt=0, description="Service duration (positive integer)")
    priority: int = Field(..., ge=1, le=10, description="Priority level (1-10, higher is more important)")

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "name": "Train-001",
                "arrival_time": 0,
                "duration": 30,
                "priority": 5
            }
        }


class ScheduleRequest(BaseModel):
    """Request model for schedule optimization."""
    trains: List[Train] = Field(..., min_items=1, max_items=100, description="List of trains to schedule")

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "trains": [
                    {"name": "Train-001", "arrival_time": 0, "duration": 30, "priority": 5},
                    {"name": "Train-002", "arrival_time": 15, "duration": 45, "priority": 3}
                ]
            }
        }


class ScheduleResponse(BaseModel):
    """Response model for schedule optimization."""
    status: str = Field(..., description="Solution status: Optimal, Infeasible, or Error")
    total_penalty: Optional[int] = Field(None, description="Total penalty value (if optimal)")
    schedule: List[Dict[str, Any]] = Field(default_factory=list, description="Optimized schedule")
    error_message: Optional[str] = Field(None, description="Error details (if error)")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    service: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting PraVaahan AI Scheduler service")
    yield
    # Shutdown
    logger.info("Shutting down PraVaahan AI Scheduler service")


# Create FastAPI application
app = FastAPI(
    title="PraVaahan AI Scheduler",
    description="An intelligent train scheduling optimization API using CP-SAT",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors."""
    logger.warning(f"Validation error in request: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation error",
            "errors": exc.errors(),
            "status": "Error"
        }
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions."""
    logger.error(f"Value error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": str(exc),
            "status": "Error"
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "status": "Error"
        }
    )


@app.get("/", response_model=HealthResponse)
async def read_root():
    """Root endpoint for health check."""
    logger.info("Health check requested")
    return HealthResponse(
        status="PraVaahan AI Scheduler is running",
        version="1.0.0",
        service="PraVaahan AI Scheduler"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    logger.debug("Health check endpoint accessed")
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        service="PraVaahan AI Scheduler"
    )


@app.post("/solve", response_model=ScheduleResponse)
async def create_schedule(request: ScheduleRequest):
    """
    Solve train scheduling optimization problem.

    Args:
        request (ScheduleRequest): Request containing train data

    Returns:
        ScheduleResponse: Optimized schedule or error information

    Raises:
        HTTPException: For various error conditions
    """
    logger.info(f"Received schedule request with {len(request.trains)} trains")

    try:
        # Convert Pydantic models to dictionaries
        trains_data = [train.model_dump() for train in request.trains]

        # Log train details for debugging
        for train in trains_data:
            logger.debug(f"Train: {train['name']}, arrival: {train['arrival_time']}, "
                        f"duration: {train['duration']}, priority: {train['priority']}")

        # Create scheduler and solve
        scheduler = TrainScheduler(trains_data)
        solution = scheduler.solve()

        # Log solution status
        logger.info(f"Schedule solution status: {solution['status']}")

        if solution['status'] == 'Error':
            logger.error(f"Solver error: {solution.get('error_message', 'Unknown error')}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=solution.get('error_message', 'Solver encountered an error')
            )

        if solution['status'] == 'Infeasible':
            logger.warning("No feasible solution found for the given constraints")
            return ScheduleResponse(
                status=solution['status'],
                total_penalty=None,
                schedule=[],
                error_message=solution.get('error_message', 'No valid schedule exists')
            )

        # Log successful solution
        if solution['status'] in ['Optimal', 'Feasible']:
            logger.info(f"Solution found with total penalty: {solution['total_penalty']}")
            for train_schedule in solution['schedule']:
                logger.debug(f"Scheduled {train_schedule['name']}: "
                           f"start={train_schedule['start_time']}, "
                           f"end={train_schedule['end_time']}, "
                           f"delay={train_schedule['delay']}")

        return ScheduleResponse(
            status=solution['status'],
            total_penalty=solution['total_penalty'],
            schedule=solution['schedule'],
            error_message=solution.get('error_message')
        )

    except ValueError as e:
        logger.error(f"Value error in schedule request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in schedule request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the request"
        )


@app.post("/solve-legacy", response_model=ScheduleResponse)
async def create_schedule_legacy(request: ScheduleRequest):
    """
    Legacy endpoint using the original solve_schedule function.

    Args:
        request (ScheduleRequest): Request containing train data

    Returns:
        ScheduleResponse: Optimized schedule or error information
    """
    logger.info(f"Received legacy schedule request with {len(request.trains)} trains")

    try:
        trains_data = [train.model_dump() for train in request.trains]
        solution = solve_schedule(trains_data)

        logger.info(f"Legacy solution status: {solution['status']}")

        return ScheduleResponse(
            status=solution['status'],
            total_penalty=solution.get('total_penalty'),
            schedule=solution.get('schedule', []),
            error_message=solution.get('error_message')
        )

    except Exception as e:
        logger.error(f"Error in legacy schedule request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the legacy request"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
