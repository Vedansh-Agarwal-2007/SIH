import logging
import sys
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import the new, advanced classes from your solver
from solver import TrainScheduler, SolverConfig, TrainData, ValidationError, SolverStatus

# --- Basic Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- API Data Models (Pydantic) ---

# The request model for the /solve endpoint, allowing for optional config overrides
class ScheduleRequest(BaseModel):
    trains: List[TrainData]
    config: Optional[SolverConfig] = Field(None, description="Optional solver configuration to override defaults.")

# The response model, updated to include the rich metrics from the solver
class ScheduleResponse(BaseModel):
    status: str
    total_penalty: Optional[int] = None
    schedule: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None
    moving_block_metrics: Optional[Dict[str, Any]] = None
    solver_metrics: Optional[Dict[str, Any]] = None

# --- FastAPI Application ---

app = FastAPI(
    title="PraVaahan AI Scheduler (Advanced)",
    description="An intelligent train scheduling API using Moving Block principles.",
    version="3.0.0"
)

@app.post("/solve", response_model=ScheduleResponse)
async def create_schedule(request: ScheduleRequest):
    """
    Receives train data and optional configuration, then returns the optimal schedule.
    """
    try:
        logger.info(f"Received schedule request for {len(request.trains)} trains.")
        
        # Instantiate the scheduler with the provided data and config
        # The config object from the request will be used if provided
        scheduler = TrainScheduler(
            trains_data=request.trains,
            config=request.config
        )
        
        # Solve the scheduling problem
        solution = scheduler.solve()
        
        # Check the solver status and return the appropriate response
        if solution.get('status') in [SolverStatus.OPTIMAL.value, SolverStatus.FEASIBLE.value]:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content=solution
            )
        else:
            logger.warning(f"Solver finished with non-optimal status: {solution.get('status')}")
            # For Infeasible or Error, the solver already provides a detailed dictionary
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=solution
            )
            
    except (ValidationError, ValueError) as e:
        logger.error(f"Validation Error in request data or config: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"An unexpected internal error occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal server error occurred.")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "PraVaahan AI Scheduler v3.0.0"}
