"""
FastAPI service for job submission and status
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path for shared module
sys.path.insert(0, str(Path(__file__).parent))

from shared.celery_app import celery_app
from shared.config import OUTPUT_DIR

app = FastAPI(title="Genjutsu 3D Generation API")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text description of 3D object")
    model: str = Field(default="shap_e", description="Model to use")
    guidance_scale: float = Field(default=15.0, ge=1.0, le=30.0)
    num_inference_steps: int = Field(default=64, ge=16, le=256)


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # PENDING, STARTED, SUCCESS, FAILURE, RETRY
    progress: Optional[float] = None
    message: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None


@app.get("/")
async def root():
    return {
        "service": "Genjutsu 3D Generation API",
        "version": "2.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    # Check Redis connection
    try:
        celery_app.broker_connection().ensure_connection(max_retries=3)
        redis_status = "connected"
    except Exception as e:
        redis_status = f"disconnected: {str(e)}"

    # Check worker availability
    stats = celery_app.control.inspect().stats()
    active_workers = len(stats) if stats else 0

    return {
        "status": "healthy" if redis_status == "connected" else "degraded",
        "redis": redis_status,
        "workers": active_workers,
        "output_dir": str(OUTPUT_DIR)
    }


@app.get("/workers")
async def list_workers():
    """List active Celery workers"""
    stats = celery_app.control.inspect().stats()
    active = celery_app.control.inspect().active()

    return {
        "workers": stats or {},
        "active_tasks": active or {}
    }


@app.post("/generate", response_model=JobResponse)
async def generate(request: GenerateRequest):
    """
    Submit a 3D generation job
    
    Returns job_id for tracking progress
    """
    try:
        # Submit task to Celery
        task = celery_app.send_task(
            'worker.generate_3d',
            args=[
                request.prompt,
                request.model,
                request.guidance_scale,
                request.num_inference_steps
            ]
        )

        return JobResponse(
            job_id=task.id,
            status="submitted",
            message=f"Job submitted for model '{request.model}'"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_status(job_id: str):
    """
    Get job status and result
    
    Status values:
    - PENDING: Job is queued
    - STARTED: Job is running
    - SUCCESS: Job completed successfully
    - FAILURE: Job failed
    - RETRY: Job is being retried
    """
    try:
        result = celery_app.AsyncResult(job_id)

        response = JobStatusResponse(
            job_id=job_id,
            status=result.state
        )

        if result.state == 'PENDING':
            response.message = "Job is queued"

        elif result.state == 'STARTED':
            # Get progress if available
            if result.info and isinstance(result.info, dict):
                response.progress = result.info.get('progress', 0.0)
                response.message = result.info.get('message', 'Processing...')
            else:
                response.message = "Job started"

        elif result.state == 'SUCCESS':
            response.message = "Generation complete"
            response.progress = 1.0
            response.result = result.result

        elif result.state == 'FAILURE':
            response.message = "Job failed"
            response.error = str(result.info)

        elif result.state == 'RETRY':
            response.message = "Job is being retried"

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job"""
    try:
        celery_app.control.revoke(job_id, terminate=True)
        return {"job_id": job_id, "status": "cancelled"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/queue")
async def queue_info():
    """Get queue statistics"""
    inspect = celery_app.control.inspect()

    return {
        "active": inspect.active() or {},
        "scheduled": inspect.scheduled() or {},
        "reserved": inspect.reserved() or {}
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)