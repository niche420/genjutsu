"""
FastAPI service for job submission and status
"""
from fastapi import FastAPI, HTTPException
import sys
from pathlib import Path

# Add parent directory to path for shared module
sys.path.insert(0, str(Path(__file__).parent))

from shared.celery_app import celery_app
from shared.config import OUTPUT_DIR
from shared.schemas.jobs import *

app = FastAPI(title="Genjutsu 3D Generation API")

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


@app.post("/generate", response_model=JobCreateResponse)
async def generate(request: JobCreateRequest):
    """
    Submit a 3D generation job
    
    Returns job_id for tracking progress
    """
    try:
        task = celery_app.send_task(
            'worker.generate_3d',
            args=[
                request.prompt,
                request.model,
                request.guidance_scale,
                request.num_inference_steps
            ]
        )

        return JobCreateResponse(
            id=task.id,
            status=JobStatus.QUEUED,
            message=f"Job submitted for model '{request.model}'"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_status(job_id: str):
    """
    Get job status and result
    """
    try:
        result = celery_app.AsyncResult(job_id)

        # Map Celery states to our JobStatus enum
        state_mapping = {
            'PENDING': JobStatus.QUEUED,
            'STARTED': JobStatus.GENERATING,
            'SUCCESS': JobStatus.COMPLETE,
            'FAILURE': JobStatus.FAILED,
            'RETRY': JobStatus.GENERATING,
            'REVOKED': JobStatus.FAILED,
        }

        status = state_mapping.get(result.state, JobStatus.QUEUED)

        response = JobStatusResponse(
            id=job_id,
            data=JobMetadata(
                status=status,
                progress=0.0,
                message=None,
                error=None
            )
        )

        if result.state == 'PENDING':
            response.data.message = "Job is queued"

        elif result.state == 'STARTED':
            # Get progress if available
            if result.info and isinstance(result.info, dict):
                response.data.progress = result.info.get('progress', 0.0)
                response.data.message = result.info.get('message', 'Processing...')
            else:
                response.data.message = "Job started"

        elif result.state == 'SUCCESS':
            response.data.message = "Generation complete"
            response.data.progress = 1.0
            if result.result and isinstance(result.result, dict):
                response.outputs = JobOutputs(ply_path=result.result.get('ply_path', ''))

        elif result.state == 'FAILURE':
            response.data.message = "Job failed"
            response.data.error = str(result.info)

        elif result.state == 'RETRY':
            response.data.message = "Job is being retried"

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job"""
    try:
        celery_app.control.revoke(job_id, terminate=True)
        return {"job_id": job_id, "status": "REVOKED"}
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