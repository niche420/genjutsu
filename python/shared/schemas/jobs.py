from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class JobCreateRequest(BaseModel):
    prompt: str = Field(..., description="Text description of 3D object")
    model: str = Field(default="shap_e", description="Model to use")
    guidance_scale: float = Field(default=15.0, ge=1.0, le=30.0)
    num_inference_steps: int = Field(default=64, ge=16, le=256)

class JobStatus(str, Enum):
    """Job status matching Rust JobStatus enum exactly"""
    QUEUED = "QUEUED"
    GENERATING = "GENERATING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"

class JobCreateResponse(BaseModel):
    id: str
    status: JobStatus
    message: Optional[str] = None

class JobOutputs(BaseModel):
    ply_path: str

class JobMetadata(BaseModel):
    status: JobStatus
    progress: float = 0.0
    message: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None

class JobStatusResponse(BaseModel):
    id: str
    data: JobMetadata
    outputs: Optional[JobOutputs] = None