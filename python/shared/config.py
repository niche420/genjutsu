"""
Shared configuration for API and Worker
"""
import os
from pathlib import Path

# Redis/Celery settings
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', REDIS_URL)

# Celery task settings
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_TIME_LIMIT = 3600  # 1 hour max
CELERY_TASK_SOFT_TIME_LIMIT = 3300  # 55 minutes soft limit
CELERY_RESULT_EXPIRES = 3600  # Results expire after 1 hour

# Output directory
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', '../outputs'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Device
DEVICE = 'cuda' if os.getenv('CUDA_VISIBLE_DEVICES') else 'cpu'

# Job defaults
DEFAULT_GUIDANCE_SCALE = 15.0
DEFAULT_NUM_INFERENCE_STEPS = 64