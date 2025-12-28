"""
Celery tasks for 3D generation
This module is imported by both the API and the worker
"""
import sys
from pathlib import Path

# Add parent directory to path for shared module
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.celery_app import celery_app

# Import the actual task implementation
from worker.worker import generate_3d

# Re-export for API to import
__all__ = ['generate_3d']