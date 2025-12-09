"""
Celery worker for 3D generation tasks
"""
import sys
from pathlib import Path
from datetime import datetime
import torch

# Add parent directory to path for shared module
sys.path.insert(0, str(Path(__file__).parent))

from shared.celery_app import celery_app
from shared.config import OUTPUT_DIR, DEVICE
from models.shap_e import ShapEModel


# Load models on worker startup
print("=" * 60)
print("Initializing Genjutsu Worker")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Output: {OUTPUT_DIR}")
print()

# Initialize models
MODELS = {}

print("Loading Shap-E...")
shap_e = ShapEModel(DEVICE)
if shap_e.load():
    MODELS['shap_e'] = shap_e
    print("✓ Shap-E ready")
else:
    print("✗ Shap-E failed to load")

print()
print(f"Loaded {len(MODELS)} model(s)")
print("=" * 60)
print()


@celery_app.task(name='worker.generate_3d', bind=True)
def generate_3d(self, prompt: str, model_name: str, guidance_scale: float, num_inference_steps: int):
    """
    Generate 3D model from text prompt

    Args:
        self: Task instance (for progress updates)
        prompt: Text description
        model_name: Model to use
        guidance_scale: Guidance scale parameter
        num_inference_steps: Number of diffusion steps

    Returns:
        dict with output_path and metadata
    """
    try:
        # Update state to STARTED
        self.update_state(
            state='STARTED',
            meta={
                'progress': 0.0,
                'message': f'Starting {model_name} generation...'
            }
        )

        # Check model exists
        if model_name not in MODELS:
            raise ValueError(f"Model '{model_name}' not available")

        model = MODELS[model_name]

        # Create output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '_')).strip()
        safe_prompt = safe_prompt[:50].replace(' ', '_')

        output_name = f"{model_name}_{safe_prompt}_{timestamp}.ply"
        output_path = OUTPUT_DIR / output_name

        print(f"\n{'='*60}")
        print(f"Job ID: {self.request.id}")
        print(f"Model: {model.get_name()}")
        print(f"Prompt: {prompt}")
        print(f"Output: {output_path}")
        print(f"Guidance: {guidance_scale}")
        print(f"Steps: {num_inference_steps}")
        print(f"{'='*60}\n")

        # Progress callback
        def progress_callback(progress: float, message: str):
            self.update_state(
                state='STARTED',
                meta={
                    'progress': progress,
                    'message': message
                }
            )
            print(f"[{progress*100:.0f}%] {message}")

        # Update progress
        progress_callback(0.1, 'Initializing model...')

        # Generate
        try:
            result_path = model.generate(
                prompt,
                output_path,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
        except ValueError as e:
            # Generation failed - return helpful error
            error_msg = str(e)
            if "flat" in error_msg.lower() or "degenerate" in error_msg.lower():
                raise ValueError(
                    f"Generated mesh is flat/invalid. Try:\n"
                    f"  • More specific prompt (add details about shape/structure)\n"
                    f"  • Higher guidance scale (try 20-25)\n"
                    f"  • Different prompt entirely"
                )
            raise

        progress_callback(1.0, 'Complete!')

        # Return result
        return {
            'output_path': str(result_path),
            'model': model_name,
            'prompt': prompt,
            'guidance_scale': guidance_scale,
            'num_inference_steps': num_inference_steps
        }

    except Exception as e:
        print(f"\n✗ Job failed: {str(e)}\n")
        raise


if __name__ == '__main__':
    # Start worker
    celery_app.worker_main([
        'worker',
        '--loglevel=info',
        '--concurrency=1',  # Single worker (GPU)
        '--pool=solo'  # Use solo pool for GPU work
    ])