import sys
from pathlib import Path
import torch
from PIL import Image
from .model import Model3DBase


class DreamGaussianModel(Model3DBase):
    """DreamGaussian: Image to 3D Gaussians (faster than GaussianDreamer)"""

    def __init__(self, device="cuda"):
        super().__init__(device)
        self.pipeline = None
        self.sd_model = None

    def load(self) -> bool:
        """Load DreamGaussian"""
        try:
            # Add DreamGaussian to path
            dg_path = Path(__file__).parent.parent / "repositories" / "dreamgaussian"
            if not dg_path.exists():
                print(f"  ✗ DreamGaussian not found at {dg_path}")
                print(f"  → Clone it: git clone https://github.com/dreamgaussian/dreamgaussian.git {dg_path}")
                return False

            sys.path.insert(0, str(dg_path))

            # Import components
            from guidance.sd_utils import StableDiffusion
            from dreamgaussian import DreamGaussian

            # Load models
            self.sd_model = StableDiffusion(self.device)
            self.pipeline = DreamGaussian(self.device)

            self.is_loaded = True
            print("  ✓ DreamGaussian loaded")
            return True

        except Exception as e:
            print(f"  ✗ Failed to load DreamGaussian: {e}")
            return False

    def generate(self, prompt: str, output_path: Path, **kwargs) -> Path:
        """Generate with DreamGaussian"""
        if not self.is_loaded:
            raise RuntimeError("DreamGaussian not loaded")

        print(f"  Generating with DreamGaussian: '{prompt}'")

        # Step 1: Generate image
        print("  [1/3] Generating image...")
        image = self.sd_model.prompt_to_img(prompt, **kwargs)

        # Step 2: Remove background
        print("  [2/3] Processing image...")
        image_rgba = self._remove_background(image)

        # Step 3: Generate 3D
        print("  [3/3] Generating 3D (2 minutes)...")
        self.pipeline.initialize_from_image(image_rgba)

        num_steps = kwargs.get('num_steps', 500)
        for step in range(num_steps):
            if step % 50 == 0:
                print(f"    Step {step}/{num_steps}")
            self.pipeline.optimize_step(step)

        # Export
        self.pipeline.export_ply(str(output_path))

        print(f"  ✓ Saved to {output_path}")
        return output_path

    def get_name(self) -> str:
        return "DreamGaussian"

    def get_estimated_time(self, **kwargs) -> int:
        num_steps = kwargs.get('num_steps', 500)
        return 50 + (num_steps * 0.2)  # 50s for image gen + 0.2s per step

    def _remove_background(self, image):
        """Remove background from image"""
        try:
            from rembg import remove
            return remove(image)
        except:
            print("    ⚠️  rembg not available, keeping background")
            return image