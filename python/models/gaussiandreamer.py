import sys
from pathlib import Path

from .model import Model3DBase

class GaussianDreamerModel(Model3DBase):
    """GaussianDreamer: Text to 3D Gaussians"""

    def __init__(self, device="cuda"):
        super().__init__(device)
        self.pipeline = None

    def load(self) -> bool:
        """Load GaussianDreamer"""
        try:
            # Add GaussianDreamer to path
            gd_path = Path(__file__).parent.parent / "repositories" / "GaussianDreamer"
            if not gd_path.exists():
                print(f"  ✗ GaussianDreamer not found at {gd_path}")
                print(f"  → Clone it: git clone https://github.com/hustvl/GaussianDreamer.git {gd_path}")
                return False

            sys.path.insert(0, str(gd_path))

            # Import GaussianDreamer
            from repositories.GaussianDreamer.threestudio.utils.config import load_config
            from repositories.GaussianDreamer.threestudio.systems import GaussianDreamer

            # Load config
            config = load_config(str(gd_path / "configs" / "gaussiandreamer.yaml"))

            # Create system
            self.pipeline = GaussianDreamer(config.system)
            self.pipeline.to(self.device)

            self.is_loaded = True
            print("  ✓ GaussianDreamer loaded")
            return True

        except Exception as e:
            print(f"  ✗ Failed to load GaussianDreamer: {e}")
            return False

    def generate(self, prompt: str, output_path: Path, **kwargs) -> Path:
        """Generate with GaussianDreamer"""
        if not self.is_loaded:
            raise RuntimeError("GaussianDreamer not loaded")

        num_iterations = kwargs.get('num_iterations', 3000)
        guidance_scale = kwargs.get('guidance_scale', 7.5)

        print(f"  Generating with GaussianDreamer: '{prompt}'")
        print(f"  Estimated time: {self.get_estimated_time(num_iterations=num_iterations) // 60} minutes")

        # Set prompt
        self.pipeline.cfg.system.prompt_processor.prompt = prompt
        self.pipeline.cfg.system.guidance.guidance_scale = guidance_scale

        # Run optimization
        for step in range(num_iterations):
            if step % 100 == 0:
                print(f"    Step {step}/{num_iterations}")
            self.pipeline.training_step(step)

        # Export
        gaussians = self.pipeline.geometry.get_gaussians()
        self._export_gaussians(gaussians, output_path)

        print(f"  ✓ Saved to {output_path}")
        return output_path

    def get_name(self) -> str:
        return "GaussianDreamer"

    def get_estimated_time(self, **kwargs) -> int:
        """Returns time in seconds"""
        num_iterations = kwargs.get('num_iterations', 3000)
        return num_iterations * 1  # ~1 second per iteration

    def _export_gaussians(self, gaussians, output_path):
        """Export Gaussians to PLY format"""
        import numpy as np

        # Extract data
        positions = gaussians['xyz'].cpu().numpy()
        colors = gaussians['features_dc'].cpu().numpy()
        opacities = gaussians['opacity'].cpu().numpy()
        scales = gaussians['scaling'].cpu().numpy()
        rotations = gaussians['rotation'].cpu().numpy()

        num_points = len(positions)

        # Write PLY
        with open(output_path, 'wb') as f:
            header = f"""ply
format binary_little_endian 1.0
element vertex {num_points}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
            f.write(header.encode('ascii'))

            for i in range(num_points):
                f.write(positions[i].astype(np.float32).tobytes())
                f.write(np.zeros(3, dtype=np.float32).tobytes())  # normals
                f.write((colors[i] * 255).clip(0, 255).astype(np.uint8).tobytes())
                f.write(opacities[i].astype(np.float32).tobytes())
                f.write(scales[i].astype(np.float32).tobytes())
                f.write(rotations[i].astype(np.float32).tobytes())