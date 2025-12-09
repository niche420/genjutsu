"""
Shap-E: Text-to-3D generation using OpenAI's Shap-E model
Fast inference (~1 minute) with high-quality Gaussian splat output
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image

from .model import Model3DBase


class ShapEModel(Model3DBase):
    """OpenAI Shap-E: Fast text-to-3D generation"""

    def __init__(self, device="cuda"):
        super().__init__(device)
        self.text_model = None
        self.diffusion_model = None

    def load(self) -> bool:
        """Load Shap-E models"""
        try:
            print("  Loading Shap-E models...")

            from shap_e.diffusion.sample import sample_latents
            from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
            from shap_e.models.download import load_model, load_config
            from shap_e.util.notebooks import decode_latent_mesh

            # Store functions we'll need
            self.sample_latents = sample_latents
            self.diffusion_from_config = diffusion_from_config
            self.load_model = load_model
            self.load_config = load_config
            self.decode_latent_mesh = decode_latent_mesh

            # Load text-to-latent model
            print("    Loading text encoder...")
            self.text_model = load_model('text300M', device=self.device)

            # Load latent diffusion model
            print("    Loading diffusion model...")
            self.diffusion_model = load_model('transmitter', device=self.device)

            self.is_loaded = True
            print("  ✓ Shap-E loaded successfully")
            return True

        except Exception as e:
            print(f"  ✗ Failed to load Shap-E: {e}")
            print(f"  → Make sure you installed: pip install git+https://github.com/openai/shap-e.git")
            return False

    def generate(self, prompt: str, output_path: Path, **kwargs) -> Path:
        """Generate 3D Gaussians from text prompt"""
        if not self.is_loaded:
            raise RuntimeError("Shap-E not loaded")

        guidance_scale = kwargs.get('guidance_scale', 15.0)
        num_inference_steps = kwargs.get('num_inference_steps', 64)

        print(f"  Generating with Shap-E: '{prompt}'")
        print(f"  Guidance scale: {guidance_scale}")
        print(f"  Inference steps: {num_inference_steps}")

        # Generate latents
        print("  [1/3] Generating latent representation...")
        latents = self.sample_latents(
            batch_size=1,
            model=self.text_model,
            diffusion=self.diffusion_from_config(self.load_config('diffusion')),
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=[prompt]),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=num_inference_steps,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

        print("  [2/3] Decoding to mesh...")

        # Get the mesh with higher resolution
        from shap_e.util.notebooks import decode_latent_mesh

        # Try different resolutions to get better quality
        try:
            # High resolution (slower but better)
            mesh = decode_latent_mesh(self.diffusion_model, latents[0]).tri_mesh()
            print(f"    ✓ Generated mesh with {len(mesh.verts)} vertices")
        except Exception as e:
            print(f"    ✗ Failed to decode mesh: {e}")
            raise

        # Check if mesh is degenerate
        vertices = mesh.verts
        bounds = vertices.max(axis=0) - vertices.min(axis=0)
        print(f"    Mesh bounds: X={bounds[0]:.3f}, Y={bounds[1]:.3f}, Z={bounds[2]:.3f}")

        if bounds.min() < 0.01:
            print(f"    ⚠️  Warning: Mesh appears flat/degenerate!")
            print(f"    Consider using a different prompt or higher guidance scale")

        print("  [3/3] Converting to Gaussian splats...")
        self._export_to_ply(mesh, output_path)

        print(f"  ✓ Saved to {output_path}")
        return output_path

    def get_name(self) -> str:
        return "Shap-E"

    def get_estimated_time(self, **kwargs) -> int:
        """Returns time in seconds"""
        num_steps = kwargs.get('num_inference_steps', 64)
        return 30 + (num_steps * 0.5)  # ~30-60 seconds

    def _export_to_ply(self, mesh, output_path):
        """Convert Shap-E mesh to PLY with proper Gaussian splat data"""
        import numpy as np
        from scipy.spatial import cKDTree

        vertices = mesh.verts
        num_points = len(vertices)

        if num_points == 0:
            raise ValueError("Mesh has no vertices!")

        print(f"  Converting {num_points} vertices to Gaussian splats...")

        # === COLORS ===
        if 'R' in mesh.vertex_channels and 'G' in mesh.vertex_channels and 'B' in mesh.vertex_channels:
            R = mesh.vertex_channels['R']
            G = mesh.vertex_channels['G']
            B = mesh.vertex_channels['B']
            colors = np.stack([R, G, B], axis=-1)
            print(f"    ✓ Using separate R, G, B channels")
        elif 'color' in mesh.vertex_channels:
            colors = mesh.vertex_channels['color']
            print(f"    ✓ Using 'color' channel")
        else:
            print(f"    ✗ No color channels, using normals")
            if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                normals = mesh.vertex_normals
                colors = (normals + 1.0) / 2.0
            else:
                colors = np.random.rand(num_points, 3) * 0.5 + 0.5  # Random colors

        colors = np.clip(colors, 0.0, 1.0)
        colors_uint8 = (colors * 255).astype(np.uint8)

        print(f"    Color range: [{colors.min():.3f}, {colors.max():.3f}]")
        print(f"    First 3 colors: {colors[:3]}")

        # === SCALES ===
        print(f"    Computing scales...")
        tree = cKDTree(vertices)
        k = min(8, num_points)
        distances, _ = tree.query(vertices, k=k)
        avg_distances = distances[:, 1:].mean(axis=1)

        scales = np.zeros((num_points, 3), dtype=np.float32)
        scales[:, 0] = avg_distances * 0.5
        scales[:, 1] = avg_distances * 0.5
        scales[:, 2] = avg_distances * 0.4
        scales = np.clip(scales, 0.005, 0.2)

        print(f"    Scale range: [{scales.min():.4f}, {scales.max():.4f}]")

        # === ROTATIONS ===
        rotations = np.zeros((num_points, 4), dtype=np.float32)
        rotations[:, 0] = 1.0  # Identity quaternion (w=1, x=0, y=0, z=0)

        # === OPACITY ===
        opacities = np.ones(num_points, dtype=np.float32) * 0.95

        # === WRITE PLY (CAREFULLY) ===
        print(f"    Writing PLY to {output_path}...")

        # Use struct to ensure correct binary layout
        import struct

        with open(output_path, 'wb') as f:
            # ASCII header
            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {num_points}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property float nx\n"
                "property float ny\n"
                "property float nz\n"
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
                "property float opacity\n"
                "property float scale_0\n"
                "property float scale_1\n"
                "property float scale_2\n"
                "property float rot_0\n"
                "property float rot_1\n"
                "property float rot_2\n"
                "property float rot_3\n"
                "end_header\n"
            )
            f.write(header.encode('ascii'))

            # Binary data - write each vertex
            for i in range(num_points):
                # Position (3 floats)
                f.write(struct.pack('fff', vertices[i][0], vertices[i][1], vertices[i][2]))

                # Normal (3 floats - placeholder)
                f.write(struct.pack('fff', 0.0, 0.0, 0.0))

                # Color (3 uint8)
                f.write(struct.pack('BBB', colors_uint8[i][0], colors_uint8[i][1], colors_uint8[i][2]))

                # Opacity (1 float)
                f.write(struct.pack('f', opacities[i]))

                # Scale (3 floats)
                f.write(struct.pack('fff', scales[i][0], scales[i][1], scales[i][2]))

                # Rotation (4 floats)
                f.write(struct.pack('ffff', rotations[i][0], rotations[i][1], rotations[i][2], rotations[i][3]))

        print(f"    ✓ Successfully wrote {num_points} splats")

        # Verify file was written
        file_size = output_path.stat().st_size
        expected_size = len(header.encode('ascii')) + (num_points * 59)
        print(f"    File size: {file_size} bytes (expected: {expected_size})")

        if file_size != expected_size:
            raise ValueError(f"PLY file size mismatch! Got {file_size}, expected {expected_size}")