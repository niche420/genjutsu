#!/usr/bin/env python3
"""
GaussianDreamer Service - Text-to-Gaussian Generation
Runs as a background service that Rust can communicate with via HTTP
"""

import argparse
import os

from flask import Flask, request, jsonify
from pathlib import Path
import torch
from datetime import datetime

app = Flask(__name__)

# GaussianDreamer will be loaded here
# You'll need to clone and set it up: https://github.com/hustvl/GaussianDreamer
try:
    # This is a placeholder - actual imports depend on GaussianDreamer's structure
    # from gaussiandreamer import GaussianDreamerPipeline
    GAUSSIANDREAMER_AVAILABLE = False
    print("⚠️  GaussianDreamer not installed. Using placeholder mode.")
except ImportError:
    GAUSSIANDREAMER_AVAILABLE = False
    print("⚠️  GaussianDreamer not found. Please install it first.")

class GaussianDreamerService:
    def __init__(self, output_dir="./outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if GAUSSIANDREAMER_AVAILABLE:
            print("✓ Loading GaussianDreamer model...")
            # self.pipeline = GaussianDreamerPipeline(
            #     device="cuda" if torch.cuda.is_available() else "cpu"
            # )
            print("✓ GaussianDreamer ready!")
        else:
            self.pipeline = None
            print("✗ Running in placeholder mode")

    def generate(self, prompt, guidance_scale=7.5, num_iterations=500):
        """
        Generate Gaussian Splats from text prompt

        Args:
            prompt: Text description
            guidance_scale: CFG scale for diffusion
            num_iterations: Number of optimization steps

        Returns:
            path to generated .ply file
        """
        print(f"Generating: '{prompt}'")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '_')).rstrip()
        safe_prompt = safe_prompt[:50].replace(' ', '_')
        output_name = f"{safe_prompt}_{timestamp}"
        output_path = self.output_dir / f"{output_name}.ply"

        if GAUSSIANDREAMER_AVAILABLE and self.pipeline:
            # Real GaussianDreamer generation
            result = self.pipeline(
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_iterations=num_iterations,
                output_path=str(output_path)
            )
            return str(output_path)
        else:
            # Placeholder: generate a simple test .ply file
            return self._generate_placeholder_ply(output_path, prompt)

    def _generate_placeholder_ply(self, output_path, prompt):
        """Generate a simple placeholder .ply for testing"""
        print(f"Generating placeholder .ply at {output_path}")

        # Create a simple colored point cloud in .ply format
        # This is just for testing - replace with real GaussianDreamer

        import numpy as np

        # Generate points in a sphere
        num_points = 5000
        phi = np.random.uniform(0, np.pi, num_points)
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        r = np.random.uniform(0.5, 1.0, num_points)

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        # Color based on prompt
        if "red" in prompt.lower():
            colors = np.array([255, 100, 100])
        elif "blue" in prompt.lower():
            colors = np.array([100, 100, 255])
        elif "yellow" in prompt.lower():
            colors = np.array([255, 255, 100])
        else:
            colors = np.array([150, 150, 180])

        # Add some variation
        color_array = np.tile(colors, (num_points, 1))
        color_array += np.random.randint(-30, 30, (num_points, 3))
        color_array = np.clip(color_array, 0, 255).astype(np.uint8)

        # Default Gaussian parameters
        scale = np.ones((num_points, 3)) * 0.05
        rotation = np.tile([1, 0, 0, 0], (num_points, 1))  # identity quaternion
        opacity = np.ones(num_points) * 0.8

        # Write PLY file with Gaussian splat format
        with open(output_path, 'wb') as f:
            # Header
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

            # Binary data
            for i in range(num_points):
                # Position (3 floats)
                f.write(x[i].astype(np.float32).tobytes())
                f.write(y[i].astype(np.float32).tobytes())
                f.write(z[i].astype(np.float32).tobytes())

                # Normal (placeholder - 3 floats)
                f.write(np.float32(0).tobytes())
                f.write(np.float32(0).tobytes())
                f.write(np.float32(0).tobytes())

                # Color (3 bytes)
                f.write(bytes(color_array[i]))

                # Opacity (1 float)
                f.write(opacity[i].astype(np.float32).tobytes())

                # Scale (3 floats)
                f.write(scale[i].astype(np.float32).tobytes())

                # Rotation quaternion (4 floats)
                f.write(rotation[i].astype(np.float32).tobytes())

        print(f"✓ Placeholder .ply generated: {output_path}")
        return str(output_path)

# Global service instance
service = None

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "gaussiandreamer_available": GAUSSIANDREAMER_AVAILABLE,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    })

@app.route('/generate', methods=['POST'])
def generate():
    """
    Generate Gaussian Splats from text

    POST body:
    {
        "prompt": "a red sports car",
        "guidance_scale": 7.5,
        "num_iterations": 500
    }
    """
    data = request.json
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    guidance_scale = data.get('guidance_scale', 7.5)
    num_iterations = data.get('num_iterations', 500)

    try:
        output_path = service.generate(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_iterations=num_iterations
        )

        return jsonify({
            "status": "success",
            "output_path": output_path,
            "prompt": prompt
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/list', methods=['GET'])
def list_outputs():
    """List all generated .ply files"""
    ply_files = list(service.output_dir.glob("*.ply"))
    return jsonify({
        "files": [str(f) for f in ply_files]
    })

def main():
    parser = argparse.ArgumentParser(description="GaussianDreamer Service")
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--output-dir', default='../outputs', help='Output directory for .ply files')

    args = parser.parse_args()

    global service
    service = GaussianDreamerService(output_dir=args.output_dir)

    print(f"""
╔════════════════════════════════════════════════════════════╗
║        GaussianDreamer Service                             ║
║        Text-to-Gaussian Generation API                     ║
╚════════════════════════════════════════════════════════════╝

Listening on: http://{args.host}:{args.port}
Output directory: {args.output_dir}

Endpoints:
  GET  /health          - Health check
  POST /generate        - Generate Gaussians from text
  GET  /list            - List generated files

Example request:
  curl -X POST http://{args.host}:{args.port}/generate \\
    -H "Content-Type: application/json" \\
    -d '{{"prompt": "a red sports car"}}'
""")

    app.run(host=args.host, port=args.port, debug=False)

if __name__ == '__main__':
    main()