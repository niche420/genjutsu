// ===========================================================================
// splat/src/shaders/gaussian.wgsl - Gaussian Splatting Shader
// ===========================================================================

struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) opacity: f32,
    @location(3) scale: vec3<f32>,
    @location(4) rotation: vec4<f32>,  // quaternion (w, x, y, z)
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) opacity: f32,
    @location(3) scale: vec3<f32>,
    @location(4) rotation: vec4<f32>,
    @builtin(point_size) point_size: f32,
}

// Rotate vector by quaternion
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = q.yzw;
    let uv = cross(qv, v);
    let uuv = cross(qv, uv);
    return v + ((uv * q.x) + uuv) * 2.0;
}

// Calculate Gaussian splat size in screen space
fn calculate_splat_size(position: vec3<f32>, scale: vec3<f32>) -> f32 {
    // Distance from camera
    let dist = distance(position, uniforms.camera_pos);
    
    // Average scale
    let avg_scale = (scale.x + scale.y + scale.z) / 3.0;
    
    // Project to screen space (approximation)
    // Closer gaussians should be larger
    let size = (avg_scale * 100.0) / max(dist, 0.1);
    
    return clamp(size, 1.0, 100.0);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Transform Gaussian center to clip space
    let world_pos = vec4<f32>(in.position, 1.0);
    out.clip_position = uniforms.view_proj * world_pos;
    
    // Pass through attributes
    out.world_position = in.position;
    out.color = in.color;
    out.opacity = in.opacity;
    out.scale = in.scale;
    out.rotation = in.rotation;
    
    // Calculate point size for splatting
    out.point_size = calculate_splat_size(in.position, in.scale);
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Get fragment position in point space [-1, 1]
    let coord = (in.clip_position.xy / in.clip_position.w) * 2.0 - 1.0;
    
    // Simple circular Gaussian falloff
    let dist_sq = dot(coord, coord);
    
    // Gaussian function: exp(-dist²/2σ²)
    // Using σ = 0.5 for visible falloff
    let gaussian = exp(-dist_sq / (2.0 * 0.5 * 0.5));
    
    // Modulate opacity by Gaussian falloff
    let final_opacity = in.opacity * gaussian;
    
    // Discard pixels with very low opacity
    if (final_opacity < 0.01) {
        discard;
    }
    
    // Output color with opacity
    return vec4<f32>(in.color, final_opacity);
}

// ===========================================================================
// Advanced shader (for future use)
// ===========================================================================

// This would implement proper 3D Gaussian splatting with:
// - Covariance matrix computation
// - Elliptical splat shape
// - View-dependent shading
// - Spherical harmonics for color

/*
@fragment
fn fs_main_advanced(in: VertexOutput) -> @location(0) vec4<f32> {
    // 1. Compute 3D covariance matrix from scale and rotation
    // Σ = R * S * S^T * R^T
    
    // 2. Project to 2D screen space
    // J = Jacobian of projection
    // Σ' = J * Σ * J^T
    
    // 3. Compute inverse of 2D covariance
    // Σ'^-1
    
    // 4. Evaluate 2D Gaussian
    // G(x) = exp(-0.5 * (x - μ)^T * Σ'^-1 * (x - μ))
    
    // 5. Apply spherical harmonics for view-dependent color
    // color = SH(view_direction)
    
    return vec4<f32>(color, opacity * gaussian_value);
}
*/