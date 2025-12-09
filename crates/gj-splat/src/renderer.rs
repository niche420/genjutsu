use wgpu::util::DeviceExt;
use gj_core::gaussian_cloud::GaussianCloud;
use crate::camera::Camera;

// Quad vertices for instanced rendering (4 corners of a billboard)
const QUAD_VERTICES: &[[f32; 2]] = &[
    [-1.0, -1.0],  // bottom-left
    [1.0, -1.0],   // bottom-right
    [-1.0, 1.0],   // top-left
    [1.0, 1.0],    // top-right
];

const QUAD_INDICES: &[u16] = &[0, 1, 2, 2, 1, 3];

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GaussianInstance {
    position: [f32; 3],
    _padding1: f32,
    color: [f32; 3],
    opacity: f32,
    scale: [f32; 3],
    _padding2: f32,
    rotation: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _padding1: f32,
    viewport: [f32; 2],
    focal: [f32; 2],
}

pub struct GaussianRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,

    quad_vertex_buffer: wgpu::Buffer,
    quad_index_buffer: wgpu::Buffer,
    instance_buffer: Option<wgpu::Buffer>,

    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,

    num_gaussians: u32,

    // Cache last camera state to avoid redundant updates
    last_view_proj: Option<[[f32; 4]; 4]>,
}

impl GaussianRenderer {
    pub async fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        format: wgpu::TextureFormat,
    ) -> Self {
        // Use the simplified, faster shader
        let shader_source = include_str!("../shaders/gaussian.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gaussian Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create quad buffers
        let quad_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Quad Vertex Buffer"),
            contents: bytemuck::cast_slice(QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let quad_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Quad Index Buffer"),
            contents: bytemuck::cast_slice(QUAD_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Gaussian Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[
                    // Quad vertices (per-vertex)
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<[f32; 2]>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        }],
                    },
                    // Gaussian instances (per-instance)
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<GaussianInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 1, // position
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 12,
                                shader_location: 2, // _padding1
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 16,
                                shader_location: 3, // color
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 28,
                                shader_location: 4, // opacity
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 32,
                                shader_location: 5, // scale
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 44,
                                shader_location: 6, // _padding2
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 48,
                                shader_location: 7, // rotation
                            },
                        ],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Don't cull for splats
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false, // Splats use alpha blending, not depth
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            device,
            queue,
            pipeline,
            quad_vertex_buffer,
            quad_index_buffer,
            instance_buffer: None,
            uniform_buffer,
            bind_group,
            num_gaussians: 0,
            last_view_proj: None,
        }
    }

    pub fn load_gaussians(&mut self, cloud: &GaussianCloud) {
        let instances: Vec<GaussianInstance> = (0..cloud.count)
            .filter(|&i| {
                let opacity = cloud.opacity[i];
                let scale_avg = (cloud.scales[i][0] + cloud.scales[i][1] + cloud.scales[i][2]) / 3.0;

                // Keep more splats - only filter out garbage
                opacity > 0.01 &&
                    scale_avg > 0.0001 &&
                    scale_avg < 10.0 &&
                    cloud.positions[i][0].is_finite() &&
                    cloud.positions[i][1].is_finite() &&
                    cloud.positions[i][2].is_finite()
            })
            .map(|i| GaussianInstance {
                position: cloud.positions[i],
                _padding1: 0.0,
                color: cloud.colors[i],  // USE ACTUAL COLORS
                opacity: cloud.opacity[i],  // DON'T multiply by 0.4!
                scale: cloud.scales[i],  // DON'T multiply by 0.5!
                _padding2: 0.0,
                rotation: cloud.rotations[i],
            })
            .collect();

        self.instance_buffer = Some(
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&instances),
                usage: wgpu::BufferUsages::VERTEX,
            })
        );

        self.num_gaussians = instances.len() as u32;
        self.last_view_proj = None;

        println!("Rendered {} / {} gaussians ({:.1}% kept)",
                 instances.len(), cloud.count,
                 100.0 * instances.len() as f32 / cloud.count.max(1) as f32);
    }

    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera: &Camera,
        viewport_size: (u32, u32),
    ) {
        // Skip if no gaussians loaded
        if self.num_gaussians == 0 {
            return;
        }

        // Calculate focal length from FOV
        let fov_rad = camera.fov.to_radians();
        let focal_y = viewport_size.1 as f32 / (2.0 * (fov_rad / 2.0).tan());
        let focal_x = focal_y * camera.aspect_ratio;

        let view_proj = camera.view_projection_matrix().to_cols_array_2d();

        // Only update uniforms if camera actually changed
        let needs_update = self.last_view_proj.as_ref() != Some(&view_proj);

        if needs_update {
            let uniforms = Uniforms {
                view_proj,
                view: camera.view_matrix().to_cols_array_2d(),
                camera_pos: camera.position.to_array(),
                _padding1: 0.0,
                viewport: [viewport_size.0 as f32, viewport_size.1 as f32],
                focal: [focal_x, focal_y],
            };

            self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
            self.last_view_proj = Some(view_proj);
        }

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Gaussian Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.1,
                        b: 0.1,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.quad_vertex_buffer.slice(..));

        if let Some(ref instance_buffer) = self.instance_buffer {
            render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
            render_pass.set_index_buffer(self.quad_index_buffer.slice(..), wgpu::IndexFormat::Uint16);

            // Draw instanced quads - 6 indices per quad, num_gaussians instances
            render_pass.draw_indexed(0..6, 0, 0..self.num_gaussians);
        }
    }
}