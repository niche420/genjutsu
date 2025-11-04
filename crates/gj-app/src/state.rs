use burn_ndarray::NdArray;
use burn_wgpu::Wgpu;
use egui_wgpu::wgpu;
use winit::window::Window;
use gj_core::gaussian_cloud::GaussianCloud;
use gj_lgm::LGMPipeline;
use gj_splat::camera::Camera;
use gj_splat::renderer::GaussianRenderer;

pub struct AppState<'wnd> {
    pub surface: wgpu::Surface<'wnd>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,

    // Renderer
    pub renderer: GaussianRenderer,
    pub camera: Camera,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,

    // LGM
    pub lgm_pipeline: LGMPipeline<Wgpu>,

    // UI State
    pub prompt: String,
    pub status: String,
    pub gaussian_cloud: Option<GaussianCloud>,
    pub mouse_pressed: bool,
    pub last_mouse_pos: Option<(f32, f32)>,
}

impl<'wnd> AppState<'wnd> {
    pub async fn new(window: &'wnd Window) -> Self {

        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    experimental_features: Default::default(),
                    memory_hints: Default::default(),
                    trace: Default::default(),
                },
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let depth_texture = Self::create_depth_texture(&device, &size);
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let renderer = GaussianRenderer::new(device.clone(), queue.clone(), surface_format).await;

        let mut camera = Camera::default();
        camera.aspect_ratio = size.width as f32 / size.height as f32;

        let lgm_device = Default::default();
        let lgm_pipeline = LGMPipeline::new(lgm_device);

        Self {
            surface,
            device,
            queue,
            config,
            size,
            renderer,
            camera,
            depth_texture,
            depth_view,
            lgm_pipeline,
            prompt: String::new(),
            status: "Ready".to_string(),
            gaussian_cloud: None,
            mouse_pressed: false,
            last_mouse_pos: None,
        }
    }

    fn create_depth_texture(device: &wgpu::Device, size: &winit::dpi::PhysicalSize<u32>) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            self.depth_texture = Self::create_depth_texture(&self.device, &new_size);
            self.depth_view = self.depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

            self.camera.aspect_ratio = new_size.width as f32 / new_size.height as f32;
        }
    }

    pub fn input(&mut self, event: &winit::event::WindowEvent) -> bool {
        use winit::event::{WindowEvent, MouseButton, ElementState, MouseScrollDelta};

        match event {
            WindowEvent::MouseInput { button: MouseButton::Left, state, .. } => {
                self.mouse_pressed = *state == ElementState::Pressed;
                if !self.mouse_pressed {
                    self.last_mouse_pos = None;
                }
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                let current_pos = (position.x as f32, position.y as f32);

                if self.mouse_pressed {
                    if let Some((last_x, last_y)) = self.last_mouse_pos {
                        let delta_x = current_pos.0 - last_x;
                        let delta_y = current_pos.1 - last_y;
                        self.camera.rotate(delta_x * 0.1, -delta_y * 0.1);
                    }
                }

                self.last_mouse_pos = Some(current_pos);
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 10.0,
                };
                self.camera.zoom(-scroll * 0.1);
                true
            }
            _ => false,
        }
    }

    pub fn update(&mut self) {
        // Update logic here
    }

    pub fn render_scene(&mut self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        if self.gaussian_cloud.is_some() {
            self.renderer.render(encoder, view, &self.depth_view, &self.camera);
        } else {
            // Clear to background color
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Clear Pass"),
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
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }
    }

    pub fn load_images(&mut self, paths: Vec<std::path::PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
        if paths.len() != 4 {
            self.status = format!("Error: Need exactly 4 images, got {}", paths.len());
            return Ok(());
        }

        self.status = "Loading images...".to_string();

        let images: Result<Vec<_>, _> = paths.iter()
            .map(|path| image::open(path).map(|img| img.to_rgba8()))
            .collect();

        let images = images?;

        self.status = "Generating 3D...".to_string();

        let cloud = self.lgm_pipeline.generate(&images)?;

        self.renderer.load_gaussians(&cloud);
        self.gaussian_cloud = Some(cloud.clone());

        self.status = format!("Generated {} Gaussians", cloud.count);

        Ok(())
    }
}