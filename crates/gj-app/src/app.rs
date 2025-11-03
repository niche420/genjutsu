use egui_wgpu::wgpu;
use winit::{
    event::*,
    event_loop::EventLoop,
    window::Window,
};
use egui_winit::State as EguiState;
use crate::state::AppState;
use crate::ui;

pub(crate) struct App<'wnd> {
    pub(crate) window: &'wnd Window,
    pub(crate) app_state: AppState<'wnd>,
    egui_state: EguiState,
    egui_renderer: egui_wgpu::Renderer,
}

impl<'wnd> App<'wnd> {
    pub(crate) async fn new(window: &'wnd Window) -> Self {
        let app_state = AppState::new(&window).await;

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            None,
            None,
            None
        );

        let egui_renderer = egui_wgpu::Renderer::new(
            &app_state.device,
            app_state.config.format,
            egui_wgpu::RendererOptions::default()
        );

        Self {
            window,
            app_state,
            egui_state,
            egui_renderer,
        }
    }



    pub(crate) fn input(&mut self, event: &WindowEvent) -> bool {
        let response = self.egui_state.on_window_event(&self.window, event);

        if response.consumed {
            return true;
        }

        self.app_state.input(event)
    }

    pub(crate) fn update(&mut self) {
        self.app_state.update();
    }

    pub(crate) fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.app_state.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.app_state.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            }
        );

        // Render 3D scene
        self.app_state.render_scene(&mut encoder, &view);

        // Prepare egui
        let egui_state = &mut self.egui_state;
        let raw_input = egui_state.take_egui_input(&self.window);

        let full_output = egui_state.egui_ctx().run(raw_input, |ctx| {
            ui::draw_ui(ctx, &mut self.app_state);
        });

        // Render egui
        egui_state.handle_platform_output(self.window, full_output.platform_output);

        let paint_jobs = egui_state.egui_ctx().tessellate(full_output.shapes, full_output.pixels_per_point);

        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.app_state.size.width, self.app_state.size.height],
            pixels_per_point: self.window.scale_factor() as f32,
        };

        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(
                &self.app_state.device,
                &self.app_state.queue,
                *id,
                image_delta,
            );
        }

        self.egui_renderer.update_buffers(
            &self.app_state.device,
            &self.app_state.queue,
            &mut encoder,
            &paint_jobs,
            &screen_descriptor,
        );

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Egui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            self.egui_renderer.render(&mut render_pass.forget_lifetime(), &paint_jobs, &screen_descriptor);
        }

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.app_state.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    pub(crate) fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.app_state.resize(new_size);
    }
}

