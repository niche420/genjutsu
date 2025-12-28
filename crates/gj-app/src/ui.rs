mod top_panel;
mod side_panel;
mod central_panel;
mod queue_panel;

pub use top_panel::TopPanel;
pub use side_panel::SidePanel;
pub use central_panel::CentralPanel;
pub use queue_panel::QueuePanel;

use std::sync::Arc;
use async_trait::async_trait;
use egui::Context;
use surrealdb_types::RecordId;
use winit::event_loop::EventLoopProxy;
use winit::window::Window;
use gj_core::Model3D;
use crate::generator::db::job::JobRecord;
use crate::events::{AppEvent, GjEvent};
use crate::gfx::GfxState;

#[derive(Debug, Clone)]
pub enum UiEvent {
    ResetCamera,
    LoadImages,
    GenerateWithModel {
        prompt: String,
        model: Model3D,
    },
    PromptChanged(String),
    ToggleWireframe(bool),
    Log(String),

    // Jobs
    LoadScene(RecordId),
    RemoveJob(RecordId),
    ClearCompletedJobs,
}

pub struct UiContext {
    pub jobs: Vec<JobRecord>,
    pub current_job_id: Option<RecordId>,
    pub event_loop_proxy: Arc<EventLoopProxy<GjEvent>>
}

impl UiContext {
    pub fn new(event_loop_proxy: Arc<EventLoopProxy<GjEvent>>) -> Self {
        Self {
            jobs: Vec::new(),
            current_job_id: None,
            event_loop_proxy
        }
    }

    pub fn send_event(&self, event: UiEvent) {
        self.event_loop_proxy.send_event(GjEvent::Ui(event)).unwrap();
    }
}

pub struct UiState {
    pub(crate) egui_state: egui_winit::State,
    pub(crate) egui_ctx: egui::Context,
    pub(crate) egui_renderer: egui_wgpu::Renderer,

    components: Vec<Box<dyn UiComponent>>,
    pub(crate) ui_ctx: UiContext,
}

impl UiState {
    pub fn new(gfx: &GfxState, window: Arc<Window>, event_loop_proxy: Arc<EventLoopProxy<GjEvent>>) -> Self {
        let egui_ctx = egui::Context::default();

        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        let egui_renderer = egui_wgpu::Renderer::new(
            &gfx.device, gfx.config.format, egui_wgpu::RendererOptions::default());

        Self {
            egui_ctx,
            egui_state,
            egui_renderer,
            components: Vec::new(),
            ui_ctx: UiContext::new(event_loop_proxy),
        }
    }

    pub fn draw(&mut self, window: &Window) -> egui::FullOutput {
        let raw_input = self.egui_state.take_egui_input(window);

        self.egui_ctx.run(raw_input, |ctx| {
            for component in self.components.iter_mut() {
                component.show(ctx, &self.ui_ctx);
            }
        })
    }

    pub fn add_component(&mut self, component: Box<dyn UiComponent>) {
        self.components.push(component);
    }

    pub fn set_jobs(&mut self, jobs: Vec<JobRecord>) {
        self.ui_ctx.jobs = jobs;
    }

    pub async fn on_app_event(&mut self, e: AppEvent) {
        for component in self.components.iter_mut() {
            component.on_app_event(e.clone()).await;
        }
    }
}

#[async_trait]
pub trait UiComponent : Send + Sync {
    fn show(&mut self, ctx: &Context, ui_ctx: &UiContext);

    async fn on_app_event(&mut self, e: AppEvent) {}
}