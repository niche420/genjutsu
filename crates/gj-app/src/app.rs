use std::sync::Arc;
use winit::{
    event::*,
    event_loop::ActiveEventLoop,
};
use winit::application::ApplicationHandler;
use winit::event_loop::{EventLoop, EventLoopProxy};
use winit::window::{WindowAttributes, WindowId};
use crate::events::{AppEvent, GjEvent};
use crate::state::AppState;

pub struct App {
    event_loop_proxy: Arc<EventLoopProxy<GjEvent>>,
    state: Option<AppState>,
    needs_redraw: bool,
}

impl App {
    pub fn new(event_loop: &mut EventLoop<GjEvent>) -> Self {
        let event_loop_proxy = Arc::new(event_loop.create_proxy());

        Self {
            event_loop_proxy,
            state: None,
            needs_redraw: false,
        }
    }
}

impl ApplicationHandler<GjEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = WindowAttributes::default()
            .with_title("Gaussian Splatting Viewer")
            .with_inner_size(winit::dpi::LogicalSize::new(1600.0, 900.0));

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        let state = pollster::block_on(AppState::new(window.clone(), self.event_loop_proxy.clone())).unwrap();
        self.state = Some(state);
        self.needs_redraw = true;
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: GjEvent) {
        if let Some(state) = &mut self.state {
            match event {
                GjEvent::Ui(e) => {
                    state.on_ui_event(e);
                    self.needs_redraw = true;
                    state.window.request_redraw();
                }
                GjEvent::App(e) => {
                    pollster::block_on(async {
                        state.ui.on_app_event(e).await;
                    });
                    self.needs_redraw = true;
                    state.window.request_redraw();
                }
                GjEvent::Gen(e) => {
                    // Handle job status updates from Python worker
                    pollster::block_on(async {
                        if let Err(e) = state.on_gen_event(e).await {
                            eprintln!("Error handling gen event: {}", e);
                        }
                    });
                    // Force immediate redraw for progress updates
                    self.needs_redraw = true;
                    state.window.request_redraw();
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else {
            return;
        };

        if state.window.id() != window_id {
            return;
        }

        // Let egui handle the event first
        let response = state.ui.egui_state.on_window_event(&state.window, &event);

        if response.repaint {
            self.needs_redraw = true;
            state.window.request_redraw();
        }

        // For camera controls, we want to handle mouse events even if egui consumes them,
        // but only if the mouse is NOT over any egui UI elements
        let handle_camera_input = match &event {
            WindowEvent::MouseInput { .. } |
            WindowEvent::CursorMoved { .. } |
            WindowEvent::MouseWheel { .. } => {
                // Check if mouse is over UI
                !state.ui.egui_ctx.is_pointer_over_area()
            }
            _ => false,
        };

        // Handle events not consumed by egui
        if !response.consumed || handle_camera_input {
            match event {
                WindowEvent::CloseRequested => {
                    event_loop.exit();
                }
                WindowEvent::Resized(physical_size) => {
                    state.resize(physical_size);
                    self.needs_redraw = true;
                }
                WindowEvent::RedrawRequested => {
                    let _ = state.render();
                    self.needs_redraw = false;
                }
                WindowEvent::CursorMoved { .. } |
                WindowEvent::MouseWheel { .. } |
                WindowEvent::MouseInput { .. } => {
                    // Mouse events should trigger redraws for smooth camera control
                    state.input(&event);
                    self.needs_redraw = true;
                    state.window.request_redraw();
                }
                _ => {
                    state.input(&event);
                }
            }
        } else {
            // Even if egui consumed the event, check if we should handle it for camera
            // Only handle camera input if the mouse is over the central panel (3D viewport)
            match event {
                WindowEvent::CursorMoved { .. } |
                WindowEvent::MouseWheel { .. } => {
                    // Always process camera input for these events
                    // The camera controller will only respond if mouse is pressed
                    state.input(&event);
                    if state.mouse_pressed {
                        self.needs_redraw = true;
                        state.window.request_redraw();
                    }
                }
                WindowEvent::MouseInput { .. } => {
                    // Track mouse state even if egui consumed the click
                    state.input(&event);
                }
                _ => {}
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Only request redraw if we actually need one
        // Remove the constant redraw requests that were causing performance issues
        if self.needs_redraw {
            if let Some(state) = &self.state {
                state.window.request_redraw();
            }
        }
    }
}