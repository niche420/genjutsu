mod app;
mod state;
mod ui;

use egui_wgpu::wgpu;
use winit::event::{ElementState, Event, KeyEvent, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

fn main() {
    env_logger::init();

    println!("🚀 3D Generation Studio");
    println!("======================\n");

    let event_loop = EventLoop::new().unwrap();
    let window = event_loop.create_window(
        Window::default_attributes()
            .with_title("3D Generation Studio")
            .with_inner_size(winit::dpi::LogicalSize::new(1600, 900))
    ).unwrap();

    let mut app = pollster::block_on(app::App::new(&window));

    event_loop.run(move |event, control_flow| {
        match event {
            Event::WindowEvent { ref event, .. } => {
                if !app.input(event) {
                    match event {
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            event: KeyEvent {
                                state: ElementState::Pressed,
                                physical_key: PhysicalKey::Code(KeyCode::Escape),
                                ..
                            },
                            ..
                        } => control_flow.exit(),
                        WindowEvent::Resized(physical_size) => {
                            app.resize(*physical_size);
                        }
                        WindowEvent::RedrawRequested => {
                            app.update();
                            match app.render() {
                                Ok(_) => {}
                                Err(wgpu::SurfaceError::Lost) => app.resize(app.app_state.size),
                                Err(wgpu::SurfaceError::OutOfMemory) => control_flow.exit(),
                                Err(e) => eprintln!("{:?}", e),
                            }
                        }
                        _ => {}
                    }
                }
            }
            Event::AboutToWait => {
                app.window.request_redraw();
            }
            _ => {}
        }
    }).unwrap();
}