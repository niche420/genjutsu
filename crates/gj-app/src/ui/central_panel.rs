use async_trait::async_trait;
use egui::{Color32, Context, RichText};
use crate::events::AppEvent;
use crate::ui::{UiComponent, UiContext};

#[derive(Default)]
pub struct CentralPanel {}

#[async_trait]
impl UiComponent for CentralPanel {
    fn show(&mut self, ctx: &Context, ui_ctx: &UiContext) {
        egui::CentralPanel::default()
            .frame(egui::Frame::default().fill(Color32::TRANSPARENT))
            .show(ctx, |ui| {
                // Always allocate space to prevent zero-size viewport issues
                ui.allocate_space(ui.available_size());

                // Show instructions centered
                ui.vertical_centered(|ui| {
                    ui.label("Viewport - 3D scene renders under the UI.");
                    ui.label("When no cloud is loaded, this area shows instructions.");
                });
            });
    }
}