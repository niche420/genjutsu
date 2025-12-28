use async_trait::async_trait;
use egui::{Color32, Context, RichText};
use crate::events::AppEvent;
use crate::ui::{UiComponent, UiContext};

#[derive(Default)]
pub struct TopPanel {}

#[async_trait]
impl UiComponent for TopPanel {
    fn show(&mut self, ctx: &Context, ui_ctx: &UiContext) {
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("\"Don't play with them spoons, I ain't Lovato\" - 2slimey");
            });
        });
    }
}