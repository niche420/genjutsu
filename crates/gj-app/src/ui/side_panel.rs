use std::sync::Arc;
use async_trait::async_trait;
use chrono::Utc;
use egui::{Context, RichText, TextEdit, Color32};
use gj_core::Model3D;
use crate::events::{AppEvent, GjEvent};
use crate::ui::{UiComponent, UiContext, UiEvent};

pub struct SidePanel {
    pub selected_model: Model3D,
    pub prompt_text: String,
}

impl Default for SidePanel {
    fn default() -> Self {
        Self {
            selected_model: Model3D::ShapE,
            prompt_text: String::new(),
        }
    }
}

#[async_trait]
impl UiComponent for SidePanel {
    fn show(&mut self, ctx: &Context, ui_ctx: &UiContext) {
        egui::SidePanel::left("side_panel")
            .default_width(340.0)
            .show(ctx, |ui| {
                ui.heading("Genjutsu");
                ui.separator();

                // === Model Info ===
                ui.heading(RichText::new("⚡ Shap-E").size(16.0));
                ui.add_space(5.0);
                ui.label(
                    RichText::new("OpenAI's fast text-to-3D model (~30-60 seconds)")
                        .small()
                        .color(Color32::LIGHT_BLUE)
                );
                ui.separator();

                // === Prompt Input ===
                ui.heading(RichText::new("✨ Text Prompt").size(16.0));
                ui.add_space(5.0);

                let text_edit = TextEdit::multiline(&mut self.prompt_text)
                    .desired_width(f32::INFINITY)
                    .desired_rows(3)
                    .hint_text("e.g., a red sports car, a medieval sword, a coffee mug...");

                ui.add(text_edit);
                ui.add_space(8.0);

                let generate_button = ui.add_enabled(
                    !self.prompt_text.trim().is_empty(),
                    egui::Button::new(
                        RichText::new("🎨 Add to Queue")
                            .size(14.0)
                    )
                        .min_size(egui::vec2(ui.available_width(), 30.0))
                );

                if generate_button.clicked() {
                    ui_ctx.send_event(UiEvent::GenerateWithModel {
                        prompt: self.prompt_text.clone(),
                        model: self.selected_model,
                    });
                    self.prompt_text.clear();  // Clear after adding to queue
                }

                ui.separator();

                // === Example Prompts ===
                ui.collapsing("💡 Example Prompts", |ui| {
                    let examples = vec![
                        "a red sports car",
                        "a medieval sword",
                        "a blue crystal gem",
                        "a wooden chair",
                        "a futuristic robot",
                        "a coffee mug",
                        "a potted plant",
                        "a castle tower",
                        "a treasure chest",
                        "a flying drone",
                    ];

                    for example in examples {
                        if ui.button(example).clicked() {
                            self.prompt_text = example.to_string();
                        }
                    }
                });

                ui.separator();

                // === Tips ===
                ui.collapsing("💭 Prompt Tips", |ui| {
                    ui.label("✓ Be specific but simple");
                    ui.label("✓ Describe one object at a time");
                    ui.label("✓ Include colors and materials");
                    ui.label("✗ Avoid complex scenes");
                    ui.label("✗ Don't use abstract concepts");

                    ui.add_space(5.0);
                    ui.label(RichText::new("Examples:").strong());
                    ui.label("  Good: 'a red metal toolbox'");
                    ui.label("  Bad: 'happiness and joy'");
                });

                ui.separator();

                // === Camera Controls ===
                ui.heading("🎮 Camera Controls");
                ui.label("• Left drag: Rotate");
                ui.label("• Mouse wheel: Zoom");

                if ui.button("🔄 Reset Camera").clicked() {
                    ui_ctx.send_event(UiEvent::ResetCamera);
                }

                ui.separator();

                // === System Info ===
                ui.collapsing("ℹ️ System Info", |ui| {
                    ui.label("Model: Shap-E (OpenAI)");
                    ui.label("Renderer: Gaussian Splatting");
                    ui.label("Backend: WebGPU (wgpu)");
                    ui.label("Generation: ~30-60 seconds");
                });
            });
    }
}