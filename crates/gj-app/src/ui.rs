use egui::{Color32, Context, RichText};
use crate::state::AppState;

pub fn draw_ui(ctx: &Context, state: &mut AppState) {
    // Top panel
    egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.heading("🎨 3D Generation Studio");
            ui.separator();
            ui.label(RichText::new(&state.status).color(Color32::LIGHT_BLUE));
        });
    });

    // Side panel
    egui::SidePanel::left("side_panel").default_width(300.0).show(ctx, |ui| {
        ui.heading("Pipeline");
        ui.separator();

        ui.label("Select 4 multi-view images:");
        if ui.button("📁 Load Images...").clicked() {
            if let Some(files) = rfd::FileDialog::new()
                .add_filter("Images", &["png", "jpg", "jpeg"])
                .pick_files()
            {
                if let Err(e) = state.load_images(files) {
                    state.status = format!("Error: {}", e);
                }
            }
        }

        ui.separator();

        if let Some(ref cloud) = state.gaussian_cloud {
            ui.heading("📊 Stats");
            ui.label(format!("Gaussians: {}", cloud.count));

            let bounds = cloud.bounds();
            ui.label(format!("Bounds: [{:.2}, {:.2}, {:.2}]",
                             bounds.size()[0], bounds.size()[1], bounds.size()[2]));

            ui.separator();

            if ui.button("💾 Export PLY").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("PLY", &["ply"])
                    .save_file()
                {
                    match cloud.to_ply() {
                        Ok(data) => {
                            if std::fs::write(&path, data).is_ok() {
                                state.status = format!("Saved to {:?}", path);
                            }
                        }
                        Err(e) => {
                            state.status = format!("Export error: {}", e);
                        }
                    }
                }
            }
        }

        ui.separator();

        ui.heading("🎮 Camera Controls");
        ui.label("• Left drag: Rotate");
        ui.label("• Mouse wheel: Zoom");
        ui.label("• Right drag: Pan (TODO)");

        if ui.button("🔄 Reset Camera").clicked() {
            state.camera = gj_splat::camera::Camera::default();
            state.camera.aspect_ratio = state.size.width as f32 / state.size.height as f32;
        }
    });

    // Central 3D view is rendered by WGPU
    egui::CentralPanel::default().show(ctx, |ui| {
        ui.centered_and_justified(|ui| {
            if state.gaussian_cloud.is_none() {
                ui.label(RichText::new("Load 4 multi-view images to generate 3D")
                    .size(24.0)
                    .color(Color32::GRAY));
            }
        });
    });
}