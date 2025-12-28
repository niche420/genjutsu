use async_trait::async_trait;
use egui::{Color32, Context, RichText, Ui};
use surrealdb_types::RecordId;
use crate::generator::db::job::JobRecord;
use crate::events::AppEvent;
use crate::job::JobStatus;
use crate::state::AppState;
use crate::ui::{UiComponent, UiContext, UiEvent};

#[derive(Default)]
pub struct QueuePanel {
    show_completed: bool,
}

impl QueuePanel {
    fn show_job_card(&self, ui: &mut Ui, ui_ctx: &UiContext, job: &JobRecord) {
        egui::Frame::none()
            .fill(Color32::from_gray(30))
            .rounding(5.0)
            .inner_margin(10.0)
            .stroke(egui::Stroke::new(1.0, Color32::from_gray(60)))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    // Status icon
                    ui.label(
                        RichText::new(job.metadata.status.icon())
                            .size(24.0)
                            .color(job.metadata.status.color())
                    );

                    ui.add_space(5.0);

                    // Job details
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(&job.inputs.prompt).strong());
                            ui.label(
                                RichText::new(format!("({})", job.inputs.model))
                                    .small()
                                    .color(Color32::GRAY)
                            );
                        });

                        if let Some(message) = &job.metadata.message {
                            ui.label(
                                RichText::new(message)
                                    .small()
                                    .color(job.metadata.status.color())
                            );
                        }

                        // Time info
                        let created_date: chrono::DateTime<chrono::Utc> = job.metadata.created_at.clone().into();
                        let time_str = if let Some(completed) = &job.metadata.completed_at {
                            let completed_date: chrono::DateTime<chrono::Utc> = completed.clone().into();
                            let duration = (completed_date - created_date).num_seconds();
                            format!("Completed in {}s", duration)
                        } else {
                            let now = chrono::Utc::now();
                            format!("Elapsed: {}s", (now - created_date).num_seconds())
                        };
                        ui.label(RichText::new(time_str).small().color(Color32::GRAY));
                    });

                    // Right side - progress/actions
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        match &job.metadata.status {
                            JobStatus::GENERATING => {
                                ui.add(
                                    egui::ProgressBar::new(job.metadata.progress)
                                        .desired_width(150.0)
                                        .show_percentage()
                                        .animate(true)
                                );
                            }
                            JobStatus::COMPLETE => {
                                // Check if this is the currently loaded scene
                                let is_current = ui_ctx.current_job_id == Some(job.id.clone());

                                if is_current {
                                    ui.label(
                                        RichText::new("👁 Viewing")
                                            .color(Color32::LIGHT_BLUE)
                                    );
                                } else {
                                    if ui.button("📦 Load Scene").clicked() {
                                        ui_ctx.send_event(UiEvent::LoadScene(job.id.clone()));
                                    }
                                }
                                ui.add_space(5.0);
                                if ui.button("🗑").clicked() {
                                    ui_ctx.send_event(UiEvent::RemoveJob(job.id.clone()));
                                }
                            }
                            JobStatus::FAILED => {
                                if let Some(error) = &job.metadata.error {
                                    ui.label(
                                        RichText::new(error)
                                            .color(Color32::RED)
                                            .small()
                                    );
                                }
                                ui.add_space(5.0);
                                if ui.button("🗑").clicked() {
                                    ui_ctx.send_event(UiEvent::RemoveJob(job.id.clone()));
                                }
                            }
                            JobStatus::QUEUED => {
                                ui.label(RichText::new("Waiting...").color(Color32::GRAY));
                            }
                        }
                    });
                });
            });

        ui.add_space(5.0);
    }
}

#[async_trait]
impl UiComponent for QueuePanel {
    fn show(&mut self, ctx: &Context, ui_ctx: &UiContext) {
        egui::TopBottomPanel::bottom("queue_panel")
            .resizable(true)
            .min_height(100.0)
            .max_height(300.0)
            .default_height(150.0)
            .show(ctx, |ui| {
                // Header
                ui.horizontal(|ui| {
                    ui.heading("🎬 Generation Queue");

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        // Clear completed button
                        if ui.button("🗑 Clear Completed").clicked() {
                            ui_ctx.send_event(UiEvent::ClearCompletedJobs);
                        }

                        ui.add_space(10.0);

                        // Toggle completed visibility
                        let toggle_text = if self.show_completed {
                            "Hide Completed"
                        } else {
                            "Show Completed"
                        };
                        if ui.button(toggle_text).clicked() {
                            self.show_completed = !self.show_completed;
                        }

                        ui.add_space(10.0);

                        // Stats
                        let active = ui_ctx.jobs.iter().filter(|j| j.metadata.status.is_active()).count();
                        let completed = ui_ctx.jobs.iter().filter(|j| j.metadata.status.is_complete()).count();

                        ui.label(
                            RichText::new(format!("Active: {} | Completed: {}", active, completed))
                                .color(Color32::GRAY)
                        );
                    });
                });

                ui.separator();

                // Job list
                egui::ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .show(ui, |ui| {
                        let mut has_visible_jobs = false;

                        for job in &ui_ctx.jobs {
                            // Filter completed if hidden
                            if !self.show_completed && job.metadata.status.is_complete() {
                                continue;
                            }

                            has_visible_jobs = true;
                            self.show_job_card(ui, ui_ctx, job);
                        }

                        if !has_visible_jobs {
                            ui.centered_and_justified(|ui| {
                                ui.label(
                                    RichText::new("No jobs in queue")
                                        .color(Color32::GRAY)
                                        .size(16.0)
                                );
                            });
                        }
                    });
            });
    }
}