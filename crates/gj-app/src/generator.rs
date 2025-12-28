// Update for crates/gj-app/src/generator.rs
use std::sync::Arc;
use chrono::Utc;
use surrealdb_types::RecordId;
use winit::event_loop::EventLoopProxy;
use gj_core::Model3D;
use db::job::JobRecord;
use crate::events::{AppEvent, GjEvent};
use crate::generator::backend::GenBackend;
use crate::generator::db::job::SurrealDatetime;
use crate::generator::db::JobDatabase;
use crate::job::{Job, JobMetadata, JobInputs, JobOutputs, JobStatus};

pub mod backend;
pub mod db;

pub struct Generator {
    backend: GenBackend,
    pub db: JobDatabase,  // Made public for verify_outputs access
    event_loop_proxy: Arc<EventLoopProxy<GjEvent>>,
}

impl Generator {
    pub async fn new(event_loop_proxy: Arc<EventLoopProxy<GjEvent>>) -> anyhow::Result<Self> {
        let backend = GenBackend::new(event_loop_proxy.clone()).await?;

        let db_path = std::env::current_dir()?.join("outputs/db");
        let db = JobDatabase::new(db_path).await?;

        Ok(Self {
            backend,
            db,
            event_loop_proxy
        })
    }

    pub async fn submit_job(&mut self, prompt: String, model: Model3D) -> anyhow::Result<()> {
        let resp = self.backend.submit_job(prompt.clone(), model).await?;
        let job = Job {
            inputs: JobInputs {
                prompt,
                model: model.id().to_string(),
                guidance_scale: 15.0,
                num_inference_steps: 64,
            },
            metadata: JobMetadata {
                status: JobStatus::QUEUED,
                progress: 0f32,
                message: resp.message,
                error: None,
                created_at: SurrealDatetime::from(Utc::now()),
                updated_at: SurrealDatetime::from(Utc::now()),
                completed_at: None,
            },
            outputs: None
        };
        let job_record = self.db.insert_job(resp.id, job).await?;

        // Emit JobQueued event so UI updates
        if let Some(record) = job_record {
            self.event_loop_proxy.send_event(GjEvent::App(AppEvent::JobQueued(record)))?;
        }

        Ok(())
    }

    pub async fn remove_job(&mut self, id: RecordId) -> anyhow::Result<()> {
        self.db.delete_job(id).await
    }

    pub async fn get_jobs(&self) -> anyhow::Result<Vec<JobRecord>> {
        self.db.get_all_jobs().await
    }

    /// Update job status when we receive callbacks from Python
    pub async fn update_job_status(
        &mut self,
        job_id: String,
        metadata: JobMetadata,
        outputs: Option<JobOutputs>
    ) -> anyhow::Result<()> {
        self.db.update_job(job_id, metadata, outputs).await
    }

    pub async fn update_job_status_by_id(
        &mut self,
        record_id: RecordId,
        metadata: JobMetadata,
        outputs: Option<JobOutputs>
    ) -> anyhow::Result<()> {
        self.db.update_job_by_id(record_id, metadata, outputs).await
    }

    /// Clear all completed jobs
    pub async fn clear_completed(&mut self) -> anyhow::Result<()> {
        self.db.clear_completed().await
    }
}