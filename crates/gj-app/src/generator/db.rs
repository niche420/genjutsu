pub mod job;

use std::path::PathBuf;
use surrealdb::engine::local::{Db, RocksDb};
use surrealdb::{Notification, Surreal};
use crate::generator::db::job::{JobRecord};
use anyhow::Result;
use log::info;
use surrealdb_types::RecordId;
use thiserror::__private17::AsDisplay;
use crate::job::{Job, JobMetadata, JobOutputs, JobStatus};

const JOBS: &str = "jobs";

#[derive(Debug, Clone)]
pub struct JobDatabase {
    db: Surreal<Db>,
}

impl JobDatabase {
    /// Initialize SurrealDB with RocksDB backend (embedded, file-based)
    pub async fn new(db_path: PathBuf) -> Result<Self> {
        info!("Setting up job database at {}", &db_path.as_display());

        // Create database directory
        std::fs::create_dir_all(&db_path)?;

        let db = Surreal::new::<RocksDb>(db_path).await?;
        // Use namespace and database
        db.use_ns("genjutsu").use_db("jobs").await?;

        Ok(Self { db })
    }

    /// Insert a new job
    pub async fn insert_job(&self, id: String, job: Job) -> Result<Option<JobRecord>> {
        let record: Option<JobRecord> = self.db
            .create((JOBS, id))
            .content(job)
            .await?;

        Ok(record)
    }

    pub async fn update_job(
        &self,
        job_id: String,
        metadata: JobMetadata,
        outputs: Option<JobOutputs>
    ) -> Result<()> {
        self.db
            .query("UPDATE $id MERGE $data")
            .bind(("id", (JOBS, job_id)))
            .bind(("data", serde_json::json!({
                "metadata": metadata,
                "outputs": outputs,
            })))
            .await?;

        Ok(())
    }

    /// Update job by RecordId (used during startup cleanup)
    pub async fn update_job_by_id(
        &self,
        job_id: RecordId,
        metadata: JobMetadata,
        outputs: Option<JobOutputs>
    ) -> Result<()> {
        self.db
            .query("UPDATE $id MERGE $data")
            .bind(("id", job_id))
            .bind(("data", serde_json::json!({
                "metadata": metadata,
                "outputs": outputs,
            })))
            .await?;

        Ok(())
    }

    /// Mark job as complete with result path
    pub async fn complete_job(&self, job_id: String, ply_path: PathBuf) -> Result<()> {
        let _: Option<JobRecord> = self.db
            .update((JOBS, job_id))
            .merge(serde_json::json!({
                "metadata.status": JobStatus::COMPLETE,
                "metadata.progress": 1.0,
                "metadata.updated_at": chrono::Utc::now(),
                "metadata.completed_at": chrono::Utc::now(),
                "outputs": {
                    "ply_path": ply_path.to_string_lossy().to_string()
                }
            }))
            .await?;

        Ok(())
    }

    /// Mark job as failed
    pub async fn fail_job(&self, job_id: String, error: String) -> Result<()> {
        let _: Option<JobRecord> = self.db
            .update((JOBS, job_id))
            .merge(serde_json::json!({
                "metadata.status": JobStatus::FAILED,
                "metadata.error": error,
                "metadata.updated_at": chrono::Utc::now(),
                "metadata.completed_at": chrono::Utc::now(),
            }))
            .await?;

        Ok(())
    }

    /// Get job by ID
    pub async fn get_job(&self, job_id: String) -> Result<Option<JobRecord>> {
        let record: Option<JobRecord> = self.db
            .select((JOBS, job_id))
            .await?;

        Ok(record)
    }

    /// Get all jobs, ordered by created_at DESC
    pub async fn get_all_jobs(&self) -> Result<Vec<JobRecord>> {
        let jobs: Vec<JobRecord> = match self.db.select(JOBS).await {
            Ok(jobs) => jobs,
            Err(_) => {
                // Table might not exist yet, return empty vec
                return Ok(Vec::new());
            }
        };

        // Sort by created_at descending
        let mut jobs = jobs;
        jobs.sort_by(|a, b| {
            let time_a: chrono::DateTime<chrono::Utc> = a.metadata.created_at.clone().into();
            let time_b: chrono::DateTime<chrono::Utc> = b.metadata.created_at.clone().into();
            time_b.cmp(&time_a)
        });

        Ok(jobs)
    }

    /// Get active jobs only
    pub async fn get_active_jobs(&self) -> Result<Vec<JobRecord>> {
        let jobs = self.get_all_jobs().await?;
        Ok(jobs.into_iter()
            .filter(|j| j.metadata.status.is_active())
            .collect())
    }

    /// Get completed jobs only
    pub async fn get_completed_jobs(&self) -> Result<Vec<JobRecord>> {
        let jobs = self.get_all_jobs().await?;
        Ok(jobs.into_iter()
            .filter(|j| j.metadata.status.is_complete())
            .collect())
    }

    /// Delete a job
    pub async fn delete_job(&self, id: RecordId) -> Result<()> {
        let _: Option<JobRecord> = self.db
            .delete(id)
            .await?;

        Ok(())
    }

    /// Clear all completed jobs
    pub async fn clear_completed(&self) -> Result<()> {
        let jobs = self.get_completed_jobs().await?;

        for job in jobs {
            let _: Option<JobRecord> = self.db.delete(job.id).await?;
        }

        Ok(())
    }

    /// Subscribe to job updates (real-time)
    pub async fn subscribe_to_job_updates(&self) -> Result<impl futures::Stream<Item = JobRecord>> {
        // This is where SurrealDB shines - LIVE queries!
        use futures::StreamExt;

        let mut response = self.db
            .query("LIVE SELECT * FROM jobs")
            .await?;

        let stream = response
            .stream::<Notification<JobRecord>>(0)?;

        // Convert Notification<JobRecord> -> JobRecord
        let mapped = stream.filter_map(|notif| async move {
            match notif {
                Ok(n) => Some(n.data),
                Err(_) => None,
            }
        });

        Ok(mapped)
    }
}