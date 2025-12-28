// crates/gj-app/src/generator/db.rs
pub mod job;

use std::path::PathBuf;
use surrealdb::engine::local::{Db, RocksDb};
use surrealdb::Surreal;
use crate::generator::db::job::JobRecord;
use anyhow::Result;
use log::info;
use surrealdb_types::RecordId;
use crate::job::{Job, JobMetadata, JobOutputs, JobStatus};

const JOBS: &str = "jobs";

#[derive(Debug, Clone)]
pub struct JobDatabase {
    db: Surreal<Db>,
}

impl JobDatabase {
    /// Initialize SurrealDB with RocksDB backend (embedded, file-based)
    pub async fn new(db_path: PathBuf) -> Result<Self> {
        info!("Setting up job database at {}", db_path.display());

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

    /// Update job using direct record selection and content replacement
    pub async fn update_job(
        &self,
        job_id: String,
        metadata: JobMetadata,
        outputs: Option<JobOutputs>
    ) -> Result<()> {
        let record_id = (JOBS, job_id.clone());

        // Get the existing job first
        let existing: Option<JobRecord> = self.db
            .select(record_id.clone())
            .await?;

        if let Some(mut job_record) = existing {
            // Update the fields
            job_record.metadata = metadata;
            if outputs.is_some() {
                job_record.outputs = outputs;
            }

            // Create a new Job struct for update
            let updated_job = Job {
                inputs: job_record.inputs,
                metadata: job_record.metadata,
                outputs: job_record.outputs,
            };

            // Replace the entire record
            let _: Option<JobRecord> = self.db
                .update(record_id)
                .content(updated_job)
                .await?;
        }

        Ok(())
    }

    /// Update job by RecordId
    pub async fn update_job_by_id(
        &self,
        job_id: RecordId,
        metadata: JobMetadata,
        outputs: Option<JobOutputs>
    ) -> Result<()> {
        // Get the existing job first
        let existing: Option<JobRecord> = self.db
            .select(job_id.clone())
            .await?;

        if let Some(mut job_record) = existing {
            // Update the fields
            job_record.metadata = metadata;
            if outputs.is_some() {
                job_record.outputs = outputs;
            }

            // Create a new Job struct for update
            let updated_job = Job {
                inputs: job_record.inputs,
                metadata: job_record.metadata,
                outputs: job_record.outputs,
            };

            // Replace the entire record
            let _: Option<JobRecord> = self.db
                .update(job_id)
                .content(updated_job)
                .await?;
        }

        Ok(())
    }

    /// Mark job as complete with result path
    pub async fn complete_job(&self, job_id: String, ply_path: PathBuf) -> Result<()> {
        let record_id = (JOBS, job_id.clone());

        // Get the existing job
        let existing: Option<JobRecord> = self.db
            .select(record_id.clone())
            .await?;

        if let Some(mut job_record) = existing {
            job_record.metadata.status = JobStatus::COMPLETE;
            job_record.metadata.progress = 1.0;
            job_record.metadata.updated_at = chrono::Utc::now().into();
            job_record.metadata.completed_at = Some(chrono::Utc::now().into());
            job_record.outputs = Some(JobOutputs {
                ply_path: ply_path.to_string_lossy().to_string()
            });

            let updated_job = Job {
                inputs: job_record.inputs,
                metadata: job_record.metadata,
                outputs: job_record.outputs,
            };

            let _: Option<JobRecord> = self.db
                .update(record_id)
                .content(updated_job)
                .await?;
        }

        Ok(())
    }

    /// Mark job as failed
    pub async fn fail_job(&self, job_id: String, error: String) -> Result<()> {
        let record_id = (JOBS, job_id.clone());

        // Get the existing job
        let existing: Option<JobRecord> = self.db
            .select(record_id.clone())
            .await?;

        if let Some(mut job_record) = existing {
            job_record.metadata.status = JobStatus::FAILED;
            job_record.metadata.error = Some(error);
            job_record.metadata.updated_at = chrono::Utc::now().into();
            job_record.metadata.completed_at = Some(chrono::Utc::now().into());

            let updated_job = Job {
                inputs: job_record.inputs,
                metadata: job_record.metadata,
                outputs: job_record.outputs,
            };

            let _: Option<JobRecord> = self.db
                .update(record_id)
                .content(updated_job)
                .await?;
        }

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

    /// Delete a job using direct record ID
    pub async fn delete_job(&self, id: RecordId) -> Result<()> {
        let _: Option<JobRecord> = self.db.delete(id).await?;
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

    /// Verify job outputs exist on disk and return the path if found
    pub fn verify_outputs(&self, job: &JobRecord) -> Option<String> {
        // First, check if we already know the path
        if let Some(ref outputs) = job.outputs {
            let path = std::env::current_dir()
                .ok()
                .and_then(|cwd| Some(cwd.join(&outputs.ply_path)));

            if let Some(path) = path {
                if path.exists() {
                    return Some(outputs.ply_path.clone());
                }
            }
        }

        // If job.outputs is None or file doesn't exist, search for the file
        // based on the job ID and prompt
        use surrealdb_types::RecordIdKey;

        let job_id_str = match &job.id.key {
            RecordIdKey::String(id) => id.clone(),
            _ => return None,
        };

        // Search in outputs directory for files matching this job
        let outputs_dir = std::env::current_dir().ok()?.join("outputs");

        if !outputs_dir.exists() {
            return None;
        }

        // Look for files containing the job ID or matching the pattern
        if let Ok(entries) = std::fs::read_dir(outputs_dir) {
            for entry in entries.flatten() {
                if let Ok(file_type) = entry.file_type() {
                    if file_type.is_file() {
                        if let Some(file_name) = entry.file_name().to_str() {
                            // Check if filename contains job_id or matches expected pattern
                            if file_name.ends_with(".ply") &&
                                (file_name.contains(&job_id_str) ||
                                    self.matches_job_pattern(file_name, job)) {
                                // Found it! Return relative path
                                if let Ok(full_path) = entry.path().canonicalize() {
                                    if let Ok(cwd) = std::env::current_dir() {
                                        if let Ok(relative) = full_path.strip_prefix(&cwd) {
                                            return Some(relative.to_string_lossy().to_string());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Check if a filename matches the expected pattern for this job
    fn matches_job_pattern(&self, filename: &str, job: &JobRecord) -> bool {
        // Pattern: {model}_{sanitized_prompt}_{timestamp}.ply
        let model = &job.inputs.model;

        // Sanitize prompt the same way Python does
        let safe_prompt: String = job.inputs.prompt
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == ' ' || *c == '_')
            .collect::<String>()
            .trim()
            .chars()
            .take(50)
            .collect::<String>()
            .replace(' ', "_");

        // Check if filename starts with model and contains the sanitized prompt
        filename.starts_with(model) &&
            filename.contains(&safe_prompt) &&
            filename.ends_with(".ply")
    }

    /// Subscribe to job updates (real-time)
    pub async fn subscribe_to_job_updates(&self) -> Result<impl futures::Stream<Item = JobRecord>> {
        use futures::StreamExt;

        let mut response = self.db
            .query("LIVE SELECT * FROM jobs")
            .await?;

        let stream = response
            .stream::<surrealdb::Notification<JobRecord>>(0)?;

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