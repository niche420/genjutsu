use log::Record;
use surrealdb_types::RecordId;
use gj_core::Model3D;
use crate::generator::db::job::JobRecord;
use crate::job::{Job, JobMetadata, JobOutputs};
use crate::ui::UiEvent;

#[derive(Debug, Clone)]
pub enum GjEvent {
    Ui(UiEvent),
    App(AppEvent),
    Gen(GenEvent),
}

#[derive(Debug, Clone)]
pub enum AppEvent {
    ImagesLoaded,
    GaussianCloudReady,
    CameraResetDone,
    Status(String),
    Progress(f32),
    Log(String),
    SceneReady,
    
    JobQueued(JobRecord),
    JobProgress {     
        job_id: String,
        progress: f32,
        message: String,
    },
    JobComplete(String),
    JobFailed {         
        job_id: String,
        error: String,
    },
}

#[derive(Debug, Clone)]
pub enum GenEvent {
    JobStatus {
        id: String,
        data: JobMetadata,
        outputs: Option<JobOutputs>,
    }
}