use serde_json::Value;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use crate::job::{Job, JobInputs, JobMetadata, JobOutputs, JobStatus};
use chrono::{DateTime, Utc, Duration};
use surrealdb_types::{RecordId, SurrealValue};

pub use surrealdb::types::Datetime as SurrealDatetime;

#[derive(Debug, Deserialize, SurrealValue, Clone)]
pub struct JobRecord {
    pub id: RecordId,
    pub inputs: JobInputs,
    pub metadata: JobMetadata,
    pub outputs: Option<JobOutputs>,
}
