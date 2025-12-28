use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use crate::job::{JobMetadata, JobOutputs};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JobCreateResponse {
    pub id: String,
    pub status: String,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JobStatusResponse {
    pub id: String,
    pub data: JobMetadata,
    pub outputs: Option<JobOutputs>
}