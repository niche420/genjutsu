use std::sync::mpsc::{channel, Sender, Receiver};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use image::RgbaImage;
use gj_core::gaussian_cloud::GaussianCloud;
use gj_core::Model3D;
use serde::{Deserialize, Serialize};

pub enum WorkerCommand {
    GenerateFromImages(Vec<RgbaImage>),
    GenerateFromPrompt { prompt: String, model: Model3D },
    CheckStatus(String), // Check job status by ID
    Shutdown,
}

pub enum WorkerResponse {
    Success(GaussianCloud),
    Error(String),
    Progress(f32),
    Status(String),
    JobSubmitted(String), // Job ID
}

pub struct InferenceWorker {
    pub(crate) command_tx: Sender<WorkerCommand>,
    pub(crate) response_rx: Receiver<WorkerResponse>,
    thread_handle: Option<JoinHandle<()>>,
}

impl InferenceWorker {
    pub fn new() -> Self {
        let (cmd_tx, cmd_rx) = channel::<WorkerCommand>();
        let (resp_tx, resp_rx) = channel::<WorkerResponse>();

        let thread_handle = thread::spawn(move || {
            // Worker loop
            loop {
                match cmd_rx.recv() {
                    Ok(WorkerCommand::GenerateFromImages(images)) => {
                        let _ = resp_tx.send(WorkerResponse::Status("Processing images...".into()));
                        let _ = resp_tx.send(WorkerResponse::Error(
                            "Image-based generation not yet implemented with Shap-E. Use text prompts instead.".into()
                        ));
                    }

                    Ok(WorkerCommand::GenerateFromPrompt { prompt, model }) => {
                        let _ = resp_tx.send(WorkerResponse::Status(
                            format!("Submitting job to {} service...", model.name())
                        ));

                        // Submit job and get job ID
                        match submit_generation_job(&prompt, model) {
                            Ok(job_id) => {
                                let _ = resp_tx.send(WorkerResponse::JobSubmitted(job_id.clone()));
                                let _ = resp_tx.send(WorkerResponse::Status(
                                    format!("Job submitted (ID: {})", job_id)
                                ));

                                // Poll for status
                                if let Err(e) = poll_job_status(&job_id, &resp_tx) {
                                    let _ = resp_tx.send(WorkerResponse::Error(
                                        format!("Failed to poll job: {}", e)
                                    ));
                                }
                            }
                            Err(e) => {
                                let _ = resp_tx.send(WorkerResponse::Error(
                                    format!("Failed to submit job: {}", e)
                                ));
                            }
                        }
                    }

                    Ok(WorkerCommand::CheckStatus(job_id)) => {
                        if let Err(e) = poll_job_status(&job_id, &resp_tx) {
                            let _ = resp_tx.send(WorkerResponse::Error(
                                format!("Failed to check status: {}", e)
                            ));
                        }
                    }

                    Ok(WorkerCommand::Shutdown) => {
                        break;
                    }

                    Err(_) => {
                        break;
                    }
                }
            }
        });

        Self {
            command_tx: cmd_tx,
            response_rx: resp_rx,
            thread_handle: Some(thread_handle),
        }
    }

    pub fn send_images(&self, images: Vec<RgbaImage>) -> Result<(), String> {
        self.command_tx
            .send(WorkerCommand::GenerateFromImages(images))
            .map_err(|e| format!("Failed to send images to worker: {}", e))
    }

    pub fn send_prompt(&self, prompt: String, model: Model3D) -> Result<(), String> {
        self.command_tx
            .send(WorkerCommand::GenerateFromPrompt { prompt, model })
            .map_err(|e| format!("Failed to send prompt to worker: {}", e))
    }

    pub fn try_recv_response(&self) -> Option<WorkerResponse> {
        self.response_rx.try_recv().ok()
    }

    pub fn shutdown(&mut self) {
        let _ = self.command_tx.send(WorkerCommand::Shutdown);
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for InferenceWorker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ============================================================================
// API Client
// ============================================================================

#[derive(Serialize)]
struct GenerateRequest {
    prompt: String,
    model: String,
    guidance_scale: f32,
    num_inference_steps: usize,
}

#[derive(Deserialize)]
struct JobResponse {
    job_id: String,
    status: String,
}

#[derive(Deserialize)]
struct JobStatusResponse {
    job_id: String,
    status: String,
    progress: Option<f32>,
    message: Option<String>,
    result: Option<JobResult>,
    error: Option<String>,
}

#[derive(Deserialize)]
struct JobResult {
    output_path: String,
    model: String,
    prompt: String,
}

/// Submit generation job and return job ID
fn submit_generation_job(prompt: &str, model: Model3D) -> Result<String, String> {
    let client = reqwest::blocking::Client::new();
    let url = "http://127.0.0.1:5000/generate";

    let request_body = GenerateRequest {
        prompt: prompt.to_string(),
        model: model.id().to_string(),
        guidance_scale: 15.0,
        num_inference_steps: 64,
    };

    let response = client
        .post(url)
        .json(&request_body)
        .send()
        .map_err(|e| format!("Failed to connect: {}. Make sure FastAPI service is running (cd python && docker-compose up)", e))?;

    if !response.status().is_success() {
        return Err(format!("Service returned error: {}", response.status()));
    }

    let result: JobResponse = response
        .json()
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    Ok(result.job_id)
}

/// Poll job status until complete or failed
fn poll_job_status(job_id: &str, resp_tx: &Sender<WorkerResponse>) -> Result<(), String> {
    let client = reqwest::blocking::Client::new();
    let url = format!("http://127.0.0.1:5000/status/{}", job_id);

    let mut last_progress = 0.0;

    loop {
        thread::sleep(Duration::from_secs(2)); // Poll every 2 seconds

        let response = client
            .get(&url)
            .send()
            .map_err(|e| format!("Failed to check status: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("Status check failed: {}", response.status()));
        }

        let status: JobStatusResponse = response
            .json()
            .map_err(|e| format!("Failed to parse status: {}", e))?;

        // Update progress if changed
        if let Some(progress) = status.progress {
            if progress != last_progress {
                let _ = resp_tx.send(WorkerResponse::Progress(progress));
                last_progress = progress;
            }
        }

        // Update status message
        if let Some(ref message) = status.message {
            let _ = resp_tx.send(WorkerResponse::Status(message.clone()));
        }

        match status.status.as_str() {
            "SUCCESS" => {
                if let Some(result) = status.result {
                    let _ = resp_tx.send(WorkerResponse::Status(
                        "Loading generated Gaussians...".into()
                    ));

                    // Load the PLY file
                    match gj_core::gaussian_cloud::GaussianCloud::from_ply(&result.output_path) {
                        Ok(cloud) => {
                            let _ = resp_tx.send(WorkerResponse::Status(
                                format!("Loaded {} Gaussians", cloud.count)
                            ));
                            let _ = resp_tx.send(WorkerResponse::Success(cloud));
                            return Ok(());
                        }
                        Err(e) => {
                            return Err(format!("Failed to load .ply: {}", e));
                        }
                    }
                } else {
                    return Err("Job succeeded but no result path returned".into());
                }
            }

            "FAILURE" => {
                let error_msg = status.error.unwrap_or_else(|| "Unknown error".into());
                let _ = resp_tx.send(WorkerResponse::Error(error_msg.clone()));
                return Err(error_msg);
            }

            "PENDING" | "STARTED" | "RETRY" => {
                // Continue polling
                continue;
            }

            _ => {
                // Unknown status, continue polling
                continue;
            }
        }
    }
}