// crates/gj-lgm/src/text_to_image.rs

use gj_core::error::{Error, Result};
use std::path::PathBuf;
use serde::{Deserialize, Serialize};

/// Configuration for GaussianDreamer service
pub struct GaussianDreamerConfig {
    pub service_url: String,
    pub guidance_scale: f32,
    pub num_iterations: usize,
}

impl Default for GaussianDreamerConfig {
    fn default() -> Self {
        Self {
            service_url: "http://127.0.0.1:5000".to_string(),
            guidance_scale: 7.5,
            num_iterations: 500,
        }
    }
}

#[derive(Serialize)]
struct GenerateRequest {
    prompt: String,
    guidance_scale: f32,
    num_iterations: usize,
}

#[derive(Deserialize)]
struct GenerateResponse {
    status: String,
    output_path: Option<String>,
    error: Option<String>,
}

/// Generate Gaussian Splats directly from text prompt using GaussianDreamer service
///
/// This communicates with a Python service running GaussianDreamer
/// Returns the path to the generated .ply file
pub fn generate_gaussians_from_prompt(
    prompt: &str,
    config: &GaussianDreamerConfig,
) -> Result<PathBuf> {
    println!("Sending prompt to GaussianDreamer service: '{}'", prompt);

    let client = reqwest::blocking::Client::new();
    let url = format!("{}/generate", config.service_url);

    let request_body = GenerateRequest {
        prompt: prompt.to_string(),
        guidance_scale: config.guidance_scale,
        num_iterations: config.num_iterations,
    };

    let response = client
        .post(&url)
        .json(&request_body)
        .send()
        .map_err(|e| Error::InvalidConfig(format!("Failed to connect to GaussianDreamer service: {}. Make sure the Python service is running.", e)))?;

    if !response.status().is_success() {
        return Err(Error::InvalidConfig(
            format!("GaussianDreamer service returned error: {}", response.status())
        ));
    }

    let result: GenerateResponse = response
        .json()
        .map_err(|e| Error::InvalidConfig(format!("Failed to parse response: {}", e)))?;

    match result.status.as_str() {
        "success" => {
            let output_path = result.output_path
                .ok_or_else(|| Error::InvalidConfig("No output path returned".to_string()))?;

            // Convert service path to host path relative to project root
            let host_path = if output_path.starts_with("/app/outputs/") {
                // Docker: /app/outputs/file.ply -> outputs/file.ply
                PathBuf::from(output_path.replace("/app/outputs/", "outputs/"))
            } else if output_path.starts_with("../outputs/") {
                // Local: ../outputs/file.ply -> outputs/file.ply
                PathBuf::from(output_path.replace("../outputs/", "outputs/"))
            } else if output_path.starts_with("outputs/") {
                // Already correct
                PathBuf::from(output_path.clone())
            } else {
                // Unknown format - try to extract just the filename
                let filename = std::path::Path::new(&output_path)
                    .file_name()
                    .ok_or_else(|| Error::InvalidConfig("Invalid output path".to_string()))?;
                PathBuf::from("outputs").join(filename)
            };

            println!("✓ GaussianDreamer generated: {}", host_path.display());

            // Verify the file exists
            if !host_path.exists() {
                return Err(Error::InvalidConfig(
                    format!("Generated file not found at: {}. Original path was: {}",
                            host_path.display(), output_path)
                ));
            }

            Ok(host_path)
        }
        "error" => {
            let error_msg = result.error.unwrap_or_else(|| "Unknown error".to_string());
            Err(Error::InvalidConfig(format!("GaussianDreamer error: {}", error_msg)))
        }
        _ => {
            Err(Error::InvalidConfig(format!("Unexpected status: {}", result.status)))
        }
    }
}

/// Check if GaussianDreamer service is running
pub fn check_service_health(service_url: &str) -> Result<bool> {
    let client = reqwest::blocking::Client::new();
    let url = format!("{}/health", service_url);

    match client.get(&url).send() {
        Ok(response) => Ok(response.status().is_success()),
        Err(_) => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Only run when service is actually running
    fn test_service_health() {
        let config = GaussianDreamerConfig::default();
        let healthy = check_service_health(&config.service_url).unwrap();
        assert!(healthy, "GaussianDreamer service should be running");
    }
}