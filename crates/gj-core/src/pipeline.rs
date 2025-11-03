use std::time::Duration;

use crate::gaussian_cloud::GaussianCloud;
use crate::error::Result;

/// Trait for 3D generation pipelines
pub trait Pipeline3D: Send + Sync {
    /// Generate 3D Gaussians from text prompt
    fn generate(&self, prompt: &str, config: &PipelineConfig) -> Result<GaussianCloud>;

    /// Get pipeline name
    fn name(&self) -> &str;

    /// Get pipeline description
    fn description(&self) -> &str;

    /// Estimated generation time
    fn estimated_time(&self, config: &PipelineConfig) -> Duration;

    /// Check if model is loaded
    fn is_loaded(&self) -> bool;
}

/// Pipeline configuration
#[derive(Clone, Debug)]
pub enum PipelineConfig {
    LGM {
        inference_steps: usize,
        guidance_scale: f32,
        num_views: usize,
    },
    DiffSplat {
        inference_steps: usize,
        grid_resolution: usize,
    },
}

impl PipelineConfig {
    pub fn lgm_default() -> Self {
        Self::LGM {
            inference_steps: 50,
            guidance_scale: 7.5,
            num_views: 4,
        }
    }

    pub fn diffsplat_default() -> Self {
        Self::DiffSplat {
            inference_steps: 20,
            grid_resolution: 128,
        }
    }
}