use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Generation failed: {0}")]
    GenerationFailed(String),

    #[error("Invalid Gaussian cloud: {0}")]
    InvalidGaussianCloud(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Render error: {0}")]
    RenderError(String),
}