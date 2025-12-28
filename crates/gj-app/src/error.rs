use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Error from backend: {0}")]
    BackendError(String),
}
