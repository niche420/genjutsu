use std::sync::Arc;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use axum::response::IntoResponse;
use crate::generator::backend::schemas::JobStatusResponse;
use crate::generator::backend::state::GenState;

pub async fn update_job_progress(
    State(state): State<Arc<GenState>>,
    Path(id): Path<String>,
    Json(resp): Json<JobStatusResponse>,
) -> impl IntoResponse {
    state.emit_job_status(id, resp);
    StatusCode::OK
}