use std::sync::Arc;
use axum::Router;
use axum::routing::{get, post, put};
use crate::generator::backend::routes::job::update_job_progress;
use crate::generator::backend::state::GenState;

mod job;

pub fn api_routes() -> Router<Arc<GenState>> {
    Router::new()
        .route("/job/{id}/progress", post(update_job_progress))
}