mod config;
mod routes;
mod state;
mod schemas;

use std::sync::Arc;
use axum::Router;
use log::info;
use tokio::net::TcpListener;
use winit::event_loop::EventLoopProxy;
use gj_core::Model3D;
use crate::error::AppError;
use crate::events::GjEvent;
use crate::generator::backend::config::GenBackendConfig;
use crate::generator::backend::routes::api_routes;
use crate::generator::backend::state::GenState;
use crate::generator::backend::schemas::{JobCreateResponse};
use crate::job::JobInputs;

pub struct GenBackend {
    config: GenBackendConfig,
    client: reqwest::Client,
}

impl GenBackend {
    pub async fn new(event_loop_proxy: Arc<EventLoopProxy<GjEvent>>) -> anyhow::Result<Self> {
        let conf = GenBackendConfig::load()?;

        let state = GenState::new(event_loop_proxy);

        let app = Router::new()
            .merge(api_routes())
            .with_state(Arc::new(state));

        let addr = std::net::SocketAddr::from(([0, 0, 0, 0], conf.backend_port));

        info!("Starting backend server on port {}", conf.backend_port);

        tokio::spawn(async move {
            let listener = TcpListener::bind(addr).await.expect("Failed to bind");
            axum::serve(listener, app).await.expect("Server failed");
        });

        // Create async reqwest client
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()?;

        Ok(Self {
            config: conf,
            client,
        })
    }

    pub async fn submit_job(&self, prompt: String, model: Model3D) -> anyhow::Result<JobCreateResponse> {
        let url = format!("http://127.0.0.1:{}/generate", self.config.genjutsu_api_port);

        let request_body = JobInputs {
            prompt,
            model: model.id().to_string(),
            guidance_scale: 15.0,
            num_inference_steps: 64,
        };

        let response = self.client
            .post(url)
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow::Error::from(AppError::BackendError(
                format!("HTTP {}: {}", status, body)
            )));
        }

        Ok(response.json().await?)
    }
}