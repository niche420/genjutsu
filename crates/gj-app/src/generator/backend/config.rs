use std::env;

#[derive(Debug, Clone)]
pub struct GenBackendConfig {
    pub backend_port: u16,
    pub genjutsu_api_port: u16,
}

impl GenBackendConfig {
    pub fn load() -> anyhow::Result<Self> {
        dotenvy::from_path("crates/gj-app/.env")?;

        let backend_port: u16 = env::var("BACKEND_PORT")
            .unwrap_or_else(|_| "3000".to_string())
            .parse()
            .expect("BACKEND_PORT must be a number");

        let genjutsu_api_port: u16 = env::var("GENJUTSU_API_PORT")
            .unwrap_or_else(|_| "5000".to_string())
            .parse()
            .expect("GENJUTSU_API_PORT must be a number");

        Ok(Self {
            backend_port,
            genjutsu_api_port,
        })
    }
}