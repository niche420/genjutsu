/// Progress callback for generation
pub trait ProgressCallback: Send {
    /// Called with progress percentage (0.0 to 1.0)
    fn update(&mut self, progress: f32, message: &str);

    /// Check if generation should be cancelled
    fn should_cancel(&self) -> bool {
        false
    }
}

/// Simple progress tracker
pub struct ProgressTracker {
    pub current_step: usize,
    pub total_steps: usize,
    pub message: String,
}

impl ProgressTracker {
    pub fn new(total_steps: usize) -> Self {
        Self {
            current_step: 0,
            total_steps,
            message: String::new(),
        }
    }

    pub fn progress(&self) -> f32 {
        if self.total_steps == 0 {
            0.0
        } else {
            self.current_step as f32 / self.total_steps as f32
        }
    }

    pub fn step(&mut self, message: impl Into<String>) {
        self.current_step += 1;
        self.message = message.into();
    }
}