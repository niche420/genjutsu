/// Camera parameters for rendering
#[derive(Clone, Debug)]
pub struct Camera {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub fov: f32,
    pub aspect_ratio: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 3.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov: 50.0,
            aspect_ratio: 16.0 / 9.0,
            near: 0.1,
            far: 100.0,
        }
    }
}

impl Camera {
    /// Get view matrix
    pub fn view_matrix(&self) -> [[f32; 4]; 4] {
        // TODO: Implement look-at matrix
        [[1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]
    }

    /// Get projection matrix
    pub fn projection_matrix(&self) -> [[f32; 4]; 4] {
        // TODO: Implement perspective projection
        [[1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]
    }
}