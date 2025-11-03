use glam::{Mat4, Vec3};

#[derive(Clone, Debug)]
pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub distance: f32,
    pub azimuth: f32,
    pub elevation: f32,
    pub fov: f32,
    pub aspect_ratio: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 3.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            distance: 3.0,
            azimuth: 0.0,
            elevation: 0.0,
            fov: 50.0,
            aspect_ratio: 16.0 / 9.0,
            near: 0.1,
            far: 100.0,
        }
    }
}

impl Camera {
    pub fn new(target: Vec3, distance: f32) -> Self {
        let mut camera = Self::default();
        camera.target = target;
        camera.distance = distance;
        camera.update_position();
        camera
    }

    pub fn update_position(&mut self) {
        let azimuth_rad = self.azimuth.to_radians();
        let elevation_rad = self.elevation.to_radians();

        let x = self.distance * elevation_rad.cos() * azimuth_rad.sin();
        let y = self.distance * elevation_rad.sin();
        let z = self.distance * elevation_rad.cos() * azimuth_rad.cos();

        self.position = self.target + Vec3::new(x, y, z);
    }

    pub fn rotate(&mut self, delta_azimuth: f32, delta_elevation: f32) {
        self.azimuth += delta_azimuth;
        self.elevation = (self.elevation + delta_elevation).clamp(-89.0, 89.0);
        self.update_position();
    }

    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance + delta).max(0.1);
        self.update_position();
    }

    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let forward = (self.target - self.position).normalize();
        let right = forward.cross(self.up).normalize();
        let up = right.cross(forward);

        self.target += right * delta_x + up * delta_y;
        self.update_position();
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(
            self.fov.to_radians(),
            self.aspect_ratio,
            self.near,
            self.far,
        )
    }

    pub fn view_projection_matrix(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }
}