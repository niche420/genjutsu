#[cfg(test)]
mod tests {
    use crate::camera::Camera;
    use super::*;

    #[test]
    fn test_camera_creation() {
        let camera = Camera::default();
        assert_eq!(camera.distance, 3.0);
    }

    #[test]
    fn test_camera_rotation() {
        let mut camera = Camera::default();
        camera.rotate(45.0, 30.0);
        assert_eq!(camera.azimuth, 45.0);
        assert_eq!(camera.elevation, 30.0);
    }
}