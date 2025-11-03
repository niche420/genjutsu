#[cfg(test)]
mod tests {
    use crate::gaussian_cloud::GaussianCloud;
    use crate::pipeline::PipelineConfig;
    use super::*;

    #[test]
    fn test_gaussian_cloud_creation() {
        let mut cloud = GaussianCloud::new();

        cloud.add_gaussian(
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            1.0,
        );

        assert_eq!(cloud.count, 1);
        assert!(cloud.validate().is_ok());
    }

    #[test]
    fn test_bounding_box() {
        let mut cloud = GaussianCloud::new();

        cloud.add_gaussian([1.0, 2.0, 3.0], [0.1; 3], [1.0, 0.0, 0.0, 0.0], [1.0; 3], 1.0);
        cloud.add_gaussian([-1.0, -2.0, -3.0], [0.1; 3], [1.0, 0.0, 0.0, 0.0], [1.0; 3], 1.0);

        let bounds = cloud.bounds();
        assert_eq!(bounds.min, [-1.0, -2.0, -3.0]);
        assert_eq!(bounds.max, [1.0, 2.0, 3.0]);
        assert_eq!(bounds.center(), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ply_export() {
        let mut cloud = GaussianCloud::new();
        cloud.add_gaussian([0.0; 3], [1.0; 3], [1.0, 0.0, 0.0, 0.0], [1.0; 3], 1.0);

        let ply = cloud.to_ply().unwrap();
        assert!(!ply.is_empty());
        assert!(ply.starts_with(b"ply\n"));
    }

    #[test]
    fn test_pipeline_config() {
        let config = PipelineConfig::lgm_default();

        match config {
            PipelineConfig::LGM { inference_steps, .. } => {
                assert_eq!(inference_steps, 50);
            }
            _ => panic!("Wrong config type"),
        }
    }
}