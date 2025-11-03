// ===========================================================================
// lgm/src/pipeline.rs
// ===========================================================================

use burn::tensor::{backend::Backend, Device};
use image::RgbaImage;
use crate::model::LGMModel;
use crate::preprocessing::{CameraInfo, preprocess_images, tensor_to_gaussian_cloud};
use gj_core::error::{Error, Result};
use gj_core::gaussian_cloud::GaussianCloud;

/// LGM Pipeline - converts multi-view images to 3D Gaussians
pub struct LGMPipeline<B: Backend> {
    model: LGMModel<B>,
    device: Device<B>,
}

impl<B: Backend> LGMPipeline<B> {
    /// Create new pipeline
    pub fn new(device: Device<B>) -> Self {
        let model = LGMModel::new(&device);
        Self { model, device }
    }

    /// Generate 3D Gaussians from 4 multi-view images
    pub fn generate(&self, images: &[RgbaImage]) -> Result<GaussianCloud> {
        if images.len() != 4 {
            return Err(Error::InvalidConfig(
                format!("Expected 4 images, got {}", images.len())
            ));
        }

        // Setup default cameras
        let cameras = CameraInfo::default_4view();

        // Preprocess
        let input = preprocess_images(&images, &cameras, &self.device)?;

        // Run model
        let output = self.model.forward(input);

        // Convert to cloud
        tensor_to_gaussian_cloud(output)
    }

    /// Generate with custom camera positions
    pub fn generate_with_cameras(
        &self,
        images: &[RgbaImage],
        cameras: &[CameraInfo],
    ) -> Result<GaussianCloud> {
        if images.len() != cameras.len() {
            return Err(Error::InvalidConfig(
                format!("Images ({}) and cameras ({}) count mismatch",
                        images.len(), cameras.len())
            ));
        }

        if images.len() != 4 {
            return Err(Error::InvalidConfig(
                "LGM requires exactly 4 views".to_string()
            ));
        }

        // Preprocess
        let input = preprocess_images(&images, cameras, &self.device)?;

        // Run model
        let output = self.model.forward(input);

        // Convert to cloud
        tensor_to_gaussian_cloud(output)
    }
}

#[cfg(test)]
mod tests {
    use burn_ndarray::NdArray;
    use super::*;
    use image::RgbaImage;

    type TestBackend = NdArray;

    fn create_test_images() -> Vec<RgbaImage> {
        (0..4).map(|_| RgbaImage::new(256, 256)).collect()
    }

    #[test]
    fn test_pipeline_creation() {
        let device = Default::default();
        let _pipeline = LGMPipeline::<TestBackend>::new(device);
    }

    #[test]
    fn test_generate() {
        let device = Default::default();
        let pipeline = LGMPipeline::<TestBackend>::new(device);

        let images = create_test_images();
        let result = pipeline.generate(&images);

        assert!(result.is_ok());
        let cloud = result.unwrap();
        assert!(cloud.count > 0);
    }

    #[test]
    fn test_wrong_image_count() {
        let device = Default::default();
        let pipeline = LGMPipeline::<TestBackend>::new(device);

        let images = vec![RgbaImage::new(256, 256)]; // Only 1 image
        let result = pipeline.generate(&images);

        assert!(result.is_err());
    }
}