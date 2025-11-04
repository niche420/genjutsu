use burn::tensor::{backend::Backend, Device, Tensor, Shape};
use image::{RgbaImage, DynamicImage, ImageBuffer, Rgba};
use gj_core::error::Error;
use gj_core::gaussian_cloud::GaussianCloud;

/// Camera information for a view
#[derive(Clone, Debug)]
pub struct CameraInfo {
    pub azimuth: f32,
    pub elevation: f32,
    pub radius: f32,
}

impl CameraInfo {
    pub fn default_4view() -> [CameraInfo; 4] {
        [
            CameraInfo { azimuth: 0.0, elevation: 0.0, radius: 2.0 },
            CameraInfo { azimuth: 90.0, elevation: 0.0, radius: 2.0 },
            CameraInfo { azimuth: 180.0, elevation: 0.0, radius: 2.0 },
            CameraInfo { azimuth: 270.0, elevation: 0.0, radius: 2.0 },
        ]
    }

    pub fn to_features(&self) -> [f32; 6] {
        let az = self.azimuth.to_radians();
        let el = self.elevation.to_radians();
        [
            az.sin(),
            az.cos(),
            el.sin(),
            el.cos(),
            self.radius / 5.0,
            (self.radius / 5.0).powi(2),
        ]
    }
}

/// Preprocess images to tensor
pub fn preprocess_images<B: Backend>(
    images: &[RgbaImage],
    cameras: &[CameraInfo],
    device: &Device<B>,
) -> gj_core::error::Result<Tensor<B, 5>> {
    if images.len() != 4 || cameras.len() != 4 {
        return Err(Error::InvalidConfig(
            "Expected 4 images and 4 cameras".to_string()
        ));
    }

    let mut all_data = Vec::new();

    for (img, camera) in images.iter().zip(cameras.iter()) {
        // Resize to 256x256
        let img = image::imageops::resize(
            img,
            256,
            256,
            image::imageops::FilterType::Lanczos3
        );

        // Convert to tensor data [9, 256, 256]
        let mut view_data = vec![0.0f32; 9 * 256 * 256];

        // RGB channels
        for y in 0..256_usize {
            for x in 0..256_usize {
                let pixel = img.get_pixel(x as u32, y as u32);
                let idx = y * 256 + x;
                view_data[idx] = pixel[0] as f32 / 255.0;                    // R
                view_data[256*256 + idx] = pixel[1] as f32 / 255.0;          // G
                view_data[2*256*256 + idx] = pixel[2] as f32 / 255.0;        // B
            }
        }

        // Camera features (constant across spatial dimensions)
        let features = camera.to_features();
        for ch in 0..6 {
            for i in 0..(256*256) {
                view_data[(3 + ch) * 256 * 256 + i] = features[ch];
            }
        }

        all_data.extend_from_slice(&view_data);
    }

    // Create tensor [1, 4, 9, 256, 256]
    let tensor = Tensor::<B, 1>::from_floats(all_data.as_slice(), device)
        .reshape([1, 4, 9, 256, 256]);

    Ok(tensor)
}


/// Helper: Create dummy multi-view images for testing
#[cfg(test)]
pub fn create_dummy_images() -> Vec<RgbaImage> {
    (0..4)
        .map(|i| {
            let mut img = RgbaImage::new(256, 256);
            // Fill with different colors for each view
            let color = match i {
                0 => Rgba([255, 0, 0, 255]),     // Red
                1 => Rgba([0, 255, 0, 255]),     // Green
                2 => Rgba([0, 0, 255, 255]),     // Blue
                3 => Rgba([255, 255, 0, 255]),   // Yellow
                _ => Rgba([0, 0, 0, 255]),
            };

            for pixel in img.pixels_mut() {
                *pixel = color;
            }

            img
        })
        .collect()
}

/// Convert tensor to GaussianCloud
pub fn tensor_to_gaussian_cloud<B: Backend>(tensor: Tensor<B, 3>) -> gj_core::error::Result<GaussianCloud> {
    let dims = tensor.dims();
    let (_batch, n, _params) = (dims[0], dims[1], dims[2]);

    let data = tensor.into_data();
    let values: Vec<f32> = data.iter::<f32>().collect();

    let mut cloud = GaussianCloud::with_capacity(n);

    for i in 0..n {
        let base = i * 14;

        let position = [
            values[base],
            values[base + 1],
            values[base + 2],
        ];

        let opacity = values[base + 3];

        // Only add if visible
        if opacity > 0.01 {
            let scale = [
                values[base + 4],
                values[base + 5],
                values[base + 6],
            ];

            let rotation = [
                values[base + 7],
                values[base + 8],
                values[base + 9],
                values[base + 10],
            ];

            let color = [
                values[base + 11],
                values[base + 12],
                values[base + 13],
            ];

            cloud.add_gaussian(position, scale, rotation, color, opacity);
        }
    }

    Ok(cloud)
}

#[cfg(test)]
mod tests {
    use burn_ndarray::NdArray;
    use super::*;
    use gj_core::gaussian_cloud::GaussianCloud;
    use crate::model::LGMModel;

    type TestBackend = NdArray;

    /// Create test Gaussian cloud
    pub fn create_test_cloud() -> GaussianCloud {
        let mut cloud = GaussianCloud::new();

        for i in 0..100 {
            let t = i as f32 / 100.0;
            let theta = t * std::f32::consts::TAU;
            let r = 0.5 + 0.3 * (t * 5.0).sin();

            let x = r * theta.cos();
            let y = (t - 0.5) * 2.0;
            let z = r * theta.sin();

            cloud.add_gaussian(
                [x, y, z],
                [0.1, 0.1, 0.1],
                [1.0, 0.0, 0.0, 0.0],
                [t, 1.0 - t, 0.5],
                0.8,
            );
        }

        cloud
    }

    #[test]
    fn test_model_creation() {
        let device = Default::default();
        let _model = LGMModel::<TestBackend>::new(&device);
    }

    #[test]
    fn test_camera_features() {
        let camera = CameraInfo {
            azimuth: 0.0,
            elevation: 0.0,
            radius: 2.0,
        };
        let features = camera.to_features();
        assert_eq!(features.len(), 6);
    }

    #[test]
    fn test_create_test_cloud() {
        let cloud = create_test_cloud();
        assert!(cloud.count > 0);
    }
}