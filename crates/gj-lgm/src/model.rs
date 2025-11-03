use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::prelude::{Backend, Device, Module};
use burn::Tensor;
use burn::tensor::activation::relu;

/// LGM Model - Multi-view images to 3D Gaussians
#[derive(Module, Debug)]
pub struct LGMModel<B: Backend> {
    // Input conv
    conv_in: Conv2d<B>,

    // Output conv (14 channels for Gaussian parameters)
    conv_out: Conv2d<B>,
}

impl<B: Backend> LGMModel<B> {
    pub fn new(device: &Device<B>) -> Self {
        Self {
            conv_in: Conv2dConfig::new([9, 64], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            conv_out: Conv2dConfig::new([64, 14], [1, 1])
                .init(device),
        }
    }

    /// Forward pass: [B, 4, 9, H, W] -> [B, N, 14]
    pub fn forward(&self, images: Tensor<B, 5>) -> Tensor<B, 3> {
        let dims = images.dims();
        let (b, num_views, _channels, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);

        // Reshape: [B, 4, 9, H, W] -> [B*4, 9, H, W]
        let x = images.reshape([b * num_views, 9, h, w]);

        // Simple processing
        let x = self.conv_in.forward(x);
        let x = relu(x);
        let x = self.conv_out.forward(x);

        // Reshape to [B, N, 14] where N = 4*H*W
        let x = x.reshape([b * num_views, 14, h * w]);
        let x = x.swap_dims(1, 2); // [B*4, H*W, 14]
        let x = x.reshape([b, num_views * h * w, 14]);

        // Apply activations
        self.apply_activations(x)
    }

    pub(crate) fn apply_activations(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = x.device();
        let dims = x.dims();
        let (batch, n, _params) = (dims[0], dims[1], dims[2]);

        // Extract data using into_data()
        let data = x.into_data();
        let values: Vec<f32> = data.iter::<f32>().collect();

        // Process each Gaussian
        let mut output = vec![0.0f32; batch * n * 14];

        for b in 0..batch {
            for i in 0..n {
                let base_in = (b * n + i) * 14;
                let base_out = base_in;

                // Position [0:3] - clamp to [-1, 1]
                for j in 0..3 {
                    output[base_out + j] = values[base_in + j].clamp(-1.0, 1.0);
                }

                // Opacity [3] - sigmoid
                let opacity = values[base_in + 3];
                output[base_out + 3] = 1.0 / (1.0 + (-opacity).exp());

                // Scale [4:7] - softplus * 0.1
                for j in 4..7 {
                    let val = values[base_in + j];
                    output[base_out + j] = (1.0 + val.exp()).ln() * 0.1;
                }

                // Rotation [7:11] - normalize quaternion
                let mut quat = [0.0f32; 4];
                for j in 0..4 {
                    quat[j] = values[base_in + 7 + j];
                }
                let norm = (quat[0]*quat[0] + quat[1]*quat[1] +
                    quat[2]*quat[2] + quat[3]*quat[3]).sqrt();
                let norm = if norm > 1e-8 { norm } else { 1.0 };
                for j in 0..4 {
                    output[base_out + 7 + j] = quat[j] / norm;
                }

                // RGB [11:14] - tanh * 0.5 + 0.5
                for j in 11..14 {
                    let val = values[base_in + j];
                    output[base_out + j] = val.tanh() * 0.5 + 0.5;
                }
            }
        }

        let total_elements = batch * n * 14;
        let tensor_1d: Tensor<B, 1> = Tensor::from_floats(output.as_slice(), &device);
        tensor_1d.reshape([batch, n, 14])
    }
}