#[cfg(test)]
mod tests {
    use super::*;
    use burn::Tensor;
    use burn_ndarray::NdArray;
    use crate::model::LGMModel;

    type TestBackend = NdArray;

    #[test]
    fn test_lgm_model_creation() {
        let device = Default::default();
        let model = LGMModel::<TestBackend>::new(&device);
        // Should not panic
    }

    #[test]
    fn test_gaussian_activations() {
        let device = Default::default();
        let model = LGMModel::<TestBackend>::new(&device);

        // Create dummy input [1, 1000, 14]
        let x = Tensor::zeros([1, 1000, 14], &device);
        let result = model.apply_activations(x);

        assert_eq!(result.dims(), [1, 1000, 14]);
    }
}