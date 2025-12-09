use crate::bounding_box::BoundingBox;
use crate::error::{Error, Result};

#[derive(Clone, Debug)]
pub struct GaussianCloud {
    /// Number of Gaussians
    pub count: usize,

    /// Positions [N, 3] - (x, y, z) in world space
    pub positions: Vec<[f32; 3]>,

    /// Scales [N, 3] - (sx, sy, sz) for each axis
    pub scales: Vec<[f32; 3]>,

    /// Rotations [N, 4] - Quaternions (w, x, y, z)
    pub rotations: Vec<[f32; 4]>,

    /// Colors [N, 3] - RGB in [0, 1]
    pub colors: Vec<[f32; 3]>,

    /// Opacity [N] - Alpha in [0, 1]
    pub opacity: Vec<f32>,

    /// Optional spherical harmonics coefficients for view-dependent color
    pub sh_coefficients: Option<Vec<Vec<f32>>>,
}

impl GaussianCloud {
    /// Create new empty cloud
    pub fn new() -> Self {
        Self {
            count: 0,
            positions: Vec::new(),
            scales: Vec::new(),
            rotations: Vec::new(),
            colors: Vec::new(),
            opacity: Vec::new(),
            sh_coefficients: None,
        }
    }

    /// Create cloud with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            count: 0,
            positions: Vec::with_capacity(capacity),
            scales: Vec::with_capacity(capacity),
            rotations: Vec::with_capacity(capacity),
            colors: Vec::with_capacity(capacity),
            opacity: Vec::with_capacity(capacity),
            sh_coefficients: None,
        }
    }

    /// Add a single Gaussian
    pub fn add_gaussian(
        &mut self,
        position: [f32; 3],
        scale: [f32; 3],
        rotation: [f32; 4],
        color: [f32; 3],
        opacity: f32,
    ) {
        self.positions.push(position);
        self.scales.push(scale);
        self.rotations.push(rotation);
        self.colors.push(color);
        self.opacity.push(opacity);
        self.count += 1;
    }

    /// Get bounding box of all Gaussians
    pub fn bounds(&self) -> BoundingBox {
        if self.count == 0 {
            return BoundingBox::default();
        }

        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];

        for pos in &self.positions {
            for i in 0..3 {
                min[i] = min[i].min(pos[i]);
                max[i] = max[i].max(pos[i]);
            }
        }

        BoundingBox { min, max }
    }

    /// Load GaussianCloud from .ply file
    pub fn from_ply<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        use std::io::Read;
        use std::fs::File;

        let mut file = File::open(path)?;
        let mut contents = Vec::new();
        file.read_to_end(&mut contents)?;

        // Parse PLY header
        let header_end = contents.windows(10)
            .position(|w| w == b"end_header")
            .ok_or_else(|| Error::InvalidGaussianCloud("No end_header found".to_string()))?;

        let header = String::from_utf8_lossy(&contents[..header_end]);

        // Extract vertex count
        let vertex_count = header.lines()
            .find(|line| line.starts_with("element vertex"))
            .and_then(|line| line.split_whitespace().last())
            .and_then(|s| s.parse::<usize>().ok())
            .ok_or_else(|| Error::InvalidGaussianCloud("No vertex count found".to_string()))?;

        // Binary data starts after "end_header\n"
        let data_start = header_end + 10 + 1;
        let data = &contents[data_start..];

        let mut cloud = Self::with_capacity(vertex_count);

        for i in 0..vertex_count {
            let offset = i * 59;
            if offset + 59 > data.len() {
                break;
            }

            let vertex_data = &data[offset..offset + 59];

            // Position (bytes 0-11)
            let position = [
                f32::from_le_bytes([vertex_data[0], vertex_data[1], vertex_data[2], vertex_data[3]]),
                f32::from_le_bytes([vertex_data[4], vertex_data[5], vertex_data[6], vertex_data[7]]),
                f32::from_le_bytes([vertex_data[8], vertex_data[9], vertex_data[10], vertex_data[11]]),
            ];

            // Skip normals (bytes 12-23) - 3 floats

            // Color (bytes 24-26) - 3 unsigned bytes
            let color = [
                vertex_data[24] as f32 / 255.0,
                vertex_data[25] as f32 / 255.0,
                vertex_data[26] as f32 / 255.0,
            ];

            // Opacity (bytes 27-30)
            let opacity = f32::from_le_bytes([vertex_data[27], vertex_data[28], vertex_data[29], vertex_data[30]]);

            // Scale (bytes 31-42)
            let scale = [
                f32::from_le_bytes([vertex_data[31], vertex_data[32], vertex_data[33], vertex_data[34]]),
                f32::from_le_bytes([vertex_data[35], vertex_data[36], vertex_data[37], vertex_data[38]]),
                f32::from_le_bytes([vertex_data[39], vertex_data[40], vertex_data[41], vertex_data[42]]),
            ];

            // Rotation (bytes 43-58)
            let rotation = [
                f32::from_le_bytes([vertex_data[43], vertex_data[44], vertex_data[45], vertex_data[46]]),
                f32::from_le_bytes([vertex_data[47], vertex_data[48], vertex_data[49], vertex_data[50]]),
                f32::from_le_bytes([vertex_data[51], vertex_data[52], vertex_data[53], vertex_data[54]]),
                f32::from_le_bytes([vertex_data[55], vertex_data[56], vertex_data[57], vertex_data[58]]),
            ];

            cloud.add_gaussian(position, scale, rotation, color, opacity);
        }

        Ok(cloud)
    }

    /// Export to PLY format (standard point cloud format)
    pub fn to_ply(&self) -> Result<Vec<u8>> {
        use std::io::Write;

        let mut buffer = Vec::new();

        // PLY header
        writeln!(buffer, "ply")?;
        writeln!(buffer, "format binary_little_endian 1.0")?;
        writeln!(buffer, "element vertex {}", self.count)?;
        writeln!(buffer, "property float x")?;
        writeln!(buffer, "property float y")?;
        writeln!(buffer, "property float z")?;
        writeln!(buffer, "property float nx")?;
        writeln!(buffer, "property float ny")?;
        writeln!(buffer, "property float nz")?;
        writeln!(buffer, "property uchar red")?;
        writeln!(buffer, "property uchar green")?;
        writeln!(buffer, "property uchar blue")?;
        writeln!(buffer, "property float opacity")?;
        writeln!(buffer, "property float scale_0")?;
        writeln!(buffer, "property float scale_1")?;
        writeln!(buffer, "property float scale_2")?;
        writeln!(buffer, "property float rot_0")?;
        writeln!(buffer, "property float rot_1")?;
        writeln!(buffer, "property float rot_2")?;
        writeln!(buffer, "property float rot_3")?;
        writeln!(buffer, "end_header")?;

        // Binary data
        for i in 0..self.count {
            // Position
            buffer.write_all(&self.positions[i][0].to_le_bytes())?;
            buffer.write_all(&self.positions[i][1].to_le_bytes())?;
            buffer.write_all(&self.positions[i][2].to_le_bytes())?;

            // Normal (placeholder)
            buffer.write_all(&0.0f32.to_le_bytes())?;
            buffer.write_all(&0.0f32.to_le_bytes())?;
            buffer.write_all(&0.0f32.to_le_bytes())?;

            // Color (convert to u8)
            buffer.push((self.colors[i][0] * 255.0) as u8);
            buffer.push((self.colors[i][1] * 255.0) as u8);
            buffer.push((self.colors[i][2] * 255.0) as u8);

            // Opacity
            buffer.write_all(&self.opacity[i].to_le_bytes())?;

            // Scale
            buffer.write_all(&self.scales[i][0].to_le_bytes())?;
            buffer.write_all(&self.scales[i][1].to_le_bytes())?;
            buffer.write_all(&self.scales[i][2].to_le_bytes())?;

            // Rotation
            buffer.write_all(&self.rotations[i][0].to_le_bytes())?;
            buffer.write_all(&self.rotations[i][1].to_le_bytes())?;
            buffer.write_all(&self.rotations[i][2].to_le_bytes())?;
            buffer.write_all(&self.rotations[i][3].to_le_bytes())?;
        }

        Ok(buffer)
    }

    /// Validate that all arrays have consistent length
    pub fn validate(&self) -> Result<()> {
        if self.positions.len() != self.count ||
            self.scales.len() != self.count ||
            self.rotations.len() != self.count ||
            self.colors.len() != self.count ||
            self.opacity.len() != self.count {
            return Err(Error::InvalidGaussianCloud(
                "Inconsistent array lengths".to_string()
            ));
        }
        Ok(())
    }
}