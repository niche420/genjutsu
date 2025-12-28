from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional


class Model3DBase(ABC):
    """Base class for all 3D generation models"""

    def __init__(self, device="cuda"):
        self.device = device
        self.is_loaded = False

    @abstractmethod
    def load(self) -> bool:
        """Load model weights and dependencies. Returns True if successful."""
        pass

    @abstractmethod
    def generate(
            self,
            prompt: str,
            output_path: Path,
            progress_callback: Optional[Callable[[float, str], None]] = None,
            **kwargs
    ) -> Path:
        """
        Generate 3D content from prompt.

        Args:
            prompt: Text description or image path
            output_path: Where to save .ply file
            progress_callback: Optional callback(progress: float, message: str) for progress updates
            **kwargs: Model-specific parameters

        Returns:
            Path to generated .ply file
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return model name"""
        pass

    @abstractmethod
    def get_estimated_time(self, **kwargs) -> int:
        """Return estimated generation time in seconds"""
        pass