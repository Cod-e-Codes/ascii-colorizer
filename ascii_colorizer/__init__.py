"""
ASCII Colorizer - Convert images and videos to colored ASCII art.

A Python package that converts images and videos into colored ASCII art
for display in modern terminals with ANSI color support.
"""

__version__ = "1.0.0"
__author__ = "ASCII Colorizer Team"
__description__ = "Convert images and videos to colored ASCII art"

from .image_processor import ImageProcessor
from .video_processor import VideoProcessor
from .renderer import Renderer

# Try to import GPU processor, but don't fail if PyTorch is not available
try:
    from .gpu_processor import GPUImageProcessor
    from .gpu_video_processor import GPUVideoProcessor
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    GPUImageProcessor = None
    GPUVideoProcessor = None

__all__ = ["ImageProcessor", "VideoProcessor", "Renderer"]

if GPU_AVAILABLE:
    __all__.extend(["GPUImageProcessor", "GPUVideoProcessor"]) 