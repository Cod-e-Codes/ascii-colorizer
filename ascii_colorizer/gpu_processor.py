"""
GPU-accelerated processor for converting images to colored ASCII art using PyTorch.

This module provides GPU acceleration for the most computationally intensive
parts of ASCII art conversion, with automatic fallback to CPU processing.
"""

import time
import gc
from typing import List, Tuple, Optional, Union
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from .utils import (
    pixel_to_ascii,
    rgb_to_ansi_truecolor,
    rgb_to_ansi_256color,
    reset_ansi,
    supports_truecolor,
    ASCII_CHARS,
    ASCII_CHARS_DETAILED
)


class GPUImageProcessor:
    """
    GPU-accelerated image processor using PyTorch for faster ASCII art conversion.
    Automatically falls back to CPU if GPU is not available or encounters errors.
    """
    
    def __init__(self, use_truecolor: bool = None, detailed_chars: bool = False, 
                 device: str = 'auto', batch_size: int = 64):
        """
        Initialize the GPU-accelerated ImageProcessor.
        
        Args:
            use_truecolor: Force TrueColor mode (None for auto-detection)
            detailed_chars: Use detailed ASCII character set for better detail
            device: Device to use ('auto', 'cuda', 'cpu')
            batch_size: Batch size for processing (affects GPU memory usage)
        """
        self.use_truecolor = supports_truecolor() if use_truecolor is None else use_truecolor
        self.detailed_chars = detailed_chars
        self.batch_size = batch_size
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.gpu_available = torch.cuda.is_available() and self.device.type == 'cuda'
        
        # Pre-compute ASCII character mapping tensor for GPU
        self.ascii_chars = ASCII_CHARS_DETAILED if detailed_chars else ASCII_CHARS
        self._setup_gpu_tensors()
        
        # Performance tracking
        self.stats = {
            'gpu_time': 0.0,
            'cpu_time': 0.0,
            'gpu_fallback_count': 0,
            'total_operations': 0
        }
    
    def _setup_gpu_tensors(self) -> None:
        """Pre-compute tensors for GPU operations."""
        if not self.gpu_available:
            return
            
        try:
            # Create luminance weights tensor for RGB to grayscale conversion
            self.luminance_weights = torch.tensor([0.299, 0.587, 0.114], 
                                                 device=self.device, dtype=torch.float32)
            
            # Pre-compute color cube for 256-color mode (6x6x6 cube)
            if not self.use_truecolor:
                color_indices = torch.arange(216, device=self.device).view(6, 6, 6)
                self.color_cube = color_indices + 16  # Offset for 256-color cube
                
        except Exception as e:
            print(f"Warning: GPU tensor setup failed, falling back to CPU: {e}")
            self.gpu_available = False
    
    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to GPU tensor."""
        # Convert to numpy then to tensor for better performance
        np_array = np.array(image)
        tensor = torch.from_numpy(np_array).float().to(self.device)
        
        # Normalize to 0-1 range
        tensor = tensor / 255.0
        
        # Ensure RGB format: (H, W, 3)
        if len(tensor.shape) == 2:  # Grayscale
            tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)
        
        return tensor
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert GPU tensor back to PIL Image."""
        # Denormalize and convert to uint8
        tensor = (tensor * 255).clamp(0, 255).byte()
        
        # Move to CPU and convert to numpy
        np_array = tensor.cpu().numpy()
        
        return Image.fromarray(np_array, 'RGB')
    
    def resize_image_gpu(self, image: Image.Image, target_width: int, target_height: int) -> Image.Image:
        """
        GPU-accelerated image resizing using PyTorch.
        
        Args:
            image: PIL Image to resize
            target_width: Target width
            target_height: Target height
            
        Returns:
            PIL.Image.Image: Resized image
        """
        if not self.gpu_available:
            # Fallback to PIL
            return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        try:
            start_time = time.time()
            
            # Convert to tensor and add batch dimension
            tensor = self._pil_to_tensor(image)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            
            # Use PyTorch's bicubic interpolation for better quality (closer to LANCZOS)
            resized_tensor = F.interpolate(
                tensor, 
                size=(target_height, target_width),
                mode='bicubic',  # Higher quality than bilinear
                align_corners=False
            )
            
            # Convert back to PIL format
            resized_tensor = resized_tensor.squeeze(0).permute(1, 2, 0)  # (H, W, C)
            result = self._tensor_to_pil(resized_tensor)
            
            self.stats['gpu_time'] += time.time() - start_time
            return result
            
        except Exception as e:
            print(f"GPU resize failed, falling back to CPU: {e}")
            self.stats['gpu_fallback_count'] += 1
            return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    def _rgb_to_grayscale_gpu(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """Convert RGB tensor to grayscale using GPU."""
        # rgb_tensor shape: (H, W, 3)
        grayscale = torch.sum(rgb_tensor * self.luminance_weights, dim=-1)
        return grayscale
    
    def _get_ascii_chars_gpu(self, grayscale_tensor: torch.Tensor) -> List[str]:
        """Convert grayscale values to ASCII characters using GPU."""
        # Normalize to character indices
        char_indices = (grayscale_tensor * (len(self.ascii_chars) - 1)).long()
        char_indices = torch.clamp(char_indices, 0, len(self.ascii_chars) - 1)
        
        # Convert to CPU for character mapping (unavoidable)
        char_indices_cpu = char_indices.cpu().numpy()
        
        # Map to ASCII characters
        ascii_chars = []
        for row in char_indices_cpu:
            row_chars = ''.join(self.ascii_chars[idx] for idx in row)
            ascii_chars.append(row_chars)
        
        return ascii_chars
    
    def _apply_colors_gpu(self, rgb_tensor: torch.Tensor, ascii_chars: List[str]) -> List[str]:
        """Apply ANSI colors to ASCII characters using GPU-processed colors."""
        colored_lines = []
        
        # Convert tensor to CPU for color code generation
        rgb_cpu = (rgb_tensor * 255).byte().cpu().numpy()
        height, width, _ = rgb_cpu.shape
        
        for y in range(height):
            line = ""
            ascii_row = ascii_chars[y]
            
            for x in range(width):
                r, g, b = rgb_cpu[y, x]
                ascii_char = ascii_row[x]
                
                if self.use_truecolor:
                    color_code = rgb_to_ansi_truecolor(int(r), int(g), int(b))
                else:
                    color_code = rgb_to_ansi_256color(int(r), int(g), int(b))
                
                line += f"{color_code}{ascii_char}{reset_ansi()}"
            
            colored_lines.append(line)
        
        return colored_lines
    
    def image_to_ascii_lines_gpu(self, image: Image.Image, use_color: bool = True) -> List[str]:
        """
        GPU-accelerated conversion of image to ASCII lines.
        
        Args:
            image: PIL Image to convert
            use_color: Whether to include ANSI color codes
            
        Returns:
            List[str]: List of ASCII lines with color codes
        """
        if not self.gpu_available:
            # Fallback to CPU processing
            return self._image_to_ascii_lines_cpu(image, use_color)
        
        try:
            start_time = time.time()
            
            # Convert image to GPU tensor
            rgb_tensor = self._pil_to_tensor(image)
            
            # Convert to grayscale for ASCII character selection
            grayscale_tensor = self._rgb_to_grayscale_gpu(rgb_tensor)
            
            # Get ASCII characters
            ascii_chars = self._get_ascii_chars_gpu(grayscale_tensor)
            
            # Apply colors if requested
            if use_color:
                result = self._apply_colors_gpu(rgb_tensor, ascii_chars)
            else:
                result = ascii_chars
            
            self.stats['gpu_time'] += time.time() - start_time
            self.stats['total_operations'] += 1
            
            return result
            
        except Exception as e:
            print(f"GPU processing failed, falling back to CPU: {e}")
            self.stats['gpu_fallback_count'] += 1
            return self._image_to_ascii_lines_cpu(image, use_color)
    
    def _image_to_ascii_lines_cpu(self, image: Image.Image, use_color: bool = True) -> List[str]:
        """CPU fallback for ASCII conversion."""
        start_time = time.time()
        
        # Convert image to numpy array for easier processing
        img_array = np.array(image)
        height, width, _ = img_array.shape
        
        ascii_lines = []
        
        for y in range(height):
            line = ""
            for x in range(width):
                # Get RGB values for this pixel
                r, g, b = img_array[y, x]
                
                # Convert to grayscale for ASCII character selection
                gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
                
                # Get ASCII character for this brightness level
                ascii_char = pixel_to_ascii(gray_value, self.detailed_chars)
                
                if use_color:
                    # Add color codes
                    if self.use_truecolor:
                        color_code = rgb_to_ansi_truecolor(r, g, b)
                    else:
                        color_code = rgb_to_ansi_256color(r, g, b)
                    
                    line += f"{color_code}{ascii_char}{reset_ansi()}"
                else:
                    line += ascii_char
            
            ascii_lines.append(line)
        
        self.stats['cpu_time'] += time.time() - start_time
        self.stats['total_operations'] += 1
        
        return ascii_lines
    
    def process_image(self, filepath: str, target_width: int = None, max_height: int = None, 
                     use_color: bool = True) -> List[str]:
        """
        Complete GPU-accelerated pipeline: load, resize, and convert image to ASCII.
        
        Args:
            filepath: Path to the image file
            target_width: Target width for output
            max_height: Maximum height for output
            use_color: Whether to include color codes
            
        Returns:
            List[str]: List of ASCII lines representing the image
        """
        try:
            # Load image
            image = Image.open(filepath)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Calculate new size (reuse existing logic)
            from .utils import calculate_new_size
            new_width, new_height = calculate_new_size(image, target_width, max_height)
            
            # GPU-accelerated resize
            resized_image = self.resize_image_gpu(image, new_width, new_height)
            
            # GPU-accelerated ASCII conversion
            ascii_lines = self.image_to_ascii_lines_gpu(resized_image, use_color)
            
            return ascii_lines
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {filepath}")
        except Exception as e:
            raise IOError(f"Failed to process image {filepath}: {str(e)}")
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics for GPU vs CPU processing."""
        total_time = self.stats['gpu_time'] + self.stats['cpu_time']
        
        return {
            'gpu_available': self.gpu_available,
            'device': str(self.device),
            'total_operations': self.stats['total_operations'],
            'gpu_time': self.stats['gpu_time'],
            'cpu_time': self.stats['cpu_time'],
            'total_time': total_time,
            'gpu_percentage': (self.stats['gpu_time'] / total_time * 100) if total_time > 0 else 0,
            'gpu_fallback_count': self.stats['gpu_fallback_count'],
            'average_time_per_op': total_time / max(self.stats['total_operations'], 1)
        }
    
    def benchmark_vs_cpu(self, test_image_path: str, iterations: int = 5) -> dict:
        """
        Benchmark GPU vs CPU performance on a test image.
        
        Args:
            test_image_path: Path to test image
            iterations: Number of iterations to run
            
        Returns:
            dict: Benchmark results
        """
        print(f"Running benchmark with {iterations} iterations...")
        
        # Load test image
        image = Image.open(test_image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to standard size for fair comparison
        test_image = image.resize((80, 40), Image.Resampling.LANCZOS)
        
        # GPU benchmark
        gpu_times = []
        if self.gpu_available:
            for i in range(iterations):
                start_time = time.time()
                self.image_to_ascii_lines_gpu(test_image, use_color=True)
                gpu_times.append(time.time() - start_time)
        
        # CPU benchmark
        cpu_times = []
        for i in range(iterations):
            start_time = time.time()
            self._image_to_ascii_lines_cpu(test_image, use_color=True)
            cpu_times.append(time.time() - start_time)
        
        # Calculate results
        avg_gpu_time = sum(gpu_times) / len(gpu_times) if gpu_times else 0
        avg_cpu_time = sum(cpu_times) / len(cpu_times)
        speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 0
        
        return {
            'gpu_available': self.gpu_available,
            'iterations': iterations,
            'image_size': test_image.size,
            'avg_gpu_time': avg_gpu_time,
            'avg_cpu_time': avg_cpu_time,
            'speedup': speedup,
            'gpu_faster': speedup > 1.0,
            'gpu_times': gpu_times,
            'cpu_times': cpu_times
        }
    
    def clear_gpu_cache(self) -> None:
        """Clear GPU cache to free memory."""
        if self.gpu_available:
            torch.cuda.empty_cache()
            gc.collect()
    
    def get_gpu_memory_info(self) -> dict:
        """Get GPU memory usage information."""
        if not self.gpu_available:
            return {'gpu_available': False}
        
        try:
            allocated = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
            reserved = torch.cuda.memory_reserved(self.device) / 1024**2   # MB
            max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**2  # MB
            
            return {
                'gpu_available': True,
                'device_name': torch.cuda.get_device_name(self.device),
                'allocated_mb': allocated,
                'reserved_mb': reserved,
                'max_allocated_mb': max_allocated,
                'total_memory_mb': torch.cuda.get_device_properties(self.device).total_memory / 1024**2
            }
        except Exception as e:
            return {'gpu_available': True, 'error': str(e)} 