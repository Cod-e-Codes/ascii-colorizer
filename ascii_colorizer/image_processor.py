"""
Image processor for converting images to colored ASCII art.

This module handles loading images, resizing them appropriately,
and converting each pixel to a colored ASCII character.
"""

from typing import List, Tuple, Optional
from PIL import Image
import numpy as np

from .utils import (
    calculate_new_size,
    pixel_to_ascii,
    rgb_to_ansi_truecolor,
    rgb_to_ansi_256color,
    reset_ansi,
    supports_truecolor
)


class ImageProcessor:
    """
    Processes images and converts them to colored ASCII art.
    """
    
    def __init__(self, use_truecolor: bool = None, detailed_chars: bool = False):
        """
        Initialize the ImageProcessor.
        
        Args:
            use_truecolor: Force TrueColor mode (None for auto-detection)
            detailed_chars: Use detailed ASCII character set for better detail
        """
        self.use_truecolor = supports_truecolor() if use_truecolor is None else use_truecolor
        self.detailed_chars = detailed_chars
    
    def load_image(self, filepath: str) -> Image.Image:
        """
        Load an image from file.
        
        Args:
            filepath: Path to the image file
            
        Returns:
            PIL.Image.Image: Loaded image
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            IOError: If the file can't be loaded as an image
        """
        try:
            image = Image.open(filepath)
            # Convert to RGB if necessary (handles RGBA, palette mode, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {filepath}")
        except Exception as e:
            raise IOError(f"Failed to load image {filepath}: {str(e)}")
    
    def resize_image(self, image: Image.Image, target_width: int = None, max_height: int = None) -> Image.Image:
        """
        Resize image while preserving aspect ratio.
        
        Args:
            image: PIL Image to resize
            target_width: Target width (None for terminal width)
            max_height: Maximum height (None for terminal height)
            
        Returns:
            PIL.Image.Image: Resized image
        """
        new_width, new_height = calculate_new_size(image, target_width, max_height)
        
        # Use LANCZOS for high-quality downsampling
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized
    
    def image_to_ascii_lines(self, image: Image.Image, use_color: bool = True) -> List[str]:
        """
        Convert an image to a list of colored ASCII lines.
        
        Args:
            image: PIL Image to convert
            use_color: Whether to include ANSI color codes
            
        Returns:
            List[str]: List of ASCII lines with color codes
        """
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
                # Using standard luminance formula
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
        
        return ascii_lines
    
    def process_image(self, filepath: str, target_width: int = None, max_height: int = None, 
                     use_color: bool = True) -> List[str]:
        """
        Complete pipeline: load, resize, and convert image to ASCII.
        
        Args:
            filepath: Path to the image file
            target_width: Target width for output
            max_height: Maximum height for output
            use_color: Whether to include color codes
            
        Returns:
            List[str]: List of ASCII lines representing the image
        """
        # Load and resize image
        image = self.load_image(filepath)
        resized_image = self.resize_image(image, target_width, max_height)
        
        # Convert to ASCII
        ascii_lines = self.image_to_ascii_lines(resized_image, use_color)
        
        return ascii_lines
    
    def get_image_info(self, filepath: str) -> dict:
        """
        Get information about an image file.
        
        Args:
            filepath: Path to the image file
            
        Returns:
            dict: Image information including size, mode, format
        """
        try:
            with Image.open(filepath) as img:
                return {
                    'size': img.size,
                    'mode': img.mode,
                    'format': img.format,
                    'width': img.width,
                    'height': img.height
                }
        except Exception as e:
            return {'error': str(e)}
    
    def save_ascii_to_file(self, ascii_lines: List[str], output_path: str, 
                          strip_colors: bool = False) -> None:
        """
        Save ASCII art to a text file.
        
        Args:
            ascii_lines: List of ASCII lines to save
            output_path: Path where to save the file
            strip_colors: Whether to remove ANSI color codes before saving
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for line in ascii_lines:
                    if strip_colors:
                        # Simple regex to remove ANSI escape sequences
                        import re
                        line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                    f.write(line + '\n')
        except Exception as e:
            raise IOError(f"Failed to save ASCII art to {output_path}: {str(e)}") 