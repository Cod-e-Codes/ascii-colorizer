"""
Utility functions for ASCII colorizer.

This module contains shared helper functions for image resizing,
color palette management, and ASCII character mapping.
"""

import os
import shutil
from typing import Tuple, List
from PIL import Image


# ASCII characters ordered by brightness (lightest to darkest)
ASCII_CHARS = " .:-=+*#%@"
ASCII_CHARS_DETAILED = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"


def get_terminal_size() -> Tuple[int, int]:
    """
    Get the current terminal size.
    
    Returns:
        Tuple[int, int]: (width, height) of the terminal
    """
    try:
        size = shutil.get_terminal_size()
        return size.columns, size.lines
    except (AttributeError, OSError):
        # Fallback values if terminal size can't be determined
        return 80, 24


def calculate_new_size(image: Image.Image, target_width: int = None, max_height: int = None) -> Tuple[int, int]:
    """
    Calculate new image dimensions while preserving aspect ratio.
    
    Args:
        image: PIL Image object
        target_width: Desired width (if None, uses terminal width)
        max_height: Maximum height (if None, uses terminal height)
    
    Returns:
        Tuple[int, int]: (new_width, new_height)
    """
    original_width, original_height = image.size
    
    if target_width is None:
        terminal_width, terminal_height = get_terminal_size()
        target_width = max(20, terminal_width - 2)  # Leave some margin, minimum 20
        
    if max_height is None:
        terminal_width, terminal_height = get_terminal_size()
        max_height = max(10, terminal_height - 4)  # Leave some margin, minimum 10
    
    # Ensure minimum dimensions
    target_width = max(1, target_width)
    max_height = max(1, max_height)
    
    # Calculate aspect ratio
    aspect_ratio = original_width / original_height
    
    # Calculate height based on width
    new_height = int(target_width / aspect_ratio)
    
    # Adjust for character aspect ratio (characters are typically taller than wide)
    # Most terminal characters have roughly 2:1 height-to-width ratio
    new_height = int(new_height * 0.5)
    
    # Ensure minimum height
    new_height = max(1, new_height)
    
    # Ensure we don't exceed max height
    if new_height > max_height:
        new_height = max_height
        target_width = int(new_height * aspect_ratio * 2)  # Adjust width accordingly
        target_width = max(1, target_width)  # Ensure minimum width
    
    return target_width, new_height


def pixel_to_ascii(pixel_value: int, use_detailed: bool = False) -> str:
    """
    Convert a grayscale pixel value to an ASCII character.
    
    Args:
        pixel_value: Grayscale value (0-255)
        use_detailed: Whether to use detailed ASCII character set
    
    Returns:
        str: ASCII character representing the pixel brightness
    """
    chars = ASCII_CHARS_DETAILED if use_detailed else ASCII_CHARS
    
    # Normalize pixel value to character index
    char_index = int((pixel_value / 255) * (len(chars) - 1))
    return chars[char_index]


def rgb_to_ansi_truecolor(r: int, g: int, b: int) -> str:
    """
    Convert RGB values to ANSI TrueColor escape sequence.
    
    Args:
        r, g, b: RGB color values (0-255)
    
    Returns:
        str: ANSI escape sequence for setting foreground color
    """
    return f"\x1b[38;2;{r};{g};{b}m"


def rgb_to_ansi_256color(r: int, g: int, b: int) -> str:
    """
    Convert RGB values to ANSI 256-color escape sequence.
    Uses the standard 6x6x6 color cube.
    
    Args:
        r, g, b: RGB color values (0-255)
    
    Returns:
        str: ANSI escape sequence for 256-color mode
    """
    # Convert to 6-level values (0-5)
    r6 = int(r * 5 / 255)
    g6 = int(g * 5 / 255)
    b6 = int(b * 5 / 255)
    
    # Calculate color index (16-231 are the color cube)
    color_index = 16 + (36 * r6) + (6 * g6) + b6
    
    return f"\x1b[38;5;{color_index}m"


def reset_ansi() -> str:
    """
    Return ANSI reset sequence to clear formatting.
    
    Returns:
        str: ANSI reset sequence
    """
    return "\x1b[0m"


def clear_screen() -> str:
    """
    Return ANSI sequence to clear screen and move cursor to top-left.
    
    Returns:
        str: ANSI clear screen sequence
    """
    return "\x1b[H\x1b[J"


def supports_truecolor() -> bool:
    """
    Check if the terminal supports TrueColor (24-bit color).
    
    Returns:
        bool: True if TrueColor is supported
    """
    # Check common environment variables that indicate TrueColor support
    colorterm = os.environ.get('COLORTERM', '').lower()
    term = os.environ.get('TERM', '').lower()
    
    # Check for TrueColor indicators
    if colorterm in ('truecolor', '24bit'):
        return True
    
    # Check for terminals known to support TrueColor
    truecolor_terms = ['xterm-256color', 'screen-256color', 'tmux-256color']
    if any(term.startswith(t) for t in truecolor_terms):
        return True
    
    # Default to False for safety
    return False


def run_neofetch() -> None:
    """
    Run neofetch command and capture its output.
    
    Returns:
        None: Prints neofetch output directly
    """
    import subprocess
    import platform
    
    try:
        # On Windows, try winfetch first, then neofetch
        if platform.system() == 'Windows':
            try:
                subprocess.run(['winfetch'], check=True)
                return
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        
        # Try neofetch
        try:
            subprocess.run(['neofetch'], check=True)
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("\nError: neofetch not found. Please install neofetch:")
            if platform.system() == 'Windows':
                print("  scoop install neofetch")
                print("  or")
                print("  winget install neofetch")
            elif platform.system() == 'Darwin':  # macOS
                print("  brew install neofetch")
            else:  # Linux
                print("  sudo apt install neofetch  # Debian/Ubuntu")
                print("  sudo dnf install neofetch  # Fedora")
                print("  sudo pacman -S neofetch    # Arch")
            
    except Exception as e:
        print(f"\nError running neofetch: {e}")


def validate_file_type(filepath: str) -> str:
    """
    Validate and determine file type.
    
    Args:
        filepath: Path to the file
    
    Returns:
        str: 'image' or 'video'
    
    Raises:
        ValueError: If file type is not supported
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    ext = os.path.splitext(filepath)[1].lower()
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
    
    if ext in image_extensions:
        return 'image'
    elif ext in video_extensions:
        return 'video'
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported types: {image_extensions | video_extensions}") 