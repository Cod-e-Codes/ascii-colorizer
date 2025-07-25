# ASCII Colorizer

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Convert images and videos into colored ASCII art for terminal display using ANSI escape sequences. Supports both CPU and GPU acceleration via PyTorch/CUDA.

## Features

- **Image Processing**: Convert JPG, PNG, BMP, GIF, TIFF, and WebP images
- **Video Processing**: Convert MP4, AVI, MOV, MKV, WebM, and other formats
- **Color Support**: TrueColor (24-bit) and 256-color terminal modes
- **GPU Acceleration**: CUDA support for faster processing (optional)
- **Performance Optimization**: Optional adaptive complexity analysis and frame skipping
- **Aspect Ratio Preservation**: Maintains proper image proportions
- **Cross-Platform**: Windows, macOS, and Linux support

## Installation

```bash
# Clone the repository
git clone https://github.com/Cod-e-Codes/ascii-colorizer.git
cd ascii-colorizer

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### GPU Support (Optional)

For GPU acceleration, ensure you have:
- NVIDIA GPU with CUDA support
- CUDA 11.0+ (tested with CUDA 12.1)
- PyTorch with CUDA support

```bash
# Install PyTorch with CUDA (if not already installed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

```bash
# Convert an image
ascii-colorizer --file image.jpg

# Convert a video
ascii-colorizer --file video.mp4 --fps 15

# Use GPU acceleration
ascii-colorizer --file video.mp4 --gpu

# High-quality output
ascii-colorizer --file image.png --detailed --width 120
```

## Usage

### Image Processing

```bash
# Basic conversion
ascii-colorizer --file photo.jpg

# Custom dimensions
ascii-colorizer --file image.png --width 100 --height 40

# Save to file
ascii-colorizer --file photo.jpg --save output.txt

# GPU acceleration
ascii-colorizer --file large_image.png --gpu

# Get image information
ascii-colorizer --file image.jpg --info
```

### Video Processing

```bash
# Play video as ASCII animation
ascii-colorizer --file movie.mp4

# Control playback speed
ascii-colorizer --file video.mp4 --fps 20

# Play first 5 seconds
ascii-colorizer --file video.mp4 --duration 5

# Extract specific frame
ascii-colorizer --file video.mp4 --frame 100

# Performance optimization
ascii-colorizer --file large_video.mp4 --fast

# GPU acceleration
ascii-colorizer --file video.mp4 --gpu --gpu-batch-size 8

# Smooth playback
ascii-colorizer --file video.mp4 --smooth

# Play video then show system info
ascii-colorizer --file video.mp4 --then-neofetch
```

### Performance Options

```bash
# Get performance recommendations
ascii-colorizer --file video.mp4 --performance

# Fast mode (automatic optimization)
ascii-colorizer --file video.mp4 --fast

# Manual frame skipping
ascii-colorizer --file video.mp4 --skip-frames 2

# GPU benchmarking
ascii-colorizer --file image.jpg --gpu --benchmark
```

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--help`, `-h` | Show help message and exit |
| `--file`, `-f` | Path to image or video file |
| `--width`, `-w` | Override output width |
| `--height` | Override maximum height |
| `--fps` | Video playback FPS (default: 10.0) |
| `--duration` | Limit video playback to specified seconds |
| `--no-color` | Disable color output |
| `--detailed` | Use detailed ASCII character set |
| `--save`, `-s` | Save ASCII art to file |
| `--skip-frames` | Skip frames for performance |
| `--frame` | Extract specific frame number |
| `--preview` | Show preview frames only |
| `--info` | Show file information |
| `--smooth` | Enable smooth video playback |
| `--then-neofetch` | Run neofetch after video completion |
| `--fast` | Optimize for performance |
| `--performance` | Show performance recommendations |
| `--adaptive` | Enable adaptive optimizations |
| `--gpu` | Enable GPU acceleration |
| `--gpu-device` | GPU device selection (auto/cuda/cpu) |
| `--gpu-batch-size` | GPU batch size (default: 4) |
| `--benchmark` | Benchmark GPU vs CPU performance |
| `--gpu-info` | Show GPU information |
| `--version` | Show version and exit |

## Requirements

### System Requirements
- **Python**: 3.12 or higher (tested with 3.12.3)
- **Terminal**: Modern terminal with ANSI color support
- **Operating System**: Windows 10+, macOS 10.15+, Linux

### Dependencies
- **Pillow**: Image processing
- **OpenCV**: Video processing  
- **NumPy**: Array operations
- **Rich**: Terminal output
- **Colorama**: Windows compatibility
- **PyTorch**: GPU acceleration (optional)

### GPU Requirements (Optional)
- **NVIDIA GPU**: CUDA-compatible
- **CUDA**: 11.0 or higher (tested with 12.1)
- **GPU Memory**: 2GB+ recommended for video processing

## Performance

### Adaptive Optimization (Optional)
When using the `--adaptive` flag, the application analyzes video complexity and optimizes settings:
- **Low/Medium Complexity**: Standard processing
- **High Complexity**: Automatic frame skipping
- **Extreme Complexity**: Aggressive optimization

### Manual Optimization
For better performance on large files:

```bash
# Fast mode (recommended)
ascii-colorizer --file large_video.mp4 --fast

# Manual optimization
ascii-colorizer --file video.mp4 --skip-frames 3 --fps 8 --width 80

# GPU acceleration (2-4x speedup)
ascii-colorizer --file video.mp4 --gpu
```

## Supported Formats

**Images**: JPEG, PNG, BMP, GIF, TIFF, WebP  
**Videos**: MP4, AVI, MOV, MKV, WebM, FLV, WMV

## Architecture

```
ascii-colorizer/
├── ascii_colorizer/           # Main package
│   ├── image_processor.py     # CPU image processing
│   ├── video_processor.py     # CPU video processing
│   ├── gpu_processor.py       # GPU image processing
│   ├── gpu_video_processor.py # GPU video processing
│   ├── renderer.py            # Terminal output
│   └── utils.py               # Shared utilities
├── cli.py                     # Command-line interface
├── requirements.txt           # Dependencies
└── setup.py                   # Package setup
```

## Troubleshooting

**Colors not displaying:**
```bash
# Check terminal support
echo $COLORTERM
export COLORTERM=truecolor
```

**GPU not detected:**
```bash
# Check GPU availability
ascii-colorizer --gpu-info

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

**Poor performance:**
```bash
# Use fast mode
ascii-colorizer --file video.mp4 --fast

# Check recommendations
ascii-colorizer --file video.mp4 --performance
```

**Video playback issues:**
```bash
# Enable smooth playback
ascii-colorizer --file video.mp4 --smooth

# Reduce dimensions
ascii-colorizer --file video.mp4 --width 80 --height 20
```

## Contributing

Contributions are welcome. Please open an issue for major changes before submitting a pull request.

## License

MIT LICENSE - see [LICENSE](LICENSE) file for details. 