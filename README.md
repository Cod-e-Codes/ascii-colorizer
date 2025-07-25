# ASCII Colorizer

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

Convert any image or video file into beautiful **colored ASCII art** that displays directly in your terminal using ANSI escape sequences. Perfect for terminal enthusiasts, digital artists, and anyone who wants to add some flair to their command-line experience!

## ✨ Features

- 🎨 **Full Color Support**: TrueColor (24-bit) and 256-color terminal support
- 🖼️ **Image Processing**: Convert JPG, PNG, BMP, GIF, TIFF, and WebP images
- 🎬 **Video Processing**: Convert MP4, AVI, MOV, MKV, WebM, and more
- 📐 **Aspect Ratio Preservation**: Automatically maintains proper proportions
- ⚡ **Performance Options**: Frame skipping and optimized FPS for smooth playback
- 💾 **Save Output**: Export ASCII art to text files
- 🖥️ **Cross-Platform**: Works on Linux, macOS, and Windows Terminal
- 🎛️ **Customizable**: Multiple ASCII character sets and sizing options

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ascii-colorizer/ascii-colorizer.git
cd ascii-colorizer

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```bash
# Convert an image
ascii-colorizer --file path/to/image.jpg

# Convert a video with custom FPS
ascii-colorizer --file path/to/video.mp4 --fps 15

# High-quality detailed output
ascii-colorizer --file image.png --detailed --width 120

# No colors (monochrome ASCII)
ascii-colorizer --file image.jpg --no-color

# Save to file
ascii-colorizer --file video.mp4 --save output.txt
```

## 📖 Detailed Usage

### Image Processing

```bash
# Basic image conversion
ascii-colorizer --file cat.jpg

# Custom dimensions
ascii-colorizer --file landscape.png --width 150 --height 50

# Save ASCII art to file
ascii-colorizer --file photo.jpg --save my_ascii_art.txt

# Get image information
ascii-colorizer --file image.png --info
```

### Video Processing

```bash
# Play video as ASCII animation
ascii-colorizer --file movie.mp4

# Control playback speed
ascii-colorizer --file video.avi --fps 20

# Extract a specific frame
ascii-colorizer --file video.mp4 --frame 100

# Create a preview (10 sample frames)
ascii-colorizer --file video.mov --preview

# Skip frames for performance
ascii-colorizer --file large_video.mp4 --skip-frames 2

# Smooth playback (no flashing between frames)
ascii-colorizer --file video.mp4 --smooth

# Fast mode for large videos (optimizes performance)
ascii-colorizer --file large_video.mp4 --fast

# Get performance recommendations
ascii-colorizer --file video.mp4 --performance

# Save entire video as ASCII
ascii-colorizer --file video.avi --save ascii_video.txt
```

### Advanced Options

```bash
# High-quality detailed ASCII characters
ascii-colorizer --file image.jpg --detailed

# Force specific dimensions
ascii-colorizer --file video.mp4 --width 100 --height 30

# Disable colors for better compatibility
ascii-colorizer --file image.png --no-color

# Process large videos efficiently
ascii-colorizer --file big_video.mp4 --skip-frames 3 --fps 10
```

## 🎛️ Command-Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--file`, `-f` | Path to image or video file | `--file image.jpg` |
| `--width`, `-w` | Override output width | `--width 120` |
| `--height` | Override maximum height | `--height 40` |
| `--fps` | Video playback FPS (default: 10.0) | `--fps 15` |
| `--no-color` | Disable color output | `--no-color` |
| `--detailed` | Use detailed ASCII character set | `--detailed` |
| `--save`, `-s` | Save ASCII art to file | `--save output.txt` |
| `--skip-frames` | Skip frames for performance | `--skip-frames 2` |
| `--preview` | Show preview frames only | `--preview` |
| `--frame` | Extract specific frame number | `--frame 50` |
| `--info` | Show file information | `--info` |
| `--smooth` | Enable smooth video playback | `--smooth` |
| `--fast` | Fast mode: optimize for performance | `--fast` |
| `--performance` | Show performance recommendations | `--performance` |
| `--no-adaptive` | Disable adaptive optimizations | `--no-adaptive` |
| `--version` | Show version information | `--version` |

## 🎨 Examples

### Image Conversion

```bash
# Convert a photo with custom width
ascii-colorizer --file sunset.jpg --width 100
```

**Output:**
```
Processing image: sunset.jpg
==================================================
size: (1920, 1080)
mode: RGB
format: JPEG
width: 1920
height: 1080
==================================================

[Colored ASCII art appears here]
```

### Video Animation

```bash
# Play a video as ASCII animation
ascii-colorizer --file dancing.mp4 --fps 12
```

**Output:**
```
Processing video: dancing.mp4
Using optimal FPS: 12.0 (requested: 12.0)

Controls:
  Ctrl+C - Stop playback
  Terminal resize - May cause display issues

[Animated ASCII art plays here]
Frame 1/240 (0.4%)
```

## 🏗️ Architecture

The project follows a modular architecture:

```
ascii-colorizer/
├── ascii_colorizer/           # Main package
│   ├── __init__.py           # Package initialization
│   ├── image_processor.py    # Image → ASCII conversion
│   ├── video_processor.py    # Video → ASCII stream
│   ├── renderer.py           # Terminal output logic
│   └── utils.py              # Shared utilities
├── cli.py                    # Command-line interface
├── requirements.txt          # Dependencies
├── setup.py                  # Package setup
└── README.md                 # This file
```

## 🔧 Requirements

- **Python**: 3.7 or higher
- **Terminal**: Modern terminal with ANSI color support
  - ✅ Windows Terminal
  - ✅ macOS Terminal
  - ✅ Linux terminals (GNOME Terminal, Konsole, etc.)
  - ✅ VS Code integrated terminal
  - ⚠️ Limited support for older terminals

### Dependencies

- **Pillow**: Image processing
- **OpenCV**: Video processing
- **Rich**: Enhanced terminal output
- **Colorama**: Windows terminal compatibility
- **NumPy**: Efficient array operations

## 🎯 Performance Tips

### 🚀 **NEW: Automatic Performance Optimization**
The ASCII Colorizer now automatically analyzes video complexity and optimizes settings!

1. **Auto-Adaptive Processing**: Videos are automatically categorized as low/medium/high/extreme complexity
2. **Smart Frame Skipping**: Automatically skips frames based on video complexity
3. **Performance Monitoring**: Real-time performance feedback during processing
4. **Memory Management**: Automatic garbage collection for long videos

### ⚡ **Manual Performance Options**

1. **Fast Mode**: Use `--fast` for automatic performance optimization
2. **Get Recommendations**: Use `--performance` to see optimization suggestions
3. **Frame Skipping**: Use `--skip-frames N` or let auto-adaptive choose (set to 0)
4. **Terminal Size**: Smaller dimensions = faster rendering
5. **FPS Control**: Lower FPS for smoother performance on slower systems
6. **Color Mode**: Use `--no-color` for fastest processing
7. **Smooth Playback**: Use `--smooth` for videos to eliminate flashing between frames

### 📊 **Performance Examples**

```bash
# Auto-optimized for any video
ascii-colorizer --file video.mp4

# Fast mode for large files (auto-optimizes everything)
ascii-colorizer --file large_video.mp4 --fast

# Check what optimizations are recommended
ascii-colorizer --file video.mp4 --performance

# Manual optimization for extreme cases
ascii-colorizer --file huge_video.mp4 --skip-frames 4 --fps 8 --width 60 --no-color
```

## 🐛 Troubleshooting

### Common Issues

**Colors not showing:**
```bash
# Check terminal color support
echo $COLORTERM
echo $TERM

# Try forcing color mode
export COLORTERM=truecolor
ascii-colorizer --file image.jpg
```

**Video not loading:**
```bash
# Check file format support
ascii-colorizer --file video.mp4 --info

# Try different video file
ascii-colorizer --file video.avi
```

**Terminal too small:**
```bash
# Use smaller dimensions
ascii-colorizer --file image.jpg --width 80 --height 20
```

**Slow performance:**
```bash
# Skip frames and reduce FPS
ascii-colorizer --file video.mp4 --skip-frames 2 --fps 8
```

## 📊 Supported Formats

### Images
- **JPEG** (.jpg, .jpeg)
- **PNG** (.png)
- **BMP** (.bmp)
- **GIF** (.gif)
- **TIFF** (.tiff, .tif)
- **WebP** (.webp)

### Videos
- **MP4** (.mp4)
- **AVI** (.avi)
- **MOV** (.mov)
- **MKV** (.mkv)
- **WebM** (.webm)
- **FLV** (.flv)
- **WMV** (.wmv)

## 🚧 Limitations

- **Performance**: Large videos may be slow on older systems
- **Terminal Compatibility**: Some older terminals don't support TrueColor
- **Memory Usage**: High-resolution videos require more memory
- **CPU Intensive**: No GPU acceleration for video processing

## 🔮 Future Enhancements

- 🎨 **Dithering algorithms** for better grayscale mapping
- 🌐 **Web interface** (Flask/Streamlit)
- 📹 **Live webcam mode** with real-time ASCII conversion
- 🎮 **Interactive controls** (pause, rewind, color toggle)
- 🎞️ **Export to HTML/GIF** with ASCII overlay
- ⚡ **GPU acceleration** for faster video processing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/ascii-colorizer/ascii-colorizer.git
cd ascii-colorizer
pip install -e ".[dev]"

# Run tests
pytest

# Code formatting
black .
flake8 .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Pillow** team for excellent image processing capabilities
- **OpenCV** community for robust video processing
- **Rich** library for beautiful terminal output
- ASCII art community for inspiration

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/ascii-colorizer/ascii-colorizer/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/ascii-colorizer/ascii-colorizer/discussions)
- 📖 **Documentation**: [Wiki](https://github.com/ascii-colorizer/ascii-colorizer/wiki)

---

**Made with ❤️ for terminal enthusiasts everywhere!** 