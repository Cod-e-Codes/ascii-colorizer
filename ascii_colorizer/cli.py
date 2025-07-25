#!/usr/bin/env python3
"""
Command-line interface for ASCII Colorizer.

This module provides the main CLI entry point for converting
images and videos to colored ASCII art with optional GPU acceleration.
"""

import argparse
import sys
import os
from typing import Optional

from ascii_colorizer import ImageProcessor, VideoProcessor, Renderer

# Try to import GPU processors
try:
    from ascii_colorizer import GPUImageProcessor, GPU_AVAILABLE
    from ascii_colorizer.gpu_video_processor import GPUVideoProcessor
except ImportError:
    GPU_AVAILABLE = False
    GPUImageProcessor = None
    GPUVideoProcessor = None

from ascii_colorizer.utils import validate_file_type, get_terminal_size


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured parser
    """
    parser = argparse.ArgumentParser(
        prog='ascii-colorizer',
        description='Convert images and videos to colored ASCII art with GPU acceleration',
        epilog='Examples:\n'
               '  %(prog)s --file image.jpg\n'
               '  %(prog)s --file video.mp4 --fps 15\n'
               '  %(prog)s --file image.png --width 100 --no-color\n'
               '  %(prog)s --file video.avi --save output.txt\n'
               '  %(prog)s --file video.mp4 --smooth\n'
               '  %(prog)s --file large_video.mp4 --fast\n'
               '  %(prog)s --file video.mp4 --performance\n'
               '  %(prog)s --file image.jpg --gpu --benchmark\n'
               '  %(prog)s --file video.mp4 --gpu --gpu-batch-size 8',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--file', '-f',
        required=True,
        help='Path to image or video file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--width', '-w',
        type=int,
        help='Override output width (default: terminal width)'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        help='Override maximum output height (default: terminal height)'
    )
    
    parser.add_argument(
        '--fps',
        type=float,
        default=10.0,
        help='Video playback FPS (default: 10.0)'
    )
    
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable color output'
    )
    
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Use detailed ASCII character set for better quality'
    )
    
    parser.add_argument(
        '--save', '-s',
        help='Save ASCII art to file'
    )
    
    parser.add_argument(
        '--skip-frames',
        type=int,
        default=0,
        help='Skip frames for faster video processing (0 for auto-adaptive)'
    )
    
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Show preview frames instead of full video'
    )
    
    parser.add_argument(
        '--frame',
        type=int,
        help='Extract specific frame number from video (0-based)'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show file information and exit'
    )
    
    parser.add_argument(
        '--smooth',
        action='store_true',
        help='Enable smooth video playback (reduces flashing between frames)'
    )
    
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Fast mode: optimize for performance over quality'
    )
    
    parser.add_argument(
        '--performance',
        action='store_true',
        help='Show performance recommendations for the file'
    )
    
    parser.add_argument(
        '--adaptive',
        action='store_true',
        help='Enable adaptive performance optimizations (experimental)'
    )
    
    # GPU acceleration options
    if GPU_AVAILABLE:
        parser.add_argument(
            '--gpu',
            action='store_true',
            help='Enable GPU acceleration using PyTorch/CUDA'
        )
        
        parser.add_argument(
            '--gpu-device',
            type=str,
            default='auto',
            choices=['auto', 'cuda', 'cpu'],
            help='GPU device to use (default: auto)'
        )
        
        parser.add_argument(
            '--gpu-batch-size',
            type=int,
            default=4,
            help='Batch size for GPU processing (default: 4, higher=more GPU memory)'
        )
        
        parser.add_argument(
            '--benchmark',
            action='store_true',
            help='Benchmark GPU vs CPU performance and exit'
        )
        
        parser.add_argument(
            '--gpu-info',
            action='store_true',
            help='Show GPU information and exit'
        )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    return parser


def show_gpu_info():
    """Display GPU information and capabilities."""
    if not GPU_AVAILABLE:
        print("❌ GPU acceleration not available (PyTorch not installed)")
        return
    
    try:
        # Create a GPU processor to get info
        gpu_proc = GPUImageProcessor()
        memory_info = gpu_proc.get_gpu_memory_info()
        
        print("🖥️  GPU Information")
        print("=" * 50)
        
        if memory_info['gpu_available']:
            print(f"✅ GPU Available: {memory_info.get('device_name', 'Unknown')}")
            print(f"💾 Total Memory: {memory_info.get('total_memory_mb', 0):.1f} MB")
            print(f"📊 Currently Allocated: {memory_info.get('allocated_mb', 0):.1f} MB")
            print(f"🔒 Reserved: {memory_info.get('reserved_mb', 0):.1f} MB")
            print("✨ CUDA Support: Available")
            
            import torch
            print(f"🔧 PyTorch Version: {torch.__version__}")
            print(f"🎯 CUDA Version: {torch.version.cuda}")
            print(f"🏃 GPU Count: {torch.cuda.device_count()}")
        else:
            print("❌ GPU not available")
            if 'error' in memory_info:
                print(f"Error: {memory_info['error']}")
                
    except Exception as e:
        print(f"❌ Error getting GPU info: {e}")


def handle_image(args, image_processor, renderer: Renderer) -> None:
    """
    Handle image processing and display with optional GPU acceleration.
    
    Args:
        args: Parsed command line arguments
        image_processor: ImageProcessor or GPUImageProcessor instance
        renderer: Renderer instance
    """
    try:
        # Show file info if requested
        if args.info:
            if hasattr(image_processor, 'get_image_info'):
                info = image_processor.get_image_info(args.file)
            else:
                # GPU processor doesn't have get_image_info, use regular processor
                temp_proc = ImageProcessor()
                info = temp_proc.get_image_info(args.file)
            
            print("Image Information:")
            print("=" * 30)
            for key, value in info.items():
                print(f"{key}: {value}")
            return
        
        # Benchmark if requested
        if hasattr(args, 'benchmark') and args.benchmark and hasattr(image_processor, 'benchmark_vs_cpu'):
            print("🏁 Running GPU vs CPU benchmark...")
            results = image_processor.benchmark_vs_cpu(args.file)
            
            print("\n📊 Benchmark Results:")
            print("=" * 50)
            print(f"Image size: {results['image_size']}")
            print(f"Iterations: {results['iterations']}")
            print(f"Average GPU time: {results['avg_gpu_time']:.4f}s")
            print(f"Average CPU time: {results['avg_cpu_time']:.4f}s")
            print(f"Speedup: {results['speedup']:.2f}x")
            print(f"GPU is {'faster' if results['gpu_faster'] else 'slower'}")
            return
        
        # Process the image
        processing_mode = "GPU" if hasattr(image_processor, 'gpu_available') and image_processor.gpu_available else "CPU"
        print(f"Processing image with {processing_mode}: {args.file}")
        
        ascii_lines = image_processor.process_image(
            args.file,
            target_width=args.width,
            max_height=args.height,
            use_color=not args.no_color
        )
        
        # Show performance stats for GPU processing
        if hasattr(image_processor, 'get_performance_stats'):
            stats = image_processor.get_performance_stats()
            if stats['total_operations'] > 0:
                print(f"⚡ Performance: {stats['average_time_per_op']:.3f}s "
                      f"({stats['gpu_percentage']:.1f}% GPU)")
        
        # Save to file if requested
        if args.save:
            print(f"Saving ASCII art to: {args.save}")
            # Use regular image processor for saving (simpler)
            if hasattr(image_processor, 'save_ascii_to_file'):
                image_processor.save_ascii_to_file(ascii_lines, args.save, strip_colors=args.no_color)
            else:
                temp_proc = ImageProcessor()
                temp_proc.save_ascii_to_file(ascii_lines, args.save, strip_colors=args.no_color)
        
        # Display the result
        renderer.render_static(ascii_lines)
        
    except Exception as e:
        print(f"Error processing image: {e}", file=sys.stderr)
        sys.exit(1)


def handle_video(args, video_processor, renderer: Renderer) -> None:
    """
    Handle video processing and display with optional GPU acceleration.
    
    Args:
        args: Parsed command line arguments
        video_processor: VideoProcessor or GPUVideoProcessor instance
        renderer: Renderer instance
    """
    try:
        # Show GPU performance recommendations if requested
        if args.performance:
            if hasattr(video_processor, 'optimize_for_gpu'):
                recommendations = video_processor.optimize_for_gpu(args.file)
            else:
                recommendations = video_processor.get_performance_recommendations(args.file)
            
            print("Performance Recommendations:")
            print("=" * 50)
            print(f"Video complexity: {recommendations['complexity'].upper()}")
            print(f"File size: {recommendations['file_size_mb']}MB")
            print(f"Recommended skip frames: {recommendations['recommended_skip_frames']}")
            print(f"Recommended FPS: {recommendations['recommended_fps']:.1f}")
            print(f"Suggested width: {recommendations['suggested_width']}")
            
            if 'gpu_available' in recommendations:
                print(f"GPU available: {'Yes' if recommendations['gpu_available'] else 'No'}")
                if 'gpu_device' in recommendations:
                    print(f"GPU device: {recommendations['gpu_device']}")
                if 'recommended_batch_size' in recommendations:
                    print(f"Recommended batch size: {recommendations['recommended_batch_size']}")
            
            print("\nOptimization tips:")
            base_tips = recommendations.get('tips', [])
            gpu_tips = recommendations.get('gpu_tips', [])
            for tip in base_tips + gpu_tips:
                print(f"  • {tip}")
            return
        
        # Benchmark video processing if requested
        if hasattr(args, 'benchmark') and args.benchmark and hasattr(video_processor, 'benchmark_video_processing'):
            print("🏁 Running GPU vs CPU video benchmark...")
            results = video_processor.benchmark_video_processing(args.file, duration_seconds=5)
            
            print("\n📊 Video Benchmark Results:")
            print("=" * 50)
            print(f"Benchmark duration: {results['benchmark_duration']}s")
            print(f"GPU frames processed: {results['gpu_frames_processed']}")
            print(f"CPU frames processed: {results['cpu_frames_processed']}")
            print(f"GPU FPS: {results['gpu_fps']:.2f}")
            print(f"CPU FPS: {results['cpu_fps']:.2f}")
            print(f"Speedup: {results['speedup']:.2f}x")
            print(f"Performance: {results['performance_improvement']}")
            return
        
        # Show file info if requested
        if args.info:
            info = video_processor.get_video_info(args.file)
            print("Video Information:")
            print("=" * 30)
            for key, value in info.items():
                print(f"{key}: {value}")
            return
        
        # Apply fast mode optimizations
        if args.fast:
            print("Fast mode enabled - optimizing for performance")
            video_info = video_processor.get_video_info(args.file)
            complexity = video_info.get('complexity_score', 'medium')
            
            # Auto-adjust settings for fast mode
            if not args.width:
                args.width = 60 if complexity in ['high', 'extreme'] else 80
            if args.skip_frames == 0:
                args.skip_frames = max(2, video_processor._get_adaptive_skip_frames(complexity))
            if args.fps == 10.0:  # Default FPS
                args.fps = max(6, video_processor.get_optimal_fps(args.file) * 0.7)
            
            print(f"Fast mode settings: width={args.width}, skip_frames={args.skip_frames}, fps={args.fps:.1f}")

        # Extract specific frame if requested
        if args.frame is not None:
            print(f"Extracting frame {args.frame} from video: {args.file}")
            ascii_lines = video_processor.extract_frame(
                args.file, args.frame,
                target_width=args.width,
                max_height=args.height,
                use_color=not args.no_color
            )
            
            if args.save:
                # Use image processor for saving
                temp_proc = ImageProcessor()
                temp_proc.save_ascii_to_file(ascii_lines, args.save, strip_colors=args.no_color)
                print(f"Frame saved to: {args.save}")
            
            renderer.render_static(ascii_lines)
            return
        
        # Show preview if requested
        if args.preview:
            print(f"Creating preview of video: {args.file}")
            preview_frames = video_processor.create_ascii_video_preview(
                args.file, num_frames=10,
                target_width=args.width,
                max_height=args.height
            )
            
            print(f"Preview: {len(preview_frames)} frames")
            renderer.display_controls()
            
            # Display preview as animation
            def preview_generator():
                for frame in preview_frames:
                    yield frame
            
            renderer.render_animation(
                preview_generator(), 
                fps=2.0,  # Slower for preview
                total_frames=len(preview_frames)
            )
            return
        
        # Save full video to file if requested
        if args.save:
            print(f"Saving ASCII video to: {args.save}")
            video_processor.save_ascii_video(
                args.file, args.save,
                target_width=args.width,
                max_height=args.height,
                skip_frames=args.skip_frames,
                strip_colors=args.no_color
            )
            print("Video saved successfully!")
            return
        
        # Process and display video animation
        processing_mode = "GPU" if hasattr(video_processor, 'gpu_available') and video_processor.gpu_available else "CPU"
        print(f"Processing video with {processing_mode}: {args.file}")
        
        # Get video info for frame count and complexity
        video_info = video_processor.get_video_info(args.file)
        total_frames = video_info.get('frame_count', None)
        complexity = video_info.get('complexity_score', 'medium')
        file_size = video_info.get('file_size_mb', 0)
        
        # Show video complexity info
        print(f"Video complexity: {complexity} ({file_size}MB, {total_frames} frames)")
        
        # Show GPU batch size for GPU processing
        if hasattr(video_processor, 'gpu_available') and video_processor.gpu_available:
            print(f"GPU batch size: {video_processor.batch_size}")
        
        # Adjust FPS if needed
        optimal_fps = video_processor.get_optimal_fps(args.file, args.fps)
        if optimal_fps != args.fps:
            print(f"Using optimal FPS: {optimal_fps:.1f} (requested: {args.fps})")
        
        # Display playback mode
        if args.smooth:
            print("Smooth playback mode enabled (reduces flashing)")
        
        if args.adaptive:
            print("Adaptive performance optimizations enabled (experimental)")
        
        # Performance warnings for large files
        if complexity in ['high', 'extreme'] and args.skip_frames == 0:
            print(f"⚠️  Large {complexity} complexity video detected!")
            if hasattr(video_processor, 'gpu_available') and video_processor.gpu_available:
                print(f"💡 GPU acceleration should help with performance")
            else:
                print(f"💡 Consider using --fast or --skip-frames for better performance")
        
        renderer.display_controls()
        
        # Create frame generator (GPU or CPU)
        if hasattr(video_processor, 'frame_generator_gpu') and video_processor.gpu_available:
            frame_gen = video_processor.frame_generator_gpu(
                args.file,
                target_width=args.width,
                max_height=args.height,
                use_color=not args.no_color,
                skip_frames=args.skip_frames,
                adaptive_performance=args.adaptive
            )
        else:
            frame_gen = video_processor.frame_generator(
                args.file,
                target_width=args.width,
                max_height=args.height,
                use_color=not args.no_color,
                skip_frames=args.skip_frames,
                adaptive_performance=args.adaptive
            )
        
        # Render animation
        renderer.render_animation(frame_gen, fps=optimal_fps, total_frames=total_frames)
        
        # Show final performance stats for GPU processing
        if hasattr(video_processor, 'get_gpu_performance_stats'):
            stats = video_processor.get_gpu_performance_stats()
            if stats['total_frames_processed'] > 0:
                print(f"\n⚡ Final Performance Stats:")
                print(f"   Total frames: {stats['total_frames_processed']}")
                print(f"   GPU processed: {stats['frames_processed_gpu']} ({stats['gpu_usage_percentage']:.1f}%)")
                print(f"   Average GPU time: {stats['avg_gpu_time_per_frame']:.3f}s/frame")
                print(f"   Peak GPU memory: {stats['memory_usage_peak_mb']:.1f}MB")
        
    except Exception as e:
        print(f"Error processing video: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """
    Main entry point for the CLI application.
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Show GPU info if requested
    if hasattr(args, 'gpu_info') and args.gpu_info:
        show_gpu_info()
        return
    
    # Validate file exists
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Determine file type
        file_type = validate_file_type(args.file)
        
        # Check terminal size if width/height not specified
        if not args.width or not args.height:
            terminal_width, terminal_height = get_terminal_size()
            if terminal_width < 80 or terminal_height < 24:
                print(f"Warning: Small terminal size ({terminal_width}x{terminal_height}). "
                      f"Consider resizing for better output quality.")
        
        # Initialize components
        use_color = not args.no_color
        use_gpu = getattr(args, 'gpu', False) and GPU_AVAILABLE
        
        if use_gpu and not GPU_AVAILABLE:
            print("Warning: GPU requested but PyTorch not available, falling back to CPU")
            use_gpu = False
        
        if file_type == 'image':
            if use_gpu:
                image_processor = GPUImageProcessor(
                    use_truecolor=None,  # Auto-detect
                    detailed_chars=args.detailed,
                    device=getattr(args, 'gpu_device', 'auto')
                )
            else:
                image_processor = ImageProcessor(
                    use_truecolor=None,  # Auto-detect
                    detailed_chars=args.detailed
                )
            
            renderer = Renderer(enable_colors=use_color)
            handle_image(args, image_processor, renderer)
            
        elif file_type == 'video':
            if use_gpu:
                video_processor = GPUVideoProcessor(
                    use_truecolor=None,  # Auto-detect
                    detailed_chars=args.detailed,
                    device=getattr(args, 'gpu_device', 'auto'),
                    batch_size=getattr(args, 'gpu_batch_size', 4)
                )
            else:
                video_processor = VideoProcessor(
                    use_truecolor=None,  # Auto-detect
                    detailed_chars=args.detailed
                )
            
            renderer = Renderer(enable_colors=use_color)
            handle_video(args, video_processor, renderer)
        
        # Cleanup
        renderer.cleanup()
        
        # Clear GPU cache if used
        if use_gpu:
            if file_type == 'image' and hasattr(image_processor, 'clear_gpu_cache'):
                image_processor.clear_gpu_cache()
            elif file_type == 'video' and hasattr(video_processor, 'gpu_image_processor'):
                video_processor.gpu_image_processor.clear_gpu_cache()
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main() 