"""
GPU-accelerated video processor for converting videos to animated ASCII art using PyTorch.

This module extends the video processing capabilities with GPU acceleration
for faster frame-by-frame conversion, especially beneficial for large videos.
"""

import cv2
import time
import torch
from typing import Generator, List, Optional
from PIL import Image

from .gpu_processor import GPUImageProcessor
from .video_processor import VideoProcessor
from .utils import validate_file_type


class GPUVideoProcessor(VideoProcessor):
    """
    GPU-accelerated video processor that uses GPUImageProcessor for faster frame processing.
    Falls back to CPU processing if GPU is not available.
    """
    
    def __init__(self, use_truecolor: bool = None, detailed_chars: bool = False, 
                 device: str = 'auto', batch_size: int = 4):
        """
        Initialize the GPU-accelerated VideoProcessor.
        
        Args:
            use_truecolor: Force TrueColor mode (None for auto-detection)
            detailed_chars: Use detailed ASCII character set for better detail
            device: Device to use ('auto', 'cuda', 'cpu')
            batch_size: Number of frames to process in parallel
        """
        super().__init__(use_truecolor, detailed_chars)
        
        # Replace the image processor with GPU version
        self.gpu_image_processor = GPUImageProcessor(
            use_truecolor=use_truecolor,
            detailed_chars=detailed_chars,
            device=device,
            batch_size=batch_size
        )
        
        self.batch_size = batch_size
        self.gpu_available = self.gpu_image_processor.gpu_available
        
        # Performance tracking
        self.gpu_stats = {
            'frames_processed_gpu': 0,
            'frames_processed_cpu': 0,
            'total_gpu_time': 0.0,
            'total_cpu_time': 0.0,
            'memory_usage_peak': 0.0
        }
    
    def frame_generator_gpu(self, filepath: str, target_width: int = None,
                           max_height: int = None, use_color: bool = True,
                           skip_frames: int = 0, adaptive_performance: bool = True) -> Generator[List[str], None, None]:
        """
        GPU-accelerated frame generator with smooth individual frame processing.
        
        Args:
            filepath: Path to the video file
            target_width: Target width for ASCII output
            max_height: Maximum height for ASCII output
            use_color: Whether to include color codes
            skip_frames: Number of frames to skip
            adaptive_performance: Enable adaptive performance optimizations
            
        Yields:
            List[str]: ASCII lines for each frame
        """
        cap = self.load_video(filepath)
        frame_count = 0
        processed_frames = 0
        
        # Adaptive optimization (reuse parent logic)
        if adaptive_performance and skip_frames == 0:
            try:
                video_info = self.get_video_info(filepath)
                complexity = video_info.get('complexity_score', 'medium')
                adaptive_skip = self._get_adaptive_skip_frames(complexity, skip_frames)
                if adaptive_skip != skip_frames and complexity in ['high', 'extreme']:
                    print(f"GPU Auto-optimization: Using {adaptive_skip} frame skip for {complexity} complexity video")
                    skip_frames = adaptive_skip
            except Exception as e:
                print(f"Warning: GPU adaptive optimization failed, continuing normally: {e}")
                pass
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if requested
                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    frame_count += 1
                    continue
                
                # Process frame immediately for smooth playback (no batching)
                start_time = time.time()
                
                # Convert OpenCV frame to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Calculate new size (reuse existing logic)
                from .utils import calculate_new_size
                new_width, new_height = calculate_new_size(pil_image, target_width, max_height)
                
                # GPU-accelerated resize and ASCII conversion
                try:
                    resized_image = self.gpu_image_processor.resize_image_gpu(pil_image, new_width, new_height)
                    ascii_lines = self.gpu_image_processor.image_to_ascii_lines_gpu(resized_image, use_color)
                    
                    # Track GPU performance
                    frame_time = time.time() - start_time
                    self.gpu_stats['frames_processed_gpu'] += 1
                    self.gpu_stats['total_gpu_time'] += frame_time
                    
                except Exception as e:
                    # Fall back to CPU processing for this frame
                    ascii_lines = self._process_single_frame_cpu(pil_image, target_width, max_height, use_color)
                
                yield ascii_lines
                frame_count += 1
                processed_frames += 1
                
                # Less frequent performance monitoring to avoid interference
                if adaptive_performance and processed_frames % 200 == 0:
                    try:
                        stats = self.get_gpu_performance_stats()
                        if stats['avg_gpu_time_per_frame'] > 0.3:
                            print(f"GPU Performance: {stats['avg_gpu_time_per_frame']:.3f}s/frame")
                    except Exception:
                        pass
                
                # Less frequent memory management
                if processed_frames % 100 == 0:
                    try:
                        self.gpu_image_processor.clear_gpu_cache()
                    except Exception:
                        pass
                        
        except Exception as e:
            print(f"Error in GPU frame processing: {e}")
            raise
        finally:
            cap.release()
    
    def frame_generator_gpu_batch(self, filepath: str, target_width: int = None,
                                 max_height: int = None, use_color: bool = True,
                                 skip_frames: int = 0, adaptive_performance: bool = True) -> Generator[List[str], None, None]:
        """
        GPU-accelerated frame generator WITH batch processing for maximum throughput.
        Use this for non-realtime scenarios like saving to file where smooth playback isn't needed.
        
        Args:
            filepath: Path to the video file
            target_width: Target width for ASCII output
            max_height: Maximum height for ASCII output
            use_color: Whether to include color codes
            skip_frames: Number of frames to skip
            adaptive_performance: Enable adaptive performance optimizations
            
        Yields:
            List[str]: ASCII lines for each frame
        """
        cap = self.load_video(filepath)
        frame_count = 0
        processed_frames = 0
        
        # Adaptive optimization (reuse parent logic)
        if adaptive_performance and skip_frames == 0:
            try:
                video_info = self.get_video_info(filepath)
                complexity = video_info.get('complexity_score', 'medium')
                adaptive_skip = self._get_adaptive_skip_frames(complexity, skip_frames)
                if adaptive_skip != skip_frames and complexity in ['high', 'extreme']:
                    print(f"GPU Batch Auto-optimization: Using {adaptive_skip} frame skip for {complexity} complexity video")
                    skip_frames = adaptive_skip
            except Exception as e:
                print(f"Warning: GPU adaptive optimization failed, continuing normally: {e}")
                pass
        
        # Batch processing for maximum GPU efficiency
        frame_batch = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # Process remaining frames in batch
                    if frame_batch:
                        yield from self._process_frame_batch(frame_batch, target_width, max_height, use_color)
                    break
                
                # Skip frames if requested
                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    frame_count += 1
                    continue
                
                # Add frame to batch
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frame_batch.append(pil_image)
                
                # Process batch when full
                if len(frame_batch) >= self.batch_size:
                    yield from self._process_frame_batch(frame_batch, target_width, max_height, use_color)
                    frame_batch = []
                
                frame_count += 1
                processed_frames += 1
                
                # Performance monitoring (less frequent for batch processing)
                if adaptive_performance and processed_frames % 100 == 0:
                    try:
                        stats = self.get_gpu_performance_stats()
                        if stats['avg_gpu_time_per_frame'] > 0.2:
                            print(f"GPU Batch Performance: {stats['avg_gpu_time_per_frame']:.3f}s/frame, "
                                  f"{stats['gpu_usage_percentage']:.1f}% GPU")
                    except Exception:
                        pass
                
                # Memory management for GPU
                if processed_frames % 50 == 0:
                    try:
                        self.gpu_image_processor.clear_gpu_cache()
                    except Exception:
                        pass
                        
        except Exception as e:
            print(f"Error in GPU batch processing: {e}")
            raise
        finally:
            cap.release()
    
    def _process_frame_batch(self, frame_batch: List[Image.Image], target_width: int = None,
                           max_height: int = None, use_color: bool = True) -> Generator[List[str], None, None]:
        """
        Process a batch of frames using GPU acceleration.
        
        Args:
            frame_batch: List of PIL Images to process
            target_width: Target width for ASCII output
            max_height: Maximum height for ASCII output
            use_color: Whether to include color codes
            
        Yields:
            List[str]: ASCII lines for each frame in the batch
        """
        start_time = time.time()
        
        try:
            for pil_image in frame_batch:
                # Calculate new size (reuse existing logic)
                from .utils import calculate_new_size
                new_width, new_height = calculate_new_size(pil_image, target_width, max_height)
                
                # GPU-accelerated resize and ASCII conversion
                resized_image = self.gpu_image_processor.resize_image_gpu(pil_image, new_width, new_height)
                ascii_lines = self.gpu_image_processor.image_to_ascii_lines_gpu(resized_image, use_color)
                
                self.gpu_stats['frames_processed_gpu'] += 1
                yield ascii_lines
                
            # Track performance
            batch_time = time.time() - start_time
            self.gpu_stats['total_gpu_time'] += batch_time
            
            # Monitor GPU memory
            try:
                memory_info = self.gpu_image_processor.get_gpu_memory_info()
                if 'allocated_mb' in memory_info:
                    self.gpu_stats['memory_usage_peak'] = max(
                        self.gpu_stats['memory_usage_peak'],
                        memory_info['allocated_mb']
                    )
            except Exception:
                pass
                
        except Exception as e:
            print(f"GPU batch processing failed, falling back to CPU: {e}")
            # Fallback to CPU processing
            for pil_image in frame_batch:
                yield self._process_single_frame_cpu(pil_image, target_width, max_height, use_color)
    
    def _process_single_frame_cpu(self, pil_image: Image.Image, target_width: int = None,
                                 max_height: int = None, use_color: bool = True) -> List[str]:
        """CPU fallback for single frame processing."""
        start_time = time.time()
        
        # Use parent class image processor for CPU fallback
        from .utils import calculate_new_size
        new_width, new_height = calculate_new_size(pil_image, target_width, max_height)
        
        resized_image = self.image_processor.resize_image(pil_image, target_width, max_height)
        ascii_lines = self.image_processor.image_to_ascii_lines(resized_image, use_color)
        
        # Track CPU performance
        cpu_time = time.time() - start_time
        self.gpu_stats['total_cpu_time'] += cpu_time
        self.gpu_stats['frames_processed_cpu'] += 1
        
        return ascii_lines
    
    def get_gpu_performance_stats(self) -> dict:
        """Get detailed GPU performance statistics."""
        base_stats = self.gpu_image_processor.get_performance_stats()
        
        total_frames = self.gpu_stats['frames_processed_gpu'] + self.gpu_stats['frames_processed_cpu']
        total_time = self.gpu_stats['total_gpu_time'] + self.gpu_stats['total_cpu_time']
        
        gpu_stats = {
            'gpu_available': self.gpu_available,
            'total_frames_processed': total_frames,
            'frames_processed_gpu': self.gpu_stats['frames_processed_gpu'],
            'frames_processed_cpu': self.gpu_stats['frames_processed_cpu'],
            'gpu_usage_percentage': (self.gpu_stats['frames_processed_gpu'] / max(total_frames, 1)) * 100,
            'total_processing_time': total_time,
            'avg_gpu_time_per_frame': self.gpu_stats['total_gpu_time'] / max(self.gpu_stats['frames_processed_gpu'], 1),
            'avg_cpu_time_per_frame': self.gpu_stats['total_cpu_time'] / max(self.gpu_stats['frames_processed_cpu'], 1),
            'memory_usage_peak_mb': self.gpu_stats['memory_usage_peak'],
            'base_gpu_stats': base_stats
        }
        
        return gpu_stats
    
    def benchmark_video_processing(self, filepath: str, duration_seconds: int = 10) -> dict:
        """
        Benchmark GPU vs CPU video processing performance.
        
        Args:
            filepath: Path to video file
            duration_seconds: How many seconds of video to process for benchmark
            
        Returns:
            dict: Benchmark results
        """
        print(f"Benchmarking GPU vs CPU video processing ({duration_seconds}s of video)...")
        
        video_info = self.get_video_info(filepath)
        fps = video_info.get('fps', 30)
        max_frames = int(duration_seconds * fps)
        
        # GPU benchmark
        gpu_start = time.time()
        gpu_frame_count = 0
        
        if self.gpu_available:
            try:
                for ascii_lines in self.frame_generator_gpu(filepath, target_width=80, max_height=40):
                    gpu_frame_count += 1
                    if gpu_frame_count >= max_frames:
                        break
            except Exception as e:
                print(f"GPU benchmark failed: {e}")
                
        gpu_time = time.time() - gpu_start
        
        # CPU benchmark (using parent class)
        cpu_start = time.time()
        cpu_frame_count = 0
        
        try:
            for ascii_lines in super().frame_generator(filepath, target_width=80, max_height=40, adaptive_performance=False):
                cpu_frame_count += 1
                if cpu_frame_count >= max_frames:
                    break
        except Exception as e:
            print(f"CPU benchmark failed: {e}")
            
        cpu_time = time.time() - cpu_start
        
        # Calculate results
        gpu_fps = gpu_frame_count / max(gpu_time, 0.001)
        cpu_fps = cpu_frame_count / max(cpu_time, 0.001)
        speedup = gpu_fps / max(cpu_fps, 0.001)
        
        return {
            'video_file': filepath,
            'benchmark_duration': duration_seconds,
            'gpu_available': self.gpu_available,
            'gpu_time': gpu_time,
            'cpu_time': cpu_time,
            'gpu_frames_processed': gpu_frame_count,
            'cpu_frames_processed': cpu_frame_count,
            'gpu_fps': gpu_fps,
            'cpu_fps': cpu_fps,
            'speedup': speedup,
            'gpu_faster': speedup > 1.0,
            'performance_improvement': f"{((speedup - 1) * 100):.1f}%" if speedup > 1 else f"{((1 - speedup) * 100):.1f}% slower"
        }
    
    def optimize_for_gpu(self, filepath: str) -> dict:
        """
        Get GPU-specific optimization recommendations for a video file.
        
        Args:
            filepath: Path to the video file
            
        Returns:
            dict: GPU optimization recommendations
        """
        base_recommendations = super().get_performance_recommendations(filepath)
        
        gpu_recommendations = base_recommendations.copy()
        gpu_recommendations['gpu_available'] = self.gpu_available
        
        if self.gpu_available:
            memory_info = self.gpu_image_processor.get_gpu_memory_info()
            
            gpu_recommendations['gpu_memory_total'] = memory_info.get('total_memory_mb', 0)
            gpu_recommendations['gpu_device'] = memory_info.get('device_name', 'Unknown')
            
            # GPU-specific tips
            gpu_tips = [
                "GPU acceleration is available and recommended",
                f"Use --gpu for {2-4}x faster processing",
                "Larger batch sizes may improve GPU utilization",
            ]
            
            # Adjust recommendations based on GPU memory
            total_memory = memory_info.get('total_memory_mb', 0)
            if total_memory > 8000:  # 8GB+
                gpu_tips.append("High-end GPU detected: can handle large videos efficiently")
                gpu_recommendations['recommended_batch_size'] = 8
            elif total_memory > 4000:  # 4-8GB
                gpu_tips.append("Mid-range GPU: good performance expected")
                gpu_recommendations['recommended_batch_size'] = 4
            else:  # <4GB
                gpu_tips.append("Lower-end GPU: consider smaller dimensions for best performance")
                gpu_recommendations['recommended_batch_size'] = 2
                
            gpu_recommendations['gpu_tips'] = gpu_tips
        else:
            gpu_recommendations['gpu_tips'] = [
                "GPU acceleration not available",
                "Install PyTorch with CUDA support for better performance",
                "CPU processing will be used"
            ]
        
        return gpu_recommendations 