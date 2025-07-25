"""
Video processor for converting videos to animated ASCII art.

This module handles loading video files, extracting frames,
and converting them to animated ASCII sequences.
"""

import cv2
import numpy as np
import time
from typing import Generator, List, Tuple, Optional
from PIL import Image

from .image_processor import ImageProcessor
from .utils import validate_file_type


class VideoProcessor:
    """
    Processes video files and converts them to animated ASCII art.
    """
    
    def __init__(self, use_truecolor: bool = None, detailed_chars: bool = False):
        """
        Initialize the VideoProcessor.
        
        Args:
            use_truecolor: Force TrueColor mode (None for auto-detection)
            detailed_chars: Use detailed ASCII character set for better detail
        """
        self.image_processor = ImageProcessor(use_truecolor, detailed_chars)
        self._performance_stats = {
            'frame_times': [],
            'avg_frame_time': 0.0,
            'slow_frames': 0
        }
        
    def load_video(self, filepath: str) -> cv2.VideoCapture:
        """
        Load a video file using OpenCV.
        
        Args:
            filepath: Path to the video file
            
        Returns:
            cv2.VideoCapture: Video capture object
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            IOError: If the video can't be loaded
        """
        # Validate file type first
        file_type = validate_file_type(filepath)
        if file_type != 'video':
            raise ValueError(f"File is not a video: {filepath}")
        
        cap = cv2.VideoCapture(filepath)
        
        if not cap.isOpened():
            raise IOError(f"Failed to open video file: {filepath}")
        
        return cap
    
    def get_video_info(self, filepath: str) -> dict:
        """
        Get information about a video file.
        
        Args:
            filepath: Path to the video file
            
        Returns:
            dict: Video information including fps, frame count, duration, etc.
        """
        try:
            cap = cv2.VideoCapture(filepath)
            
            if not cap.isOpened():
                return {'error': 'Could not open video file'}
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            # Calculate complexity metrics
            complexity_score = self._calculate_complexity_score(width, height, frame_count, duration)
            
            return {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration_seconds': duration,
                'duration_formatted': f"{int(duration // 60)}:{int(duration % 60):02d}",
                'complexity_score': complexity_score,
                'file_size_mb': self._get_file_size_mb(filepath)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_complexity_score(self, width: int, height: int, frame_count: int, duration: float) -> str:
        """
        Calculate video complexity for performance optimization.
        
        Returns:
            str: 'low', 'medium', 'high', or 'extreme'
        """
        # Calculate metrics
        resolution_score = (width * height) / 1000000  # Megapixels
        duration_score = duration / 60  # Minutes
        frame_density = frame_count / max(duration, 1)  # Frames per second
        
        total_score = resolution_score + duration_score + (frame_density / 30)
        
        if total_score < 2:
            return 'low'
        elif total_score < 5:
            return 'medium'
        elif total_score < 10:
            return 'high'
        else:
            return 'extreme'
    
    def _get_file_size_mb(self, filepath: str) -> float:
        """Get file size in megabytes."""
        try:
            import os
            size_bytes = os.path.getsize(filepath)
            return round(size_bytes / (1024 * 1024), 1)
        except:
            return 0.0
    
    def _get_adaptive_skip_frames(self, complexity: str, user_skip: int = 0) -> int:
        """
        Get adaptive frame skipping based on video complexity.
        
        Args:
            complexity: Video complexity score
            user_skip: User-specified skip frames
            
        Returns:
            int: Recommended skip frames
        """
        if user_skip > 0:
            return user_skip
        
        skip_recommendations = {
            'low': 0,
            'medium': 1,
            'high': 2,
            'extreme': 4
        }
        
        return skip_recommendations.get(complexity, 1)
    
    def _monitor_performance(self, frame_time: float) -> None:
        """
        Monitor frame processing performance and adjust if needed.
        
        Args:
            frame_time: Time taken to process the current frame
        """
        try:
            # Only monitor if frame_time is reasonable (prevent weird timing issues)
            if frame_time < 0 or frame_time > 60:  # Ignore unrealistic frame times
                return
                
            self._performance_stats['frame_times'].append(frame_time)
            
            # Keep only last 10 frame times for rolling average
            if len(self._performance_stats['frame_times']) > 10:
                self._performance_stats['frame_times'].pop(0)
            
            # Calculate average safely
            if self._performance_stats['frame_times']:
                self._performance_stats['avg_frame_time'] = sum(self._performance_stats['frame_times']) / len(self._performance_stats['frame_times'])
            
            # Count slow frames (> 0.5 seconds)
            if frame_time > 0.5:
                self._performance_stats['slow_frames'] += 1
        except Exception:
            # If anything goes wrong with performance monitoring, just continue
            # Don't let performance monitoring break video playback
            pass
    
    def frame_generator(self, filepath: str, target_width: int = None, 
                       max_height: int = None, use_color: bool = True,
                       skip_frames: int = 0, adaptive_performance: bool = True,
                       max_frames: Optional[int] = None) -> Generator[List[str], None, None]:
        """
        Generator that yields ASCII frames from a video with performance optimizations.
        
        Args:
            filepath: Path to the video file
            target_width: Target width for ASCII output
            max_height: Maximum height for ASCII output
            use_color: Whether to include color codes
            skip_frames: Number of frames to skip (0 for auto-adaptive)
            adaptive_performance: Enable adaptive performance optimizations
            
        Yields:
            List[str]: ASCII lines for each frame
        """
        cap = self.load_video(filepath)
        frame_count = 0
        processed_frames = 0
        
        # Get video info for adaptive optimization (but don't let it interfere)
        if adaptive_performance and skip_frames == 0:
            try:
                video_info = self.get_video_info(filepath)
                complexity = video_info.get('complexity_score', 'medium')
                
                # Only use adaptive skip for very complex videos
                adaptive_skip = self._get_adaptive_skip_frames(complexity, skip_frames)
                if adaptive_skip != skip_frames and complexity in ['high', 'extreme']:
                    print(f"Auto-optimization: Using {adaptive_skip} frame skip for {complexity} complexity video")
                    skip_frames = adaptive_skip
            except Exception as e:
                # If anything goes wrong with adaptive optimization, just continue normally
                print(f"Warning: Adaptive optimization failed, continuing normally: {e}")
                pass
        
        try:
            while True:
                start_time = time.time()
                ret, frame = cap.read()
                
                if not ret:
                    break  # End of video
                
                # Skip frames if requested (for performance)
                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    frame_count += 1
                    continue
                
                # Convert OpenCV frame (BGR) to PIL Image (RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Resize the frame
                resized_image = self.image_processor.resize_image(
                    pil_image, target_width, max_height
                )
                
                # Convert to ASCII
                ascii_lines = self.image_processor.image_to_ascii_lines(
                    resized_image, use_color
                )
                
                # Performance monitoring (but don't let it interfere with playback)
                if adaptive_performance:
                    try:
                        frame_time = time.time() - start_time
                        self._monitor_performance(frame_time)
                    except Exception:
                        # If performance monitoring fails, don't stop the video
                        pass
                
                yield ascii_lines
                frame_count += 1
                processed_frames += 1
                
                # Stop if we've reached max_frames
                if max_frames is not None and processed_frames >= max_frames:
                    break
                
                # Gentle performance feedback (less intrusive)
                if adaptive_performance and processed_frames % 50 == 0:  # Every 50 frames instead of 30
                    try:
                        avg_time = self._performance_stats.get('avg_frame_time', 0)
                        if avg_time > 0.5:  # Only warn if really slow (changed from 0.3 to 0.5)
                            print(f"Performance: Avg {avg_time:.2f}s/frame. Consider using --skip-frames {skip_frames + 1}")
                    except Exception:
                        # Don't let performance feedback break video playback
                        pass
                
                # Gentle memory cleanup (less frequent)
                if processed_frames % 200 == 0:  # Every 200 frames instead of 100
                    try:
                        import gc
                        gc.collect()
                    except Exception:
                        # Don't let garbage collection break video playback
                        pass
                
        except Exception as e:
            print(f"Error in frame processing: {e}")
            raise
        finally:
            cap.release()
    
    def extract_frame(self, filepath: str, frame_number: int, 
                     target_width: int = None, max_height: int = None,
                     use_color: bool = True) -> List[str]:
        """
        Extract and convert a specific frame to ASCII.
        
        Args:
            filepath: Path to the video file
            frame_number: Frame number to extract (0-based)
            target_width: Target width for ASCII output
            max_height: Maximum height for ASCII output
            use_color: Whether to include color codes
            
        Returns:
            List[str]: ASCII lines for the specified frame
        """
        cap = self.load_video(filepath)
        
        try:
            # Set the frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Could not read frame {frame_number}")
            
            # Convert to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Resize and convert to ASCII
            resized_image = self.image_processor.resize_image(
                pil_image, target_width, max_height
            )
            ascii_lines = self.image_processor.image_to_ascii_lines(
                resized_image, use_color
            )
            
            return ascii_lines
            
        finally:
            cap.release()
    
    def create_ascii_video_preview(self, filepath: str, num_frames: int = 10,
                                 target_width: int = None, max_height: int = None) -> List[List[str]]:
        """
        Create a preview with evenly spaced frames from the video.
        
        Args:
            filepath: Path to the video file
            num_frames: Number of preview frames to extract
            target_width: Target width for ASCII output
            max_height: Maximum height for ASCII output
            
        Returns:
            List[List[str]]: List of ASCII frame representations
        """
        video_info = self.get_video_info(filepath)
        
        if 'error' in video_info:
            raise IOError(f"Could not get video info: {video_info['error']}")
        
        total_frames = video_info['frame_count']
        complexity = video_info.get('complexity_score', 'medium')
        
        print(f"Video complexity: {complexity} ({video_info.get('file_size_mb', 0)}MB)")
        
        # Calculate frame positions for preview
        if num_frames >= total_frames:
            frame_positions = list(range(total_frames))
        else:
            step = total_frames // num_frames
            frame_positions = [i * step for i in range(num_frames)]
        
        preview_frames = []
        
        for i, frame_pos in enumerate(frame_positions):
            try:
                print(f"Extracting preview frame {i+1}/{len(frame_positions)}...")
                ascii_frame = self.extract_frame(
                    filepath, frame_pos, target_width, max_height, use_color=True
                )
                preview_frames.append(ascii_frame)
            except Exception as e:
                print(f"Warning: Could not extract frame {frame_pos}: {e}")
                continue
        
        return preview_frames
    
    def save_ascii_video(self, filepath: str, output_path: str, 
                        target_width: int = None, max_height: int = None,
                        skip_frames: int = 0, strip_colors: bool = False) -> None:
        """
        Save entire video as ASCII art to a text file.
        
        Args:
            filepath: Path to the video file
            output_path: Path where to save ASCII video
            target_width: Target width for output
            max_height: Maximum height for output
            skip_frames: Number of frames to skip
            strip_colors: Whether to remove ANSI color codes
        """
        video_info = self.get_video_info(filepath)
        complexity = video_info.get('complexity_score', 'medium')
        
        print(f"Processing {complexity} complexity video ({video_info.get('file_size_mb', 0)}MB)...")
        
        # Auto-adjust skip frames for file output
        if skip_frames == 0:
            skip_frames = self._get_adaptive_skip_frames(complexity)
            if skip_frames > 0:
                print(f"Auto-optimization: Using {skip_frames} frame skip for better performance")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write header with video info
                f.write("=" * 50 + "\n")
                f.write("ASCII VIDEO CONVERSION\n")
                f.write("=" * 50 + "\n")
                
                for key, value in video_info.items():
                    if 'error' not in key:
                        f.write(f"{key}: {value}\n")
                
                f.write("=" * 50 + "\n\n")
                
                # Process frames
                frame_count = 0
                start_time = time.time()
                
                for ascii_lines in self.frame_generator(
                    filepath, target_width, max_height, 
                    use_color=not strip_colors, skip_frames=skip_frames,
                    adaptive_performance=True
                ):
                    f.write(f"FRAME {frame_count}\n")
                    f.write("-" * 20 + "\n")
                    
                    for line in ascii_lines:
                        if strip_colors:
                            # Remove ANSI escape sequences
                            import re
                            line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                        f.write(line + "\n")
                    
                    f.write("\n" + "=" * 50 + "\n\n")
                    frame_count += 1
                    
                    # Enhanced progress indication
                    if frame_count % 10 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed if elapsed > 0 else 0
                        print(f"Processed {frame_count} frames... ({fps:.1f} fps)")
                        
        except Exception as e:
            raise IOError(f"Failed to save ASCII video to {output_path}: {str(e)}")
    
    def get_optimal_fps(self, filepath: str, max_fps: float = 15.0) -> float:
        """
        Get optimal FPS for terminal playback based on video complexity.
        
        Args:
            filepath: Path to the video file
            max_fps: Maximum FPS for terminal display
            
        Returns:
            float: Recommended FPS for smooth terminal playback
        """
        video_info = self.get_video_info(filepath)
        
        if 'error' in video_info:
            return max_fps
        
        original_fps = video_info.get('fps', max_fps)
        complexity = video_info.get('complexity_score', 'medium')
        
        # Adjust FPS based on complexity
        fps_limits = {
            'low': min(original_fps, max_fps),
            'medium': min(original_fps, max_fps * 0.8),
            'high': min(original_fps, max_fps * 0.6),
            'extreme': min(original_fps, max_fps * 0.4)
        }
        
        optimal_fps = fps_limits.get(complexity, max_fps)
        
        return optimal_fps
    
    def get_performance_recommendations(self, filepath: str) -> dict:
        """
        Get performance recommendations for a video file.
        
        Args:
            filepath: Path to the video file
            
        Returns:
            dict: Performance recommendations
        """
        video_info = self.get_video_info(filepath)
        complexity = video_info.get('complexity_score', 'medium')
        
        recommendations = {
            'complexity': complexity,
            'file_size_mb': video_info.get('file_size_mb', 0),
            'recommended_skip_frames': self._get_adaptive_skip_frames(complexity),
            'recommended_fps': self.get_optimal_fps(filepath),
            'suggested_width': 80 if complexity in ['high', 'extreme'] else 120,
            'tips': []
        }
        
        if complexity == 'extreme':
            recommendations['tips'].extend([
                "Consider using --skip-frames 4 or higher",
                "Use smaller dimensions (--width 60)",
                "Lower FPS (--fps 8)",
                "Consider --no-color for fastest processing"
            ])
        elif complexity == 'high':
            recommendations['tips'].extend([
                "Use --skip-frames 2 for better performance",
                "Consider --width 80 for faster rendering",
                "Lower FPS recommended (--fps 10)"
            ])
        elif complexity == 'medium':
            recommendations['tips'].append("Default settings should work well")
        else:
            recommendations['tips'].append("This video should process smoothly")
        
        return recommendations 