"""
Terminal renderer for displaying colored ASCII art.

This module handles the display of ASCII art in the terminal,
including screen clearing, cursor positioning, and color output.
"""

import sys
import time
from typing import List, Optional
import colorama
from .utils import clear_screen, get_terminal_size


class Renderer:
    """
    Handles rendering ASCII art to the terminal with proper formatting.
    """
    
    def __init__(self, enable_colors: bool = True):
        """
        Initialize the renderer.
        
        Args:
            enable_colors: Whether to enable color output
        """
        self.enable_colors = enable_colors
        self.previous_frame_height = 0
        
        # Initialize colorama for Windows compatibility
        colorama.init(autoreset=True)
    
    def render_static(self, ascii_lines: List[str], clear_first: bool = True) -> None:
        """
        Render static ASCII art to the terminal.
        
        Args:
            ascii_lines: List of ASCII lines to display
            clear_first: Whether to clear the screen first
        """
        if clear_first:
            print(clear_screen(), end='')
        
        for line in ascii_lines:
            if self.enable_colors:
                print(line)
            else:
                # Strip color codes if colors are disabled
                import re
                clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                print(clean_line)
    
    def _move_cursor_home(self) -> str:
        """
        Return ANSI sequence to move cursor to top-left without clearing screen.
        
        Returns:
            str: ANSI cursor home sequence
        """
        return "\x1b[H"
    
    def _clear_line(self) -> str:
        """
        Return ANSI sequence to clear current line.
        
        Returns:
            str: ANSI clear line sequence
        """
        return "\x1b[2K"
    
    def _hide_cursor(self) -> str:
        """
        Return ANSI sequence to hide cursor.
        
        Returns:
            str: ANSI hide cursor sequence
        """
        return "\x1b[?25l"
    
    def _show_cursor(self) -> str:
        """
        Return ANSI sequence to show cursor.
        
        Returns:
            str: ANSI show cursor sequence
        """
        return "\x1b[?25h"
    
    def render_animation(self, frame_generator, fps: float = 10.0, 
                        total_frames: Optional[int] = None) -> None:
        """
        Render animated ASCII art frame by frame with smooth transitions.
        
        Args:
            frame_generator: Generator yielding ASCII line lists
            fps: Frames per second
            total_frames: Total number of frames (for progress indication)
        """
        frame_delay = 1.0 / fps
        frame_count = 0
        first_frame = True
        
        try:
            # Hide cursor for smoother animation
            print(self._hide_cursor(), end='')
            
            for ascii_lines in frame_generator:
                current_frame_height = len(ascii_lines)
                
                if first_frame:
                    # Clear screen only for the first frame
                    print(clear_screen(), end='')
                    first_frame = False
                else:
                    # Move cursor to home position without clearing
                    print(self._move_cursor_home(), end='')
                    
                    # If the new frame is shorter than the previous one,
                    # we need to clear the extra lines
                    if current_frame_height < self.previous_frame_height:
                        # Render the current frame first
                        for line in ascii_lines:
                            if self.enable_colors:
                                print(f"{self._clear_line()}{line}")
                            else:
                                import re
                                clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                                print(f"{self._clear_line()}{clean_line}")
                        
                        # Clear any remaining lines from the previous frame
                        for _ in range(self.previous_frame_height - current_frame_height):
                            print(self._clear_line())
                        
                        # Move cursor back to position for progress info
                        print(f"\x1b[{current_frame_height + 1}H", end='')
                    else:
                        # Normal rendering - just overwrite existing content
                        for line in ascii_lines:
                            if self.enable_colors:
                                print(f"{self._clear_line()}{line}")
                            else:
                                import re
                                clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                                print(f"{self._clear_line()}{clean_line}")
                
                # Show progress if total frames is known
                if total_frames:
                    progress = (frame_count + 1) / total_frames * 100
                    print(f"{self._clear_line()}Frame {frame_count + 1}/{total_frames} ({progress:.1f}%)")
                else:
                    print(f"{self._clear_line()}Frame {frame_count + 1}")
                
                self.previous_frame_height = current_frame_height
                frame_count += 1
                
                # Flush output to ensure smooth playback
                sys.stdout.flush()
                
                # Wait for next frame
                time.sleep(frame_delay)
                
        except KeyboardInterrupt:
            print(f"\n{self._show_cursor()}\nAnimation stopped by user.")
            sys.exit(0)
        except Exception as e:
            print(f"\n{self._show_cursor()}\nError during animation: {e}")
            sys.exit(1)
        finally:
            # Always show cursor when done
            print(self._show_cursor(), end='')
    
    def render_with_info(self, ascii_lines: List[str], info: dict = None) -> None:
        """
        Render ASCII art with additional information.
        
        Args:
            ascii_lines: List of ASCII lines to display
            info: Dictionary containing image/video information
        """
        print(clear_screen(), end='')
        
        # Display info header if provided
        if info:
            print("=" * 50)
            for key, value in info.items():
                print(f"{key}: {value}")
            print("=" * 50)
            print()
        
        # Render the ASCII art
        for line in ascii_lines:
            if self.enable_colors:
                print(line)
            else:
                import re
                clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                print(clean_line)
    
    def check_terminal_size(self, required_width: int, required_height: int) -> bool:
        """
        Check if terminal is large enough for the content.
        
        Args:
            required_width: Minimum required width
            required_height: Minimum required height
            
        Returns:
            bool: True if terminal is large enough
        """
        terminal_width, terminal_height = get_terminal_size()
        
        if terminal_width < required_width or terminal_height < required_height:
            print(f"Warning: Terminal size ({terminal_width}x{terminal_height}) "
                  f"may be too small for optimal display "
                  f"(recommended: {required_width}x{required_height})")
            return False
        
        return True
    
    def display_controls(self) -> None:
        """
        Display control instructions for video playback.
        """
        print("\nControls:")
        print("  Ctrl+C - Stop playback")
        print("  Terminal resize - May cause display issues")
        print("\nTip: For smoother playback, ensure your terminal supports fast refresh rates")
        print()
    
    def cleanup(self) -> None:
        """
        Clean up renderer resources and reset terminal.
        """
        # Reset terminal colors and cursor
        print("\x1b[0m", end='')  # Reset colors
        print(self._show_cursor(), end='')  # Show cursor
        
        # Reset previous frame height
        self.previous_frame_height = 0
        
        # Deinitialize colorama
        colorama.deinit() 