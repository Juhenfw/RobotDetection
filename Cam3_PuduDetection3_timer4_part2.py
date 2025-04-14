import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import csv
from datetime import datetime, timedelta
import threading
import json
import pickle

# ========== TIMER SYSTEM ==========
class TimerSystem:
    """Advanced timer system for object tracking with multiple visualization options"""
    
    def __init__(self, fps, enable_logging=True):
        self.fps = fps
        self.enable_logging = enable_logging
        self.log_file = f"timer_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        # Core timing data structures
        self.stationary_since = {}  # {object_id: first_stationary_time}
        self.timer_start_time = {}  # {area: {object_id: start_time}}
        self.active_timers = {}     # {area: {object_id: is_active}}
        self.pause_data = {}        # {area: {object_id: [list of (pause_time, resume_time)]}}
        self.total_paused_time = {} # {area: {object_id: cumulative_paused_time}}
        self.object_frame_counts = {} # {area: {object_id: frame_count}}
        
        # Timer statistics
        self.area_stats = {}        # {area: {'count': 0, 'total_duration': 0, ...}}
        
        # Duration thresholds for color coding
        self.duration_thresholds = [
            (10, (0, 255, 0)),     # Green for < 10s
            (30, (0, 255, 255)),   # Yellow for < 30s
            (60, (0, 165, 255)),   # Orange for < 60s
            (float('inf'), (0, 0, 255))  # Red for >= 60s
        ]
        
        # Initialize log file if logging is enabled
        if self.enable_logging:
            with open(self.log_file, 'w') as f:
                f.write(f"Timer System Log - Started at {datetime.now()}\n")
                f.write(f"FPS: {self.fps}\n\n")
    
    def initialize_area(self, area_name):
        """Initialize data structures for a new checkpoint area"""
        if area_name not in self.timer_start_time:
            self.timer_start_time[area_name] = {}
            self.active_timers[area_name] = {}
            self.pause_data[area_name] = {}
            self.total_paused_time[area_name] = {}
            self.object_frame_counts[area_name] = {}
            self.area_stats[area_name] = {
                'count': 0, 
                'total_duration': 0, 
                'avg_duration': 0, 
                'max_duration': 0,
                'total_counts': 0, 
                'avg_seconds_from_counts': 0,
                'min_duration': float('inf'),
                'durations': []  # Store all durations for percentile calculations
            }
    
    def mark_stationary(self, object_id, current_time):
        """Mark an object as initially stationary"""
        self.stationary_since[object_id] = current_time
        self._log(f"Object {object_id} marked initially stationary at {self._format_timestamp(current_time)}")
    
    def check_if_stationary_long_enough(self, object_id, current_time, threshold):
        """Check if object has been stationary long enough to start timer"""
        if object_id in self.stationary_since:
            return current_time - self.stationary_since[object_id] >= threshold
        return False
    
    def reset_stationary(self, object_id):
        """Reset stationary status when object moves"""
        if object_id in self.stationary_since:
            self._log(f"Object {object_id} is moving - reset stationary status")
            del self.stationary_since[object_id]
    
    def start_timer(self, area_name, object_id, current_time):
        """Start a timer for an object in a checkpoint area"""
        self.initialize_area(area_name)
        self.timer_start_time[area_name][object_id] = current_time
        self.active_timers[area_name][object_id] = True
        self.pause_data[area_name][object_id] = []
        self.total_paused_time[area_name][object_id] = 0
        self._log(f"Started timer for object {object_id} in {area_name} at {self._format_timestamp(current_time)}")
    
    def pause_timer(self, area_name, object_id, current_time):
        """Pause a timer (when object briefly moves)"""
        if (area_name in self.active_timers and 
            object_id in self.active_timers[area_name] and 
            self.active_timers[area_name][object_id]):
            
            self.active_timers[area_name][object_id] = False
            self.pause_data[area_name][object_id].append((current_time, None))
            self._log(f"Paused timer for object {object_id} in {area_name} at {self._format_timestamp(current_time)}")
    
    def resume_timer(self, area_name, object_id, current_time):
        """Resume a timer"""
        if (area_name in self.pause_data and 
            object_id in self.pause_data[area_name] and 
            self.pause_data[area_name][object_id] and 
            self.pause_data[area_name][object_id][-1][1] is None):
            
            pause_time = self.pause_data[area_name][object_id][-1][0]
            self.pause_data[area_name][object_id][-1] = (pause_time, current_time)
            pause_duration = current_time - pause_time
            self.total_paused_time[area_name][object_id] += pause_duration
            
            self.active_timers[area_name][object_id] = True
            self._log(f"Resumed timer for object {object_id} in {area_name} at {self._format_timestamp(current_time)}")
            self._log(f"Pause duration: {self._format_duration(pause_duration)}")
    
    def end_timer(self, area_name, object_id, current_time, frame_count, video_start_time):
        """End a timer and calculate statistics"""
        if (area_name in self.timer_start_time and 
            object_id in self.timer_start_time[area_name]):
            
            start_time = self.timer_start_time[area_name][object_id]
            raw_duration = current_time - start_time
            
            # Adjust for paused time
            if area_name in self.total_paused_time and object_id in self.total_paused_time[area_name]:
                adjusted_duration = raw_duration - self.total_paused_time[area_name][object_id]
            else:
                adjusted_duration = raw_duration
            
            # Get frame counts
            frame_counts = self.object_frame_counts[area_name].get(object_id, 0)
            seconds_from_counts = self.counts_to_seconds(frame_counts)
            
            # Extract class name from object_id (format: class_name_x_y)
            class_name = object_id.split('_')[0]
            
            # Update area statistics
            self.area_stats[area_name]['count'] += 1
            self.area_stats[area_name]['total_duration'] += adjusted_duration
            self.area_stats[area_name]['avg_duration'] = (
                self.area_stats[area_name]['total_duration'] / self.area_stats[area_name]['count']
            )
            self.area_stats[area_name]['max_duration'] = max(
                self.area_stats[area_name]['max_duration'], adjusted_duration
            )
            self.area_stats[area_name]['min_duration'] = min(
                self.area_stats[area_name]['min_duration'], adjusted_duration
            )
            self.area_stats[area_name]['durations'].append(adjusted_duration)
            
            # Update count statistics
            self.area_stats[area_name]['total_counts'] += frame_counts
            self.area_stats[area_name]['avg_seconds_from_counts'] = (
                self.counts_to_seconds(self.area_stats[area_name]['total_counts']) / 
                self.area_stats[area_name]['count']
                if self.area_stats[area_name]['count'] > 0 else 0
            )
            
            # Log the timer end
            self._log(f"Ended timer for object {object_id} in {area_name}:")
            self._log(f"  - Start time: {self._format_timestamp(start_time)}")
            self._log(f"  - End time: {self._format_timestamp(current_time)}")
            self._log(f"  - Raw duration: {self._format_duration(raw_duration)}")
            self._log(f"  - Adjusted duration: {self._format_duration(adjusted_duration)}")
            self._log(f"  - Frame counts: {frame_counts} ({seconds_from_counts:.2f}s)")
            
            # Deactivate the timer
            self.active_timers[area_name][object_id] = False
            
            # Reset frame count
            self.object_frame_counts[area_name][object_id] = 0
            
            # Return the record for this stop
            return {
                'object_id': class_name,
                'area': area_name,
                'entry_time': start_time - video_start_time,  # Relative time
                'exit_time': current_time - video_start_time,  # Relative time
                'raw_duration': raw_duration,
                'adjusted_duration': adjusted_duration,
                'paused_time': self.total_paused_time[area_name].get(object_id, 0),
                'counts': frame_counts,
                'seconds_from_counts': seconds_from_counts
            }
        
        return None
    
    def increment_frame_count(self, area_name, object_id):
        """Increment frame count for an object in an area"""
        self.initialize_area(area_name)
        if object_id not in self.object_frame_counts[area_name]:
            self.object_frame_counts[area_name][object_id] = 0
        self.object_frame_counts[area_name][object_id] += 1
    
    def get_timer_duration(self, area_name, object_id, current_time):
        """Get the current duration of a timer"""
        if (area_name in self.timer_start_time and 
            object_id in self.timer_start_time[area_name]):
            
            raw_duration = current_time - self.timer_start_time[area_name][object_id]
            
            # Adjust for paused time
            if area_name in self.total_paused_time and object_id in self.total_paused_time[area_name]:
                adjusted_duration = raw_duration - self.total_paused_time[area_name][object_id]
            else:
                adjusted_duration = raw_duration
                
            return adjusted_duration
        
        return 0
    
    def get_stationary_wait_time(self, object_id, current_time):
        """Get how long an object has been waiting to be considered stationary"""
        if object_id in self.stationary_since:
            return current_time - self.stationary_since[object_id]
        return 0
    
    def get_active_timer_count(self, area_name):
        """Get count of active timers in an area"""
        if area_name in self.active_timers:
            return sum(1 for obj_id, is_active in self.active_timers[area_name].items() if is_active)
        return 0
    
    def get_timer_color(self, duration):
        """Get color for timer based on duration thresholds"""
        for threshold, color in self.duration_thresholds:
            if duration < threshold:
                return color
        return self.duration_thresholds[-1][1]  # Return the last color
    
    def get_percentile_statistics(self, area_name, percentiles=[50, 75, 90, 95]):
        """Calculate percentile statistics for an area"""
        if area_name in self.area_stats and self.area_stats[area_name]['durations']:
            durations = sorted(self.area_stats[area_name]['durations'])
            result = {}
            for p in percentiles:
                idx = int(len(durations) * p / 100)
                if idx >= len(durations):
                    idx = len(durations) - 1
                result[f'p{p}'] = durations[idx]
            return result
        return {f'p{p}': 0 for p in percentiles}
    
    def save_timer_state(self, filename="timer_state.pkl"):
        """Save timer state to file for later resumption"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'stationary_since': self.stationary_since,
                'timer_start_time': self.timer_start_time,
                'active_timers': self.active_timers,
                'pause_data': self.pause_data,
                'total_paused_time': self.total_paused_time,
                'object_frame_counts': self.object_frame_counts,
                'area_stats': self.area_stats
            }, f)
        self._log(f"Timer state saved to {filename}")
    
    def load_timer_state(self, filename="timer_state.pkl"):
        """Load timer state from file"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                state = pickle.load(f)
                self.stationary_since = state['stationary_since']
                self.timer_start_time = state['timer_start_time']
                self.active_timers = state['active_timers']
                self.pause_data = state['pause_data']
                self.total_paused_time = state['total_paused_time']
                self.object_frame_counts = state['object_frame_counts']
                self.area_stats = state['area_stats']
            self._log(f"Timer state loaded from {filename}")
            return True
        return False
    
    def counts_to_seconds(self, counts):
        """Convert frame counts to seconds based on video FPS"""
        return counts / self.fps if self.fps > 0 else 0
    
    def _format_timestamp(self, timestamp):
        """Format a Unix timestamp to readable datetime"""
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    
    def _format_duration(self, seconds):
        """Format duration in seconds to minutes:seconds.milliseconds format"""
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        ms = int((seconds - int(seconds)) * 1000)
        return f"{minutes}m {secs}s {ms}ms"
    
    def _log(self, message):
        """Log a message if logging is enabled"""
        if self.enable_logging:
            with open(self.log_file, 'a') as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {message}\n")
    
    def draw_timer(self, frame, area_name, object_id, duration, x, y, frame_count=0, seconds_from_counts=0, display_mode="compact"):
        """Draw timer information on the frame
        
        Args:
            frame: The video frame to draw on
            area_name: Name of the checkpoint area
            object_id: ID of the tracked object
            duration: Current timer duration in seconds
            x, y: Coordinates to place the timer display
            frame_count: Number of frames object has been in area
            seconds_from_counts: Seconds calculated from frame count
            display_mode: "compact" or "full" display mode
        """
        # Get color based on duration
        color = self.get_timer_color(duration)
        
        # Format duration as MM:SS
        duration_str = self._format_duration(duration)
        
        # Get class name from object_id
        class_name = object_id.split('_')[0]
        
        # Format timer text
        if display_mode == "compact":
            timer_text = f"{class_name}: {duration_str}"
        else:
            # Full display mode
            timer_text = f"{area_name} - {class_name}: {duration_str}"
            counts_text = f"Frames: {frame_count} (~{seconds_from_counts:.1f}s)"
            
            # Draw background rectangle for better readability
            text_size = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y-25), (x + text_size[0], y+5), (0, 0, 0), -1)
            
            # Draw timer text
            cv2.putText(frame, timer_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw frame count text below timer
            cv2.putText(frame, counts_text, (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            return
        
        # Compact mode display
        cv2.putText(frame, timer_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)



# ========== VISUALIZATION SYSTEM ==========
class VisualizationSystem:
    """Enhanced visualization system for better UI and timer displays"""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.text_color = (255, 255, 255)
        self.text_thickness = 2
        
        # Define UI regions
        self.header_height = 50
        self.sidebar_width = 250
        self.footer_height = 130
        
    def draw_ui_framework(self, frame):
        """Draw basic UI framework on frame"""
        # Draw semi-transparent overlay for header
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.header_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw semi-transparent overlay for sidebar
        overlay = frame.copy()
        cv2.rectangle(overlay, (self.width - self.sidebar_width, self.header_height), 
                    (self.width, self.height - self.footer_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw semi-transparent overlay for footer
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, self.height - self.footer_height), 
                    (self.width, self.height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        return frame
    
    def draw_header(self, frame, processing_fps, current_time, video_start_time):
        """Draw header with processing info and current time"""
        # Draw processing speed
        cv2.putText(frame, f"Processing: {processing_fps:.1f} FPS", 
                  (20, 35), self.font, 1, (0, 255, 255), 2)
        
        # Draw current time
        relative_time = current_time - video_start_time
        time_text = self.format_time(relative_time)
        cv2.putText(frame, f"Time: {time_text}", 
                  (self.width - 250, 35), self.font, 1, (0, 255, 255), 2)
        
        return frame
    
    def draw_checkpoints(self, frame, checkpoint_areas, area_colors, current_detections):
        """Draw checkpoint areas with active detection count"""
        for name, coords in checkpoint_areas.items():
            # Ambil koordinat area
            x1, y1, x2, y2 = coords
            
            # Ambil warna area
            color = area_colors.get(name, (0, 255, 0))  # Default hijau jika tidak diatur
            
            # Cek apakah area ini memiliki deteksi aktif
            has_detections = False
            detection_count = 0
            
            # Periksa tipe current_detections[name]
            if current_detections:
                if name in current_detections:
                    # Jika current_detections[name] adalah set atau list
                    if isinstance(current_detections[name], (set, list)):
                        detection_count = len(current_detections[name])
                        has_detections = detection_count > 0
                    # Jika current_detections[name] adalah int (penghitung)
                    elif isinstance(current_detections[name], int):
                        detection_count = current_detections[name]
                        has_detections = detection_count > 0
            
            # Gambar persegi panjang untuk area
            thickness = 3 if has_detections else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Tambahkan label area dengan jumlah deteksi
            label = f"{name}: {detection_count} objects"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Latar belakang label
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_size[0] + 10, y1), color, -1)
            
            # Text label
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
        return frame

    
    def draw_sidebar_stats(self, frame, timer_system):
        """Draw statistics in sidebar"""
        # Header
        cv2.putText(frame, "STOP STATISTICS", 
                  (self.width - self.sidebar_width + 10, self.header_height + 30), 
                  self.font, 0.7, (255, 255, 255), 2)
        
        # Draw line separator
        cv2.line(frame, 
                (self.width - self.sidebar_width + 10, self.header_height + 40),
                (self.width - 10, self.header_height + 40),
                (200, 200, 200), 1)
        
        y_offset = self.header_height + 60
        
        # Show statistics for each area
        for area_name, stats in timer_system.area_stats.items():
            if stats['count'] > 0:
                # Area name
                cv2.putText(frame, area_name, 
                          (self.width - self.sidebar_width + 10, y_offset), 
                          self.font, 0.7, (0, 255, 255), 2)
                y_offset += 25
                
                # Count
                cv2.putText(frame, f"Stops: {stats['count']}", 
                          (self.width - self.sidebar_width + 20, y_offset), 
                          self.font, 0.5, self.text_color, 1)
                y_offset += 20
                
                # Average duration
                avg_text = f"Avg: {self.format_time(stats['avg_duration'])}"
                cv2.putText(frame, avg_text, 
                          (self.width - self.sidebar_width + 20, y_offset), 
                          self.font, 0.5, self.text_color, 1)
                y_offset += 20
                
                # Min/Max duration
                if stats['min_duration'] < float('inf'):
                    min_text = f"Min: {self.format_time(stats['min_duration'])}"
                    cv2.putText(frame, min_text, 
                              (self.width - self.sidebar_width + 20, y_offset), 
                              self.font, 0.5, self.text_color, 1)
                    y_offset += 20
                
                max_text = f"Max: {self.format_time(stats['max_duration'])}"
                cv2.putText(frame, max_text, 
                          (self.width - self.sidebar_width + 20, y_offset), 
                          self.font, 0.5, self.text_color, 1)
                y_offset += 20
                
                # Percentile statistics
                percentiles = timer_system.get_percentile_statistics(area_name)
                for p, value in percentiles.items():
                    pct_text = f"{p}: {self.format_time(value)}"
                    cv2.putText(frame, pct_text, 
                              (self.width - self.sidebar_width + 20, y_offset), 
                              self.font, 0.5, self.text_color, 1)
                    y_offset += 20
                
                # Add spacing between areas
                y_offset += 10
            else:
                cv2.putText(frame, f"{area_name}: No stops", 
                          (self.width - self.sidebar_width + 10, y_offset), 
                          self.font, 0.6, (0, 165, 255), 2)
                y_offset += 30
        
        return frame
    
    def draw_footer(self, frame, active_timers_by_area):
        """Draw footer with active timers summary"""
        y_offset = self.height - self.footer_height + 30
        
        # Header
        cv2.putText(frame, "ACTIVE TIMERS", (20, y_offset), 
                  self.font, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        # Active timers by area
        for area_name, active_count in active_timers_by_area.items():
            # Text color based on status (orange or green)
            text_color = (0, 255, 0) if active_count > 0 else (0, 165, 255)
            cv2.putText(frame, f"{area_name}: {active_count} active", 
                      (20, y_offset), self.font, 0.6, text_color, 2)
            y_offset += 25
        
        return frame
    
    def draw_stationary_wait(self, frame, object_id, wait_time, threshold, x1, y1):
        """Draw waiting status for objects becoming stationary"""
        if wait_time < threshold:
            # Calculate progress percentage
            progress = wait_time / threshold * 100
            
            # Status text
            status_text = f"Waiting: {wait_time:.1f}/{threshold:.1f}s ({progress:.0f}%)"
            
            # Background for text
            text_size = cv2.getTextSize(status_text, self.font, 0.6, 2)[0]
            cv2.rectangle(frame, (x1-5, y1-text_size[1]-10), (x1+text_size[0]+5, y1), 
                        (0, 0, 0), -1)
            
            # Progress color (gradient from red to green)
            r = int(255 * (1 - progress/100))
            g = int(255 * (progress/100))
            color = (0, g, r)  # BGR
            
            # Draw text
            cv2.putText(frame, status_text, (x1, y1-5), 
                      self.font, 0.6, color, 2)
            
            # Draw progress bar
            bar_width = 100
            bar_height = 5
            
            # Background bar
            cv2.rectangle(frame, (x1, y1-text_size[1]-20), (x1+bar_width, y1-text_size[1]-15), 
                        (100, 100, 100), -1)
            
            # Progress bar
            progress_width = int(bar_width * progress / 100)
            cv2.rectangle(frame, (x1, y1-text_size[1]-20), (x1+progress_width, y1-text_size[1]-15), 
                        color, -1)
        
        return frame
    
    def draw_object_highlight(self, frame, x1, y1, x2, y2, color, thickness=3):
        """Draw highlighted bounding box for tracked objects"""
        # Draw main rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw corner highlights for emphasis
        corner_length = 20
        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, thickness + 1)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, thickness + 1)
        
        # Top-right corner
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, thickness + 1)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, thickness + 1)
        
        # Bottom-left corner
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, thickness + 1)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, thickness + 1)
        
        # Bottom-right corner
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, thickness + 1)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, thickness + 1)
        
        return frame
    
    @staticmethod
    def format_time(seconds):
        """Format seconds into MM:SS.MS format"""
        minutes = int(seconds) // 60
        seconds_part = int(seconds) % 60
        milliseconds = int((seconds - int(seconds)) * 100)
        return f"{minutes:02d}:{seconds_part:02d}.{milliseconds:02d}"


# ========== DATA LOGGING SYSTEM ==========
class DataLogger:
    """Data logging system for storing and retrieving timer data"""
    
    def __init__(self, session_id=None):
        self.session_id = session_id or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logs_dir = 'tracking_logs'
        self.ensure_log_dir()
        
        # Filenames
        self.raw_data_file = os.path.join(self.logs_dir, f"stop_data_{self.session_id}.csv")
        self.summary_file = os.path.join(self.logs_dir, f"summary_{self.session_id}.txt")
        self.checkpoint_data = {}  # {area_name: [records]}
        
        # Initialize CSV file with headers
        with open(self.raw_data_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'object_id', 'area', 'entry_time', 'exit_time', 
                'raw_duration', 'adjusted_duration', 'paused_time', 'counts',
                'seconds_from_counts'
            ])
    
    def ensure_log_dir(self):
        """Make sure log directory exists"""
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
    
    def log_stop(self, record):
        """Log a completed stop to CSV file"""
        with open(self.raw_data_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                record['object_id'],
                record['area'],
                self.format_time_for_log(record['entry_time']),
                self.format_time_for_log(record['exit_time']),
                record['raw_duration'],
                record['adjusted_duration'],
                record['paused_time'],
                record['counts'],
                record['seconds_from_counts']
            ])
        
        # Store record for summary
        if record['area'] not in self.checkpoint_data:
            self.checkpoint_data[record['area']] = []
        self.checkpoint_data[record['area']].append(record)
    
    def generate_summary(self, timer_system):
        """Generate summary of all stops and save to file"""
        with open(self.summary_file, 'w') as f:
            f.write(f"TRACKING SESSION SUMMARY - {self.session_id}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("CHECKPOINT STATISTICS\n")
            f.write("====================\n\n")
            
            for area_name, stats in timer_system.area_stats.items():
                if stats['count'] > 0:
                    f.write(f"Checkpoint: {area_name}\n")
                    f.write(f"  Total stops: {stats['count']}\n")
                    f.write(f"  Average duration: {self.format_seconds(stats['avg_duration'])}\n")
                    
                    if stats['min_duration'] < float('inf'):
                        f.write(f"  Minimum duration: {self.format_seconds(stats['min_duration'])}\n")
                    
                    f.write(f"  Maximum duration: {self.format_seconds(stats['max_duration'])}\n")
                    
                    # Percentile statistics
                    percentiles = timer_system.get_percentile_statistics(area_name)
                    f.write("  Percentiles:\n")
                    for p, value in percentiles.items():
                        f.write(f"    {p}: {self.format_seconds(value)}\n")
                    
                    f.write(f"  Total frames counted: {stats['total_counts']}\n")
                    f.write(f"  Avg time from counts: {self.format_seconds(stats['avg_seconds_from_counts'])}\n")
                    f.write("\n")
                else:
                    f.write(f"Checkpoint: {area_name} - No stops recorded\n\n")
        
        print(f"Summary saved to {self.summary_file}")
    
    def save_checkpoint_data(self):
        """Save checkpoint data to pickle file for later analysis"""
        checkpoint_file = os.path.join(self.logs_dir, f"checkpoint_data_{self.session_id}.pkl")
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(self.checkpoint_data, f)
        print(f"Checkpoint data saved to {checkpoint_file}")
    
    @staticmethod
    def format_time_for_log(seconds):
        """Format seconds for log files"""
        minutes = int(seconds) // 60
        seconds_part = seconds % 60
        return f"{minutes:02d}:{seconds_part:06.3f}"
    
    @staticmethod
    def format_seconds(seconds):
        """Format seconds into human-readable format"""
        minutes = int(seconds) // 60
        seconds_part = int(seconds) % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{minutes:02d}:{seconds_part:02d}.{milliseconds:03d}"


# ========== OBJECT TRACKING SYSTEM ==========
class ObjectTracker:
    """Enhanced object tracking system with robust tracking and object identity preservation"""
    
    def __init__(self, model_path, fps, stationary_threshold=3.0, area_names=None):
        # Initialize YOLO model
        self.model = YOLO(model_path)
        self.fps = fps
        self.stationary_threshold = stationary_threshold
        
        # Tracking data structures
        self.tracked_objects = {}  # {track_id: {position_history, class_name, etc.}}
        self.track_id_counter = 0
        self.position_history_len = int(fps * 2)  # 2 seconds of history
        self.position_variance_threshold = 100  # pixels^2
        
        # Object ID mapping
        self.object_ids = {}  # {track_id: unique_object_id}
        
        # Checkpoint areas
        self.checkpoint_areas = {}
        self.area_colors = {}
        if area_names:
            # We'll define the actual areas later when the frame size is known
            for i, name in enumerate(area_names):
                self.area_colors[name] = (
                    (0, 255, 0),     # Green
                    (0, 255, 255),   # Yellow
                    (0, 165, 255),   # Orange
                    (0, 0, 255),     # Red
                    (255, 0, 0),     # Blue
                    (255, 0, 255),   # Magenta
                )[i % 6]  # Cycle through colors
        
        # Current detections in areas
        self.current_area_detections = {}  # {area_name: {object_ids}}
        
        # Movement detection
        self.movement_threshold = 5  # pixels per frame
    
    def setup_default_areas(self, frame_width, frame_height):
        """Set up default checkpoint areas if not manually specified"""
        if not self.checkpoint_areas:
            if len(self.area_colors) > 0:
                # Divide the frame into equal areas based on the provided names
                num_areas = len(self.area_colors)
                area_width = frame_width // num_areas
                
                for i, name in enumerate(self.area_colors.keys()):
                    x1 = i * area_width
                    y1 = frame_height // 4
                    x2 = (i + 1) * area_width
                    y2 = frame_height * 3 // 4
                    self.checkpoint_areas[name] = (x1, y1, x2, y2)
                    self.current_area_detections[name] = set()
            else:
                # Create a single default checkpoint area in the center
                x1 = frame_width // 4
                y1 = frame_height // 4
                x2 = frame_width * 3 // 4
                y2 = frame_height * 3 // 4
                self.checkpoint_areas["Default"] = (x1, y1, x2, y2)
                self.current_area_detections["Default"] = set()
                self.area_colors["Default"] = (0, 255, 0)  # Green
    
    def detect_and_track(self, frame, timer_system, visualizer, current_time):
        """Detect objects and track them with enhanced accuracy"""
        # Clear current detections
        for area in self.current_area_detections:
            self.current_area_detections[area] = set()
        
        # Run YOLO detection
        results = self.model.track(frame, persist=True, verbose=False)
        
        # Process detections
        if results[0].boxes:
            boxes = results[0].boxes.xywh.cpu().numpy()
            if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().numpy()
                classes = results[0].boxes.cls.int().cpu().numpy()
                
                for i, (x, y, w, h) in enumerate(boxes):
                    track_id = int(track_ids[i])
                    class_id = int(classes[i])
                    class_name = results[0].names[class_id]
                    
                    # Get center coordinates
                    center_x, center_y = int(x), int(y)
                    box_x1, box_y1 = int(x - w/2), int(y - h/2)
                    box_x2, box_y2 = int(x + w/2), int(y + h/2)
                    
                    # Create or update tracked object
                    if track_id not in self.tracked_objects:
                        self.tracked_objects[track_id] = {
                            'class_name': class_name,
                            'position_history': [],
                            'last_seen': current_time,
                            'is_stationary': False,
                            'current_area': None
                        }
                        
                        # Create unique object ID
                        if track_id not in self.object_ids:
                            self.object_ids[track_id] = f"{class_name}_{track_id}"
                    
                    # Update position history
                    position = (center_x, center_y)
                    history = self.tracked_objects[track_id]['position_history']
                    history.append(position)
                    if len(history) > self.position_history_len:
                        history.pop(0)
                    
                    # Update last seen time
                    self.tracked_objects[track_id]['last_seen'] = current_time
                    
                    # Check for motion
                    is_moving = True
                    
                    if len(history) >= 5:  # Need enough history to calculate variance
                        # Calculate position variance to determine if object is stationary
                        x_coords = [p[0] for p in history[-5:]]
                        y_coords = [p[1] for p in history[-5:]]
                        x_variance = sum((x - sum(x_coords)/len(x_coords))**2 for x in x_coords) / len(x_coords)
                        y_variance = sum((y - sum(y_coords)/len(y_coords))**2 for y in y_coords) / len(y_coords)
                        total_variance = x_variance + y_variance
                        
                        is_moving = total_variance > self.position_variance_threshold
                    
                    # Handle object's area and timer state
                    object_id = self.object_ids[track_id]
                    
                    # Check if object is in any checkpoint area
                    in_area = False
                    
                    for area_name, (area_x1, area_y1, area_x2, area_y2) in self.checkpoint_areas.items():
                        # Check if the center of the object is inside the area
                        if area_x1 <= center_x <= area_x2 and area_y1 <= center_y <= area_y2:
                            in_area = True
                            
                            # Mark this object as detected in this area
                            self.current_area_detections[area_name].add(object_id)
                            
                            # Update current area
                            self.tracked_objects[track_id]['current_area'] = area_name
                            
                            # If moving, reset stationary status
                            if is_moving:
                                timer_system.reset_stationary(object_id)
                                
                                # If timer was active, pause it
                                if (area_name in timer_system.active_timers and 
                                    object_id in timer_system.active_timers[area_name] and 
                                    timer_system.active_timers[area_name][object_id]):
                                    timer_system.pause_timer(area_name, object_id, current_time)
                            else:
                                # If not moving, update stationary status
                                if object_id not in timer_system.stationary_since:
                                    timer_system.mark_stationary(object_id, current_time)
                                    
                                # Check if object has been stationary long enough to start/resume timer
                                stationary_long_enough = timer_system.check_if_stationary_long_enough(
                                    object_id, current_time, self.stationary_threshold
                                )
                                
                                if stationary_long_enough:
                                    # If timer not started, start it
                                    if (area_name not in timer_system.timer_start_time or 
                                        object_id not in timer_system.timer_start_time[area_name]):
                                        timer_system.start_timer(area_name, object_id, current_time)
                                    
                                    # If timer was paused, resume it
                                    elif (area_name in timer_system.active_timers and 
                                          object_id in timer_system.active_timers[area_name] and 
                                          not timer_system.active_timers[area_name][object_id]):
                                        timer_system.resume_timer(area_name, object_id, current_time)
                                    
                                    # Increment frame count for active timers
                                    if (area_name in timer_system.active_timers and 
                                        object_id in timer_system.active_timers[area_name] and 
                                        timer_system.active_timers[area_name][object_id]):
                                        timer_system.increment_frame_count(area_name, object_id)
                                        
                                # If not stationary long enough, show waiting progress
                                else:
                                    wait_time = timer_system.get_stationary_wait_time(object_id, current_time)
                                    visualizer.draw_stationary_wait(
                                        frame, object_id, wait_time, self.stationary_threshold, box_x1, box_y1 - 30
                                    )
                            
                            # Draw timer if exists
                            if (area_name in timer_system.timer_start_time and 
                                object_id in timer_system.timer_start_time[area_name]):
                                
                                duration = timer_system.get_timer_duration(area_name, object_id, current_time)
                                is_active = False
                                
                                if (area_name in timer_system.active_timers and 
                                    object_id in timer_system.active_timers[area_name]):
                                    is_active = timer_system.active_timers[area_name][object_id]
                                
                                # Get frame count
                                frame_counts = 0
                                if (area_name in timer_system.object_frame_counts and 
                                    object_id in timer_system.object_frame_counts[area_name]):
                                    frame_counts = timer_system.object_frame_counts[area_name][object_id]
                                
                                # Calculate seconds from counts
                                seconds_from_counts = timer_system.counts_to_seconds(frame_counts)
                                
                                # Get color based on duration
                                color = timer_system.get_timer_color(duration)
                                
                                # Draw the timer
                                timer_system.draw_timer(frame, area_name, object_id, duration, 
                                                     box_x1, box_y1 - 60, frame_counts, seconds_from_counts, 
                                                     "full")
                    
                    # If not in any area but was previously in an area, end the timer
                    if not in_area and self.tracked_objects[track_id]['current_area'] is not None:
                        area_name = self.tracked_objects[track_id]['current_area']
                        
                        # Check if timer exists for this object
                        if (area_name in timer_system.timer_start_time and 
                            object_id in timer_system.timer_start_time[area_name]):
                            
                            # Get frame count
                            frame_counts = 0
                            if (area_name in timer_system.object_frame_counts and 
                                object_id in timer_system.object_frame_counts[area_name]):
                                frame_counts = timer_system.object_frame_counts[area_name][object_id]
                            
                            # End the timer
                            record = timer_system.end_timer(
                                area_name, object_id, current_time, frame_counts, 0
                            )
                            
                            # Reset current area
                            self.tracked_objects[track_id]['current_area'] = None
                            
                            # Return the record if timer ended
                            return record
                    
                    # Highlight the object
                    color = (0, 255, 0)  # Default: green
                    
                    # Color based on timer status if in an area
                    if self.tracked_objects[track_id]['current_area'] is not None:
                        area_name = self.tracked_objects[track_id]['current_area']
                        
                        if (area_name in timer_system.timer_start_time and 
                            object_id in timer_system.timer_start_time[area_name]):
                            
                            duration = timer_system.get_timer_duration(area_name, object_id, current_time)
                            color = timer_system.get_timer_color(duration)
                    
                    # Draw object highlight
                    visualizer.draw_object_highlight(frame, box_x1, box_y1, box_x2, box_y2, color)
                    
                    # Add class name and ID
                    label = f"{class_name} {track_id}"
                    cv2.putText(frame, label, (box_x1, box_y1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return None  # No timer completed on this frame


# ========== MAIN APPLICATION ==========
def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Enhanced Robot Tracker')
    parser.add_argument('--source', type=str, default='C:/Users/YMPI/Downloads/PuduVideo2/20250412140122.ts', help='Video source (0 for webcam, or path to video file)')
    parser.add_argument('--model', type=str, default='D:/DATA/FOR_DL/objectdetection/magang2025/trainedPuduRobot_yolov8n.pt', help='Path to YOLO model')
    parser.add_argument('--areas', nargs='+', default=['Zone1', 'Zone2'], help='Names of checkpoint areas')
    parser.add_argument('--fps', type=float, default=30.0, help='Video FPS (used for timing calculations)')
    parser.add_argument('--threshold', type=float, default=3.0, help='Stationary threshold in seconds')
    args = parser.parse_args()
    
    # Initialize video source
    try:
        if args.source.isdigit():
            source = int(args.source)
        else:
            source = args.source
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise Exception(f"Could not open video source: {args.source}")
    except Exception as e:
        print(f"Error opening video source: {e}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = args.fps
    
    # Initialize systems
    timer_system = TimerSystem(fps)
    visualizer = VisualizationSystem(width, height)
    tracker = ObjectTracker(args.model, fps, args.threshold, args.areas)
    data_logger = DataLogger()
    
    # Setup default areas based on frame size
    tracker.setup_default_areas(width, height)

    # Tambahkan kode ini setelah pemanggilan tracker.setup_default_areas(width, height)
    tracker.checkpoint_areas = {
        'Area1': [122, 262, 437, 875],
        'Area2': [754, 630, 1267, 982],
        'Area3': [900, 144, 1226, 580]
    }

    # Update warna area jika diperlukan
    tracker.area_colors = {
        'Area1': (0, 255, 0),      # Hijau
        'Area2': (0, 255, 0),    # Oranye
        'Area3': (0, 255, 0)       # Biru
    }

    # Perbarui penghitung deteksi untuk area yang baru
    tracker.current_area_detections = {area: 0 for area in tracker.checkpoint_areas}

    
    # FPS calculation variables
    fps_start_time = time.time()
    fps_frame_count = 0
    processing_fps = 0
    
    # Timing variables
    video_start_time = time.time()
    
    print(f"Starting tracking with model: {args.model}")
    print(f"Checkpoint areas: {list(tracker.checkpoint_areas.keys())}")
    print(f"Press 'q' to quit, 's' to save a summary")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame")
            break
        
        # Get current time
        current_time = time.time()
        
        # Calculate FPS
        fps_frame_count += 1
        if current_time - fps_start_time >= 1:
            processing_fps = fps_frame_count / (current_time - fps_start_time)
            fps_frame_count = 0
            fps_start_time = current_time
        
        # Draw UI framework
        frame = visualizer.draw_ui_framework(frame)
        
        # Draw checkpoint areas
        frame = visualizer.draw_checkpoints(frame, tracker.checkpoint_areas, 
                                          tracker.area_colors, tracker.current_area_detections)
        
        # Detect and track objects
        record = tracker.detect_and_track(frame, timer_system, visualizer, current_time)
        
        # If a timer completed, log it
        if record:
            data_logger.log_stop(record)
        
        # Get active timer counts for display
        active_timers_by_area = {
            area: timer_system.get_active_timer_count(area)
            for area in tracker.checkpoint_areas.keys()
        }
        
        # Draw header
        frame = visualizer.draw_header(frame, processing_fps, current_time, video_start_time)
        
        # Draw sidebar stats
        frame = visualizer.draw_sidebar_stats(frame, timer_system)
        
        # Draw footer with active timer summary
        frame = visualizer.draw_footer(frame, active_timers_by_area)

        display_frame = cv2.resize(frame, (1920, 900))  # Atur ukuran sesuai kebutuhan
        
        # Show the frame
        cv2.imshow('Enhanced Robot Tracker', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s'):
            print("Saving summary...")
            data_logger.generate_summary(timer_system)
            data_logger.save_checkpoint_data()
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Generate final summary
    data_logger.generate_summary(timer_system)
    data_logger.save_checkpoint_data()
    
    print("Tracking ended. Summary saved.")


if __name__ == '__main__':
    main()

       
