

import os
import yt_dlp
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import re
from urllib.parse import urlparse, parse_qs
import time
from datetime import datetime, timedelta
import uuid
import asyncio
from typing import Dict, Optional, Any
import logging
from enum import Enum
import tempfile
import shutil
import threading
import glob
from concurrent.futures import ThreadPoolExecutor
import weakref
import psutil
import signal
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Task status enum
class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

# ✅ ENHANCED TASK MANAGER FOR LONG VIDEOS
class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self.cleanup_interval = 7200  # 2 hours for long videos
        self._lock = threading.RLock()
        self.max_concurrent_downloads = 3  # Limit concurrent downloads
        self.active_downloads = 0
        
    def create_task(self, task_id: str, url: str, format_id: str, quality: str) -> str:
        """Create a new download task"""
        with self._lock:
            self.tasks[task_id] = {
                "id": task_id,
                "url": url,
                "format_id": format_id,
                "quality": quality,
                "status": TaskStatus.PENDING,
                "progress": 0,
                "message": "Task created",
                "filename": None,
                "download_url": None,
                "error": None,
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "download_speed": "0 B/s",
                "eta": "Unknown",
                "file_size": "Unknown",
                "downloaded_bytes": 0,
                "total_bytes": 0,
                "retry_count": 0,
                "max_retries": 5
            }
        return task_id
    
    def update_task(self, task_id: str, **kwargs):
        """Update task information"""
        with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id].update(kwargs)
                self.tasks[task_id]["updated_at"] = datetime.now()
    
    def get_task(self, task_id: str) -> Optional[Dict]:
        """Get task information"""
        with self._lock:
            return self.tasks.get(task_id)
    
    def can_start_download(self) -> bool:
        """Check if we can start a new download"""
        with self._lock:
            return self.active_downloads < self.max_concurrent_downloads
    
    def increment_active_downloads(self):
        """Increment active download counter"""
        with self._lock:
            self.active_downloads += 1
    
    def decrement_active_downloads(self):
        """Decrement active download counter"""
        with self._lock:
            self.active_downloads = max(0, self.active_downloads - 1)
    
    def cleanup_old_tasks(self):
        """Remove old completed/failed tasks"""
        current_time = datetime.now()
        to_remove = []
        
        with self._lock:
            for task_id, task in self.tasks.items():
                time_diff = (current_time - task["updated_at"]).total_seconds()
                if time_diff > self.cleanup_interval and task["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    to_remove.append(task_id)
            
            for task_id in to_remove:
                del self.tasks[task_id]
                logger.info(f"Cleaned up old task: {task_id}")

# Pydantic models
class URLRequest(BaseModel):
    url: str

class DownloadRequest(BaseModel):
    url: str
    format_id: str
    quality: str

# Initialize FastAPI app
app = FastAPI(
    title="YouTube Downloader API - Long Video Support",
    description="High-performance async YouTube video downloader with long video support",
    version="3.0.0"
)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="images"), name="images")

# Initialize task manager
task_manager = TaskManager()

# ✅ ENHANCED FILE MANAGER FOR LARGE FILES
class FileManager:
    def __init__(self):
        self._files = {}
        self._lock = threading.RLock()
        self.max_file_age = 3600  # 1 hour for large files
    
    def register_file(self, task_id: str, file_path: str, temp_dir: str):
        """Register a file for tracking and cleanup"""
        with self._lock:
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            self._files[task_id] = {
                'file_path': file_path,
                'temp_dir': temp_dir,
                'filename': os.path.basename(file_path),
                'created_at': datetime.now(),
                'downloaded': False,
                'file_size': file_size,
                'access_count': 0
            }
    
    def mark_downloaded(self, task_id: str):
        """Mark file as downloaded"""
        with self._lock:
            if task_id in self._files:
                self._files[task_id]['downloaded'] = True
                self._files[task_id]['access_count'] += 1
    
    def get_file_info(self, task_id: str):
        """Get file information"""
        with self._lock:
            return self._files.get(task_id)
    
    def cleanup_file(self, task_id: str):
        """Clean up a specific file"""
        with self._lock:
            if task_id in self._files:
                file_info = self._files[task_id]
                try:
                    if os.path.exists(file_info['temp_dir']):
                        shutil.rmtree(file_info['temp_dir'])
                        logger.info(f"Cleaned up temp directory: {file_info['temp_dir']}")
                except Exception as e:
                    logger.error(f"Error cleaning up {task_id}: {e}")
                finally:
                    del self._files[task_id]
    
    def cleanup_old_files(self):
        """Clean up old downloaded files"""
        current_time = datetime.now()
        to_cleanup = []
        
        with self._lock:
            for task_id, file_info in self._files.items():
                age = (current_time - file_info['created_at']).total_seconds()
                # Clean up files older than max_file_age or accessed files older than 10 minutes
                if age > self.max_file_age or (file_info['downloaded'] and age > 600):
                    to_cleanup.append(task_id)
        
        for task_id in to_cleanup:
            self.cleanup_file(task_id)

# Initialize file manager
file_manager = FileManager()
executor = ThreadPoolExecutor(max_workers=5)  # Reduced for stability

def sanitize_title(title):
    """Sanitize title for filename"""
    return re.sub(r'[\\/*?:"<>|]', "_", title)

# Create necessary directories
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

class YouTubeDownloader:
    def __init__(self):
        self.download_semaphore = asyncio.Semaphore(3)  # Limit concurrent downloads
    
    def is_valid_youtube_url(self, url):
        """Validate if the URL is a valid YouTube video URL"""
        youtube_patterns = [
            r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
            r'https?://(?:www\.)?youtube\.com/embed/[\w-]+',
            r'https?://youtu\.be/[\w-]+',
            r'https?://(?:m\.)?youtube\.com/watch\?v=[\w-]+',
            r'https?://(?:www\.)?youtube\.com/v/[\w-]+',
            r'https?://(?:www\.)?youtube\.com/shorts/[\w-]+'
        ]
        
        for pattern in youtube_patterns:
            if re.match(pattern, url):
                return True
        return False
    
    def extract_video_id(self, url):
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:shorts\/)([0-9A-Za-z_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def get_unique_filename(self, base_path, title, ext):
        """Generate unique filename to avoid conflicts"""
        safe_title = sanitize_title(title)
        base_filename = f"{safe_title}.{ext}"
        full_path = os.path.join(base_path, base_filename)
        
        if not os.path.exists(full_path):
            return base_filename
        
        counter = 1
        while True:
            new_filename = f"{safe_title}_{counter}.{ext}"
            new_full_path = os.path.join(base_path, new_filename)
            if not os.path.exists(new_full_path):
                return new_filename
            counter += 1
    
    async def extract_video_info(self, url):
        """Extract video information using yt-dlp"""
        try:
            # Clean URL
            video_id = self.extract_video_id(url)
            if video_id:
                clean_url = f"https://www.youtube.com/watch?v={video_id}"
            else:
                clean_url = url
            
            # ✅ OPTIMIZED YT-DLP OPTIONS FOR LONG VIDEOS
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'format': 'best',
                'socket_timeout': 60,
                'retries': 10,
                'fragment_retries': 10,
                'skip_unavailable_fragments': True,
                'keep_fragments': False,
                'http_chunk_size': 10485760,  # 10MB chunks
            }
            
            logger.info(f"Extracting info for: {clean_url}")
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(
                None, 
                self._extract_info_sync, 
                clean_url, 
                ydl_opts
            )
            
            logger.info(f"Video title: {info.get('title', 'Unknown')}")
            logger.info(f"Duration: {info.get('duration', 0)} seconds")
            logger.info(f"Available formats: {len(info.get('formats', []))}")
            
            # Extract video information
            duration = info.get('duration', 0)
            is_long_video = duration > 3600  # More than 1 hour
            
            video_info = {
                'title': info.get('title', 'YouTube Video'),
                'duration': duration,
                'duration_string': self._format_duration(duration),
                'is_long_video': is_long_video,
                'thumbnail': info.get('thumbnail', ''),
                'uploader': info.get('uploader', 'YouTube Channel'),
                'view_count': info.get('view_count', 0),
                'like_count': info.get('like_count', 0),
                'upload_date': info.get('upload_date', ''),
                'description': info.get('description', '')[:200] + '...' if info.get('description') else '',
                'formats': []
            }
            
            # Check available formats
            available_formats = info.get('formats', [])
            has_1080p = any(f.get('height', 0) >= 1080 for f in available_formats if f.get('vcodec') != 'none')
            has_720p = any(f.get('height', 0) >= 720 for f in available_formats if f.get('vcodec') != 'none')
            has_480p = any(f.get('height', 0) >= 480 for f in available_formats if f.get('vcodec') != 'none')
            
            logger.info(f"Available qualities - 1080p: {has_1080p}, 720p: {has_720p}, 480p: {has_480p}")
            
            formats = []
            
            # ✅ OPTIMIZED FORMAT SELECTION FOR LONG VIDEOS
            if has_1080p:
                formats.append({
                    'format_id': 'best[height>=1080]',
                    'quality': f'Ultra HD (1080p) - Best Quality {"⚠️ Large file for long videos" if is_long_video else ""}',
                    'height': 1080,
                    'width': 1920,
                    'ext': 'mp4',
                    'filesize': 0,
                    'fps': 30,
                    'has_audio': True,
                    'vcodec': 'h264',
                    'acodec': 'aac',
                    'recommended': not is_long_video,  # Not recommended for very long videos
                    'estimated_size': self._estimate_file_size(duration, 1080)
                })
            
            if has_720p:
                formats.append({
                    'format_id': 'best[height>=720]',
                    'quality': f'Full HD (720p) - High Quality {"✅ Recommended for long videos" if is_long_video else ""}',
                    'height': 720,
                    'width': 1280,
                    'ext': 'mp4',
                    'filesize': 0,
                    'fps': 30,
                    'has_audio': True,
                    'vcodec': 'h264',
                    'acodec': 'aac',
                    'recommended': is_long_video or not has_1080p,  # Recommended for long videos
                    'estimated_size': self._estimate_file_size(duration, 720)
                })
            
            if has_480p:
                formats.append({
                    'format_id': 'best[height>=480]',
                    'quality': 'HD (480p) - Good Quality ⚡ Fastest download',
                    'height': 480,
                    'width': 854,
                    'ext': 'mp4',
                    'filesize': 0,
                    'fps': 30,
                    'has_audio': True,
                    'vcodec': 'h264',
                    'acodec': 'aac',
                    'recommended': False,
                    'estimated_size': self._estimate_file_size(duration, 480)
                })
            
            # Always add best available
            formats.append({
                'format_id': 'best',
                'quality': 'Best Available Quality (Auto) 🔄 Adaptive',
                'height': 720,
                'width': 1280,
                'ext': 'mp4',
                'filesize': 0,
                'fps': 30,
                'has_audio': True,
                'vcodec': 'h264',
                'acodec': 'aac',
                'recommended': len(formats) == 0,
                'estimated_size': self._estimate_file_size(duration, 720)
            })
            
            # Add audio option
            formats.append({
                'format_id': 'bestaudio',
                'quality': '🎵 Audio Only (320kbps MP3) 📱 Mobile friendly',
                'height': 0,
                'width': 0,
                'ext': 'mp3',
                'filesize': 0,
                'fps': 0,
                'has_audio': True,
                'vcodec': 'none',
                'acodec': 'mp3',
                'estimated_size': self._estimate_audio_size(duration)
            })
            
            video_info['formats'] = formats
            logger.info(f"Processed {len(formats)} formats")
            
            return video_info
        
        except Exception as e:
            logger.error(f"Error extracting video info: {str(e)}")
            error_msg = str(e)
            if "Sign in to confirm you're not a bot" in error_msg or "bot" in error_msg.lower():
                raise Exception("YouTube is blocking requests. Please try again in a few minutes.")
            elif "Private video" in error_msg:
                raise Exception("This video is private and cannot be downloaded.")
            elif "Video unavailable" in error_msg:
                raise Exception("This video is unavailable or has been removed.")
            else:
                raise Exception(f"Failed to extract video info: {error_msg}")
    
    def _format_duration(self, seconds):
        """Format duration in human readable format"""
        if not seconds:
            return "Unknown"
        
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def _estimate_file_size(self, duration, height):
        """Estimate file size based on duration and quality"""
        if not duration:
            return "Unknown"
        
        # Rough estimates in MB per minute
        bitrates = {
            1080: 8,   # 8 MB per minute
            720: 5,    # 5 MB per minute  
            480: 3     # 3 MB per minute
        }
        
        bitrate = bitrates.get(height, 5)
        size_mb = (duration / 60) * bitrate
        
        if size_mb > 1024:
            return f"~{size_mb/1024:.1f} GB"
        else:
            return f"~{size_mb:.0f} MB"
    
    def _estimate_audio_size(self, duration):
        """Estimate audio file size"""
        if not duration:
            return "Unknown"
        
        # 320kbps MP3 = ~2.4 MB per minute
        size_mb = (duration / 60) * 2.4
        
        if size_mb > 1024:
            return f"~{size_mb/1024:.1f} GB"
        else:
            return f"~{size_mb:.0f} MB"
    
    def _extract_info_sync(self, url, ydl_opts):
        """Synchronous info extraction for thread pool"""
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url, download=False)
    
    async def download_video_async(self, task_id: str, url: str, format_id: str, quality: str):
        """Download video asynchronously with enhanced support for long videos"""
        async with self.download_semaphore:  # Limit concurrent downloads
            try:
                # Wait if too many downloads are active
                while not task_manager.can_start_download():
                    task_manager.update_task(
                        task_id,
                        status=TaskStatus.PENDING,
                        message="Waiting for download slot..."
                    )
                    await asyncio.sleep(5)
                
                task_manager.increment_active_downloads()
                
                task_manager.update_task(
                    task_id, 
                    status=TaskStatus.PROCESSING,
                    progress=5,
                    message="Starting download..."
                )
                
                # Clean URL
                video_id = self.extract_video_id(url)
                if video_id:
                    clean_url = f"https://www.youtube.com/watch?v={video_id}"
                else:
                    clean_url = url
                
                logger.info(f"Downloading: {clean_url}")
                logger.info(f"Format: {format_id}")
                logger.info(f"Quality: {quality}")
                
                # Generate unique filename
                unique_id = str(uuid.uuid4())[:8]
                
                task_manager.update_task(
                    task_id,
                    progress=10,
                    message="Preparing download..."
                )
                
                # Run download in thread pool
                loop = asyncio.get_event_loop()
                filename = await loop.run_in_executor(
                    None,
                    self._download_video_sync,
                    clean_url,
                    format_id,
                    quality,
                    unique_id,
                    task_id
                )
                
                # Register file for proper cleanup
                file_manager.register_file(task_id, filename, os.path.dirname(filename))
                
                download_url = f"/download-file/{task_id}"
                
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.COMPLETED,
                    progress=100,
                    message="Download completed successfully!",
                    filename=os.path.basename(filename),
                    download_url=download_url
                )
                
                logger.info(f"Download completed: {filename}")
                return filename
            
            except Exception as e:
                logger.error(f"Download error for task {task_id}: {str(e)}")
                error_msg = str(e)
                
                # Enhanced error handling for long videos
                if "Sign in to confirm you're not a bot" in error_msg:
                    error_msg = "YouTube is blocking download requests. Please try again later."
                elif "Requested format is not available" in error_msg:
                    error_msg = "The requested quality is not available. Please try a lower quality."
                elif "HTTP Error 403" in error_msg:
                    error_msg = "Access denied. This video may be restricted."
                elif "Connection broken" in error_msg or "timeout" in error_msg.lower():
                    error_msg = "Connection timeout. This may happen with very long videos. Please try again."
                elif "No space left on device" in error_msg:
                    error_msg = "Server storage full. Please try again later."
                
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    progress=0,
                    message="Download failed",
                    error=error_msg
                )
                raise Exception(error_msg)
            
            finally:
                task_manager.decrement_active_downloads()
    
    def _download_video_sync(self, url, format_id, quality, unique_id, task_id):
        """Synchronous download optimized for long videos"""
        temp_dir = None
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix='youtube_dl_long_')
            
            def progress_hook(d):
                if d['status'] == 'downloading':
                    try:
                        # Enhanced progress tracking for long videos
                        if 'total_bytes' in d and d['total_bytes']:
                            progress = int((d['downloaded_bytes'] / d['total_bytes']) * 80) + 15  # 15-95%
                            speed = d.get('speed', 0)
                            eta = d.get('eta', 0)
                            
                            # Format speed
                            if speed:
                                if speed > 1024 * 1024:
                                    speed_str = f"{speed / (1024 * 1024):.1f} MB/s"
                                elif speed > 1024:
                                    speed_str = f"{speed / 1024:.1f} KB/s"
                                else:
                                    speed_str = f"{speed:.0f} B/s"
                            else:
                                speed_str = "0 B/s"
                            
                            # Format ETA
                            if eta:
                                eta_str = self._format_duration(eta)
                            else:
                                eta_str = "Unknown"
                            
                            task_manager.update_task(
                                task_id,
                                progress=min(progress, 95),
                                message=f"Downloading... {d.get('_percent_str', '50%')}",
                                download_speed=speed_str,
                                eta=eta_str,
                                downloaded_bytes=d['downloaded_bytes'],
                                total_bytes=d['total_bytes']
                            )
                        elif '_percent_str' in d:
                            percent_str = d['_percent_str'].replace('%', '')
                            try:
                                progress = int(float(percent_str) * 0.8) + 15  # 15-95%
                                task_manager.update_task(
                                    task_id,
                                    progress=min(progress, 95),
                                    message=f"Downloading... {d.get('_percent_str', '50%')}"
                                )
                            except:
                                pass
                    except Exception as e:
                        logger.error(f"Progress hook error: {e}")
                        pass
                elif d['status'] == 'finished':
                    task_manager.update_task(
                        task_id,
                        progress=98,
                        message="Processing downloaded file..."
                    )
            
            if 'Audio Only' in quality or format_id == 'bestaudio':
                # MP3 AUDIO DOWNLOAD
                temp_filename = f"audio_{unique_id}.%(ext)s"
                
                # ✅ OPTIMIZED AUDIO OPTIONS FOR LONG VIDEOS
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': os.path.join(temp_dir, temp_filename),
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '320',
                    }],
                    'quiet': False,
                    'no_warnings': False,
                    'progress_hooks': [progress_hook],
                    'socket_timeout': 120,  # 2 minutes timeout
                    'retries': 10,
                    'fragment_retries': 15,
                    'skip_unavailable_fragments': True,
                    'keep_fragments': False,
                    'http_chunk_size': 10485760,  # 10MB chunks
                    'concurrent_fragment_downloads': 3,
                }
                
                # Get title for final filename
                info_opts = {'quiet': True, 'no_warnings': True, 'socket_timeout': 60}
                with yt_dlp.YoutubeDL(info_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    title = info.get('title', 'YouTube Audio')
                
                final_filename = self.get_unique_filename(temp_dir, f"{title}_audio", 'mp3')
            
            else:
                # VIDEO DOWNLOAD - OPTIMIZED FOR LONG VIDEOS
                temp_filename = f"video_{unique_id}.%(ext)s"
                
                # Get title for final filename
                info_opts = {'quiet': True, 'no_warnings': True, 'socket_timeout': 60}
                with yt_dlp.YoutubeDL(info_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    title = info.get('title', 'YouTube Video')
                    duration = info.get('duration', 0)
                
                final_filename = self.get_unique_filename(temp_dir, title, 'mp4')
                
                # ✅ ENHANCED FORMAT SELECTION FOR LONG VIDEOS
                if format_id == 'best[height>=1080]':
                    selected_format = (
                        "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/"
                        "bestvideo[height<=1080][ext=webm]+bestaudio[ext=webm]/"
                        "best[height<=1080][ext=mp4]/best[height<=1080]/"
                        "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/"
                        "best[height<=720]/best"
                    )
                elif format_id == 'best[height>=720]':
                    selected_format = (
                        "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/"
                        "bestvideo[height<=720][ext=webm]+bestaudio[ext=webm]/"
                        "best[height<=720][ext=mp4]/best[height<=720]/"
                        "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/"
                        "best[height<=480]/best"
                    )
                elif format_id == 'best[height>=480]':
                    selected_format = (
                        "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/"
                        "bestvideo[height<=480][ext=webm]+bestaudio[ext=webm]/"
                        "best[height<=480][ext=mp4]/best[height<=480]/"
                        "best"
                    )
                else:
                    selected_format = (
                        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/"
                        "bestvideo[ext=webm]+bestaudio[ext=webm]/"
                        "best[ext=mp4]/best"
                    )
                
                # ✅ OPTIMIZED YT-DLP OPTIONS FOR LONG VIDEOS
                ydl_opts = {
                    'format': selected_format,
                    'outtmpl': os.path.join(temp_dir, temp_filename),
                    'merge_output_format': 'mp4',
                    'quiet': False,
                    'no_warnings': False,
                    'progress_hooks': [progress_hook],
                    
                    # ✅ ENHANCED SETTINGS FOR LONG VIDEOS
                    'socket_timeout': 300,  # 5 minutes timeout
                    'http_chunk_size': 20971520,  # 20MB chunks for large files
                    'fragment_retries': 20,  # More retries for long videos
                    'retries': 15,
                    'file_access_retries': 5,
                    'skip_unavailable_fragments': True,
                    'keep_fragments': False,
                    'concurrent_fragment_downloads': 4,  # More concurrent downloads
                    
                    # Connection stability
                    'http_headers': {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    },
                    
                    # Post-processing
                    'prefer_ffmpeg': True,
                    'postprocessors': [{
                        'key': 'FFmpegVideoConvertor',
                        'preferedformat': 'mp4',
                    }] if not temp_filename.endswith('.mp4') else [],
                }
                
                # Adjust settings based on video duration
                if duration > 7200:  # More than 2 hours
                    ydl_opts.update({
                        'socket_timeout': 600,  # 10 minutes timeout
                        'http_chunk_size': 52428800,  # 50MB chunks
                        'fragment_retries': 30,
                        'retries': 20,
                        'concurrent_fragment_downloads': 2,  # Fewer concurrent for stability
                    })
                    logger.info("Applied long video optimizations (2+ hours)")
                elif duration > 3600:  # More than 1 hour
                    ydl_opts.update({
                        'socket_timeout': 450,  # 7.5 minutes timeout
                        'http_chunk_size': 31457280,  # 30MB chunks
                        'fragment_retries': 25,
                        'retries': 18,
                    })
                    logger.info("Applied medium video optimizations (1+ hour)")
            
            logger.info(f"Using format: {ydl_opts.get('format', 'default')}")
            logger.info(f"Chunk size: {ydl_opts.get('http_chunk_size', 0) / 1024 / 1024:.1f} MB")
            
            # Download the video/audio with retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([url])
                    break  # Success, exit retry loop
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Download attempt {attempt + 1} failed: {e}. Retrying...")
                        task_manager.update_task(
                            task_id,
                            message=f"Retrying download... (Attempt {attempt + 2}/{max_retries})"
                        )
                        time.sleep(10)  # Wait before retry
                    else:
                        raise e
            
            # Find the downloaded file
            temp_path = None
            for file in os.listdir(temp_dir):
                if unique_id in file:
                    temp_path = os.path.join(temp_dir, file)
                    break
            
            if not temp_path or not os.path.exists(temp_path):
                # Try to find any recently created file
                files = []
                for file in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, file)
                    if os.path.getctime(file_path) > time.time() - 1800:  # Created in last 30 minutes
                        files.append((file_path, os.path.getctime(file_path)))
                
                if files:
                    # Get the most recently created file
                    files.sort(key=lambda x: x[1], reverse=True)
                    temp_path = files[0][0]
                else:
                    raise Exception("Downloaded file not found")
            
            # Rename to final filename
            final_path = os.path.join(temp_dir, final_filename)
            
            # Remove target file if it exists
            if os.path.exists(final_path):
                os.remove(final_path)
            
            os.rename(temp_path, final_path)
            
            # Verify file size
            file_size = os.path.getsize(final_path)
            logger.info(f"Download completed: {final_filename} ({file_size / 1024 / 1024:.1f} MB)")
            
            return final_path  # Return full path
        
        except Exception as e:
            # Cleanup on error
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
            raise e

# ✅ ENHANCED CLEANUP FOR LONG VIDEOS
async def cleanup_temp_files():
    """Background task to clean up temporary files"""
    while True:
        try:
            file_manager.cleanup_old_files()
            
            # Additional system cleanup for long video downloads
            # Check system memory and disk space
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            if memory.percent > 85:
                logger.warning(f"High memory usage: {memory.percent}%")
            
            if disk.percent > 90:
                logger.warning(f"High disk usage: {disk.percent}%")
                # Force cleanup of old files
                file_manager.max_file_age = 1800  # Reduce to 30 minutes
            else:
                file_manager.max_file_age = 3600  # Reset to 1 hour
            
            await asyncio.sleep(120)  # Check every 2 minutes
        except Exception as e:
            logger.error(f"Cleanup task error: {e}")
            await asyncio.sleep(120)

# Initialize downloader
downloader = YouTubeDownloader()

# Background task for cleanup
async def cleanup_task():
    """Periodic cleanup of old tasks"""
    while True:
        await asyncio.sleep(1800)  # Run every 30 minutes
        task_manager.cleanup_old_tasks()

# Start cleanup task on startup
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_task())
    asyncio.create_task(cleanup_temp_files())

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page"""
    return templates.TemplateResponse('index.html', {"request": request})

@app.post("/extract")
async def extract_video_info(request_data: URLRequest):
    """Extract video information from URL"""
    try:
        url = request_data.url.strip()
        
        logger.info(f"Received URL: {url}")
        
        if not url:
            raise HTTPException(status_code=400, detail='Please provide a valid URL')
        
        if not downloader.is_valid_youtube_url(url):
            raise HTTPException(status_code=400, detail='Please provide a valid YouTube video URL')
        
        video_info = await downloader.extract_video_info(url)
        logger.info(f"Returning video info with {len(video_info['formats'])} formats")
        
        return {'success': True, 'data': video_info}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Extract error: {str(e)}")
        error_msg = str(e)
        if "blocking requests" in error_msg or "bot" in error_msg.lower():
            raise HTTPException(status_code=429, detail='YouTube is temporarily blocking requests. Please try again in a few minutes.')
        else:
            raise HTTPException(status_code=500, detail=error_msg)

@app.post("/download")
async def download_video_json(background_tasks: BackgroundTasks, request_data: DownloadRequest):
    """Download video with JSON payload"""
    return await _process_download(
        background_tasks,
        request_data.url,
        request_data.format_id,
        request_data.quality
    )

@app.post("/download/form")
async def download_video_form(
    background_tasks: BackgroundTasks,
    video_url: str = Form(...),
    format: str = Form(...)):
    """Download video with FormData"""
    # Parse format if it contains both format_id and quality
    if '|' in format:
        format_id, quality = format.split('|', 1)
    else:
        format_id = format
        quality = "Best Available Quality"
    
    return await _process_download(background_tasks, video_url, format_id, quality)

async def _process_download(background_tasks: BackgroundTasks, url: str, format_id: str, quality: str):
    """Process download request and return task ID"""
    try:
        url = url.strip()
        
        logger.info(f"Download request - URL: {url}, Format: {format_id}, Quality: {quality}")
        
        if not url or not format_id:
            raise HTTPException(status_code=400, detail='Missing required parameters')
        
        if not downloader.is_valid_youtube_url(url):
            raise HTTPException(status_code=400, detail='Please provide a valid YouTube video URL')
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Create task
        task_manager.create_task(task_id, url, format_id, quality)
        
        # Add background task
        background_tasks.add_task(
            downloader.download_video_async,
            task_id,
            url,
            format_id,
            quality
        )
        
        return {
            'success': True,
            'task_id': task_id,
            'message': 'Download started. Long videos may take more time. Use the task_id to check progress.',
            'status_url': f'/task/{task_id}'
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Get task status and progress"""
    task = task_manager.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return {
        'success': True,
        'task': task
    }

@app.get("/tasks")
async def get_all_tasks():
    """Get all tasks (for debugging/monitoring)"""
    return {
        'success': True,
        'tasks': list(task_manager.tasks.values()),
        'total_tasks': len(task_manager.tasks),
        'active_downloads': task_manager.active_downloads
    }

@app.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """Delete a specific task"""
    if task_id not in task_manager.tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    del task_manager.tasks[task_id]
    return {'success': True, 'message': 'Task deleted successfully'}

# ✅ ENHANCED FILE DOWNLOAD WITH STREAMING FOR LARGE FILES
@app.get("/download-file/{task_id}")
async def download_file(task_id: str):
    """Download file with streaming support for large files"""
    try:
        file_info = file_manager.get_file_info(task_id)
        
        if not file_info:
            raise HTTPException(status_code=404, detail="File not found or expired")
        
        file_path = file_info['file_path']
        filename = file_info['filename']
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found on disk")
        
        # Mark as downloaded for cleanup
        file_manager.mark_downloaded(task_id)
        
        # Determine content type
        content_type = "video/mp4"
        if filename.endswith('.mp3'):
            content_type = "audio/mpeg"
        elif filename.endswith('.webm'):
            content_type = "video/webm"
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # ✅ USE FileResponse FOR PROPER LARGE FILE SERVING
        response = FileResponse(
            path=file_path,
            filename=filename,
            media_type=content_type,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET",
                "Access-Control-Allow-Headers": "*",
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
            }
        )
        
        # Schedule cleanup after download
        async def cleanup_after_download():
            await asyncio.sleep(30)  # Wait 30 seconds after response for large files
            file_manager.cleanup_file(task_id)
        
        asyncio.create_task(cleanup_after_download())
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error serving file: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        'status': 'OK',
        'message': 'YouTube Downloader API - Long Video Support',
        'features': [
            'Long video support (1-10+ hours)',
            'Optimized for large files',
            'Enhanced error recovery',
            'Automatic file cleanup',
            'Memory and disk monitoring'
        ],
        'timestamp': datetime.now().isoformat(),
        'active_tasks': len([t for t in task_manager.tasks.values() if t['status'] == TaskStatus.PROCESSING]),
        'total_tasks': len(task_manager.tasks),
        'temp_files': len(file_manager._files),
        'active_downloads': task_manager.active_downloads,
        'system': {
            'memory_usage': f"{memory.percent}%",
            'disk_usage': f"{disk.percent}%",
            'available_memory': f"{memory.available / 1024 / 1024 / 1024:.1f} GB",
            'available_disk': f"{disk.free / 1024 / 1024 / 1024:.1f} GB"
        }
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    tasks = list(task_manager.tasks.values())
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    stats = {
        'total_tasks': len(tasks),
        'pending_tasks': len([t for t in tasks if t['status'] == TaskStatus.PENDING]),
        'processing_tasks': len([t for t in tasks if t['status'] == TaskStatus.PROCESSING]),
        'completed_tasks': len([t for t in tasks if t['status'] == TaskStatus.COMPLETED]),
        'failed_tasks': len([t for t in tasks if t['status'] == TaskStatus.FAILED]),
        'temp_files': len(file_manager._files),
        'active_downloads': task_manager.active_downloads,
        'max_concurrent_downloads': task_manager.max_concurrent_downloads,
        'system_resources': {
            'memory_percent': memory.percent,
            'disk_percent': disk.percent,
            'available_memory_gb': memory.available / 1024 / 1024 / 1024,
            'available_disk_gb': disk.free / 1024 / 1024 / 1024
        },
        'uptime': datetime.now().isoformat()
    }
    
    return {'success': True, 'stats': stats}

# ✅ GRACEFUL SHUTDOWN FOR LONG DOWNLOADS
@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown"""
    logger.info("Shutting down gracefully...")
    
    # Wait for active downloads to complete (max 5 minutes)
    shutdown_timeout = 300  # 5 minutes
    start_time = time.time()
    
    while task_manager.active_downloads > 0 and (time.time() - start_time) < shutdown_timeout:
        logger.info(f"Waiting for {task_manager.active_downloads} active downloads to complete...")
        await asyncio.sleep(10)
    
    # Force cleanup
    file_manager.cleanup_old_files()
    logger.info("Shutdown complete")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5003, reload=True, workers=1)
