


import os
import yt_dlp
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import re
from urllib.parse import urlparse, parse_qs
import time
from datetime import datetime
import uuid
import asyncio
from typing import Dict, Optional
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Task status enum
class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Task tracking system
class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self.cleanup_interval = 3600  # 1 hour
        
    def create_task(self, task_id: str, url: str, format_id: str, quality: str) -> str:
        """Create a new download task"""
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
            "updated_at": datetime.now()
        }
        return task_id
    
    def update_task(self, task_id: str, **kwargs):
        """Update task information"""
        if task_id in self.tasks:
            self.tasks[task_id].update(kwargs)
            self.tasks[task_id]["updated_at"] = datetime.now()
    
    def get_task(self, task_id: str) -> Optional[Dict]:
        """Get task information"""
        return self.tasks.get(task_id)
    
    def cleanup_old_tasks(self):
        """Remove old completed/failed tasks"""
        current_time = datetime.now()
        to_remove = []
        
        for task_id, task in self.tasks.items():
            time_diff = (current_time - task["updated_at"]).total_seconds()
            if time_diff > self.cleanup_interval and task["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                to_remove.append(task_id)
        
        for task_id in to_remove:
            del self.tasks[task_id]
            logger.info(f"Cleaned up old task: {task_id}")

# Pydantic models for request bodies
class URLRequest(BaseModel):
    url: str

class DownloadRequest(BaseModel):
    url: str
    format_id: str
    quality: str

# Initialize FastAPI app
app = FastAPI(
    title="YouTube Downloader API",
    description="High-performance async YouTube video downloader with task tracking",
    version="2.0.0"
)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize task manager
task_manager = TaskManager()

def sanitize_title(title):
    """Sanitize title for filename"""
    return re.sub(r'[\\/*?:"<>|]', "_", title)

# Create necessary directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

class YouTubeDownloader:
    def __init__(self):
        self.output_path = 'outputs'
            
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
                
            # Simple, working yt-dlp options
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'format': 'best',
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
            logger.info(f"Available formats: {len(info.get('formats', []))}")
                        
            # Extract basic video information
            video_info = {
                'title': info.get('title', 'YouTube Video'),
                'duration': info.get('duration', 0),
                'thumbnail': info.get('thumbnail', ''),
                'uploader': info.get('uploader', 'YouTube Channel'),
                'view_count': info.get('view_count', 0),
                'like_count': info.get('like_count', 0),
                'upload_date': info.get('upload_date', ''),
                'description': info.get('description', '')[:200] + '...' if info.get('description') else '',
                'formats': []
            }
                        
            # Check actual available formats
            available_formats = info.get('formats', [])
            has_1080p = any(f.get('height', 0) >= 1080 for f in available_formats if f.get('vcodec') != 'none')
            has_720p = any(f.get('height', 0) >= 720 for f in available_formats if f.get('vcodec') != 'none')
            has_480p = any(f.get('height', 0) >= 480 for f in available_formats if f.get('vcodec') != 'none')
                        
            logger.info(f"Available qualities - 1080p: {has_1080p}, 720p: {has_720p}, 480p: {has_480p}")
                        
            formats = []
                        
            # Only add formats that are actually available
            if has_1080p:
                formats.append({
                    'format_id': 'best[height>=1080]',
                    'quality': 'Full HD (1080p)',
                    'height': 1080,
                    'width': 1920,
                    'ext': 'mp4',
                    'filesize': 0,
                    'fps': 30,
                    'has_audio': True,
                    'vcodec': 'h264',
                    'acodec': 'aac',
                    'recommended': True
                })
                        
            if has_720p:
                formats.append({
                    'format_id': 'best[height>=720]',
                    'quality': 'HD (720p)',
                    'height': 720,
                    'width': 1280,
                    'ext': 'mp4',
                    'filesize': 0,
                    'fps': 30,
                    'has_audio': True,
                    'vcodec': 'h264',
                    'acodec': 'aac',
                    'recommended': not has_1080p
                })
                        
            if has_480p:
                formats.append({
                    'format_id': 'best[height>=480]',
                    'quality': 'Standard (480p)',
                    'height': 480,
                    'width': 854,
                    'ext': 'mp4',
                    'filesize': 0,
                    'fps': 30,
                    'has_audio': True,
                    'vcodec': 'h264',
                    'acodec': 'aac',
                    'recommended': False
                })
                        
            # Always add best available and audio options
            formats.append({
                'format_id': 'best',
                'quality': 'Best Available Quality',
                'height': 720,
                'width': 1280,
                'ext': 'mp4',
                'filesize': 0,
                'fps': 30,
                'has_audio': True,
                'vcodec': 'h264',
                'acodec': 'aac',
                'recommended': len(formats) == 0
            })
                        
            # Add working MP3 audio option
            formats.append({
                'format_id': 'bestaudio',
                'quality': '🎵 Audio Only (320kbps MP3)',
                'height': 0,
                'width': 0,
                'ext': 'mp3',
                'filesize': 0,
                'fps': 0,
                'has_audio': True,
                'vcodec': 'none',
                'acodec': 'mp3'
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
    
    def _extract_info_sync(self, url, ydl_opts):
        """Synchronous info extraction for thread pool"""
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url, download=False)
        
    async def download_video_async(self, task_id: str, url: str, format_id: str, quality: str):
        """Download video asynchronously with progress tracking"""
        try:
            task_manager.update_task(
                task_id, 
                status=TaskStatus.PROCESSING, 
                progress=10, 
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
                progress=20, 
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
            
            download_url = f"/download/{filename}"
            
            task_manager.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                progress=100,
                message="Download completed successfully!",
                filename=filename,
                download_url=download_url
            )
            
            logger.info(f"Download completed: {filename}")
            return filename
                    
        except Exception as e:
            logger.error(f"Download error for task {task_id}: {str(e)}")
            error_msg = str(e)
            if "Sign in to confirm you're not a bot" in error_msg:
                error_msg = "YouTube is blocking download requests. Please try again later."
            elif "Requested format is not available" in error_msg:
                error_msg = "The requested quality is not available for this video. Please try a lower quality."
            elif "HTTP Error 403" in error_msg:
                error_msg = "Access denied. This video may be restricted or require sign-in."
            
            task_manager.update_task(
                task_id,
                status=TaskStatus.FAILED,
                progress=0,
                message="Download failed",
                error=error_msg
            )
            raise Exception(error_msg)
    
    def _download_video_sync(self, url, format_id, quality, unique_id, task_id):
        """Synchronous download for thread pool"""
        def progress_hook(d):
            if d['status'] == 'downloading':
                try:
                    if 'total_bytes' in d and d['total_bytes']:
                        progress = int((d['downloaded_bytes'] / d['total_bytes']) * 70) + 20  # 20-90%
                    elif '_percent_str' in d:
                        percent_str = d['_percent_str'].replace('%', '')
                        progress = int(float(percent_str) * 0.7) + 20  # 20-90%
                    else:
                        progress = 50  # Default progress
                    
                    task_manager.update_task(
                        task_id,
                        progress=min(progress, 90),
                        message=f"Downloading... {d.get('_percent_str', '50%')}"
                    )
                except:
                    pass
            elif d['status'] == 'finished':
                task_manager.update_task(
                    task_id,
                    progress=95,
                    message="Processing downloaded file..."
                )
        
        if 'Audio Only' in quality or format_id == 'bestaudio':
            # MP3 AUDIO DOWNLOAD
            temp_filename = f"audio_{unique_id}.%(ext)s"
                        
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(self.output_path, temp_filename),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '320',
                }],
                'quiet': True,
                'progress_hooks': [progress_hook],
            }
                        
            # Get title for final filename
            info_opts = {'quiet': True, 'no_warnings': True}
            with yt_dlp.YoutubeDL(info_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'YouTube Audio')
                        
            final_filename = self.get_unique_filename(self.output_path, f"{title}_audio", 'mp3')
                
        else:
            # VIDEO DOWNLOAD
            temp_filename = f"video_{unique_id}.%(ext)s"
                        
            # Get title for final filename
            info_opts = {'quiet': True, 'no_warnings': True}
            with yt_dlp.YoutubeDL(info_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'YouTube Video')
                        
            final_filename = self.get_unique_filename(self.output_path, title, 'mp4')
                        
            # Robust format selection with multiple fallbacks
            if format_id == 'best[height>=1080]':
                selected_format = (
                    "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080][ext=webm]+bestaudio[ext=webm]/"
                    "best[height<=1080][ext=mp4]/best[height<=1080]/"
                    "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/"
                    "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
                )
            elif format_id == 'best[height>=720]':
                selected_format = (
                    "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=720][ext=webm]+bestaudio[ext=webm]/"
                    "best[height<=720][ext=mp4]/best[height<=720]/"
                    "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480]/"
                    "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
                )
            elif format_id == 'best[height>=480]':
                selected_format = (
                    "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=480][ext=webm]+bestaudio[ext=webm]/"
                    "best[height<=480][ext=mp4]/best[height<=480]/"
                    "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[height<=360]/"
                    "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
                )
            elif format_id == 'best':
                selected_format = (
                    "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo[ext=webm]+bestaudio[ext=webm]/"
                    "best[ext=mp4]/best[ext=webm]/best"
                )
            else:
                selected_format = f"{format_id}/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
            
            ydl_opts = {
                'format': selected_format,
                'outtmpl': os.path.join(self.output_path, temp_filename),
                'merge_output_format': 'mp4',
                'quiet': False,
                'no_warnings': False,
                'http_chunk_size': 10485760,  # 10MB chunks
                'fragment_retries': 5,
                'retries': 5,
                'file_access_retries': 3,
                'writesubtitles': False,
                'writeautomaticsub': False,
                'prefer_ffmpeg': True,
                'progress_hooks': [progress_hook],
                'postprocessors': [{
                    'key': 'FFmpegVideoConvertor',
                    'preferedformat': 'mp4',
                }] if not temp_filename.endswith('.mp4') else [],
            }
                
        logger.info(f"Using format: {ydl_opts.get('format', 'default')}")
        
        # Download the video/audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
                
        # Find the downloaded file
        temp_path = None
        for file in os.listdir(self.output_path):
            if unique_id in file:
                temp_path = os.path.join(self.output_path, file)
                break
                
        if not temp_path or not os.path.exists(temp_path):
            # Try to find any recently created file
            files = []
            for file in os.listdir(self.output_path):
                file_path = os.path.join(self.output_path, file)
                if os.path.getctime(file_path) > time.time() - 300:  # Created in last 5 minutes
                    files.append((file_path, os.path.getctime(file_path)))
                        
            if files:
                # Get the most recently created file
                files.sort(key=lambda x: x[1], reverse=True)
                temp_path = files[0][0]
            else:
                raise Exception("Downloaded file not found")
                
        # Rename to final filename
        final_path = os.path.join(self.output_path, final_filename)
                
        # Remove target file if it exists
        if os.path.exists(final_path):
            os.remove(final_path)
                
        os.rename(temp_path, final_path)
                
        logger.info(f"Download completed: {final_filename}")
        return final_filename

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
    """Download video with JSON payload (backward compatibility)"""
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
    format: str = Form(...)
):
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
            'message': 'Download started. Use the task_id to check progress.',
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
        'total_tasks': len(task_manager.tasks)
    }

@app.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """Delete a specific task"""
    if task_id not in task_manager.tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    del task_manager.tasks[task_id]
    return {'success': True, 'message': 'Task deleted successfully'}

@app.get("/download/{filename}")
async def serve_file(filename: str):
    """Serve downloaded files"""
    try:
        file_path = os.path.join('outputs', filename)
        if os.path.exists(file_path):
            return FileResponse(file_path, filename=filename)
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving file: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'OK',
        'timestamp': datetime.now().isoformat(),
        'active_tasks': len([t for t in task_manager.tasks.values() if t['status'] == TaskStatus.PROCESSING]),
        'total_tasks': len(task_manager.tasks)
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    tasks = list(task_manager.tasks.values())
    
    stats = {
        'total_tasks': len(tasks),
        'pending_tasks': len([t for t in tasks if t['status'] == TaskStatus.PENDING]),
        'processing_tasks': len([t for t in tasks if t['status'] == TaskStatus.PROCESSING]),
        'completed_tasks': len([t for t in tasks if t['status'] == TaskStatus.COMPLETED]),
        'failed_tasks': len([t for t in tasks if t['status'] == TaskStatus.FAILED]),
        'uptime': datetime.now().isoformat()
    }
    
    return {'success': True, 'stats': stats}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5003, reload=True, workers=1)
