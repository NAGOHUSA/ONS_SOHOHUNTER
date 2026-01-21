#!/usr/bin/env python3
"""
fetch_lasco.py - Download LASCO C2 and C3 images from SOHO NASA
"""

import os
import sys
import time
import requests
import pathlib
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SOHO LASCO base URLs
SOHO_BASE_URL = "https://soho.nascom.nasa.gov"
LASCO_REALTIME_URL = f"{SOHO_BASE_URL}/data/realtime"

# Time format used in SOHO filenames
TIME_FORMAT = "%Y%m%d_%H%M"

class SohoImageFetcher:
    """Fetcher for LASCO C2 and C3 images"""
    
    def __init__(self, root_dir: str = "frames"):
        self.root_dir = pathlib.Path(root_dir)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; SOHO-Comet-Hunter/1.0; +https://github.com/NAGOHUSA/ONS_SOHOHUNTER)'
        })
        
    def ensure_dirs(self):
        """Create necessary directories"""
        for cam in ["C2", "C3"]:
            (self.root_dir / cam).mkdir(parents=True, exist_ok=True)
    
    def get_available_times(self, hours_back: int = 24) -> List[datetime]:
        """Generate list of available times for last N hours"""
        now = datetime.utcnow()
        start_time = now - timedelta(hours=hours_back)
        
        # LASCO C2 images are taken every ~12 minutes
        # LASCO C3 images are taken every ~24 minutes
        # We'll generate timestamps every 12 minutes
        times = []
        current = start_time
        
        while current <= now:
            times.append(current)
            current += timedelta(minutes=12)
        
        return times
    
    def build_url(self, timestamp: datetime, instrument: str) -> str:
        """Build SOHO image URL for given timestamp and instrument"""
        # Format: https://soho.nascom.nasa.gov/data/realtime/c2/1024/20250121_1200_c2_1024.jpg
        date_str = timestamp.strftime("%Y%m%d")
        time_str = timestamp.strftime("%H%M")
        
        # First try the realtime directory (most recent images)
        url = f"{LASCO_REALTIME_URL}/{instrument.lower()}/1024/{date_str}_{time_str}_{instrument.lower()}_1024.jpg"
        
        return url
    
    def fetch_image(self, url: str, save_path: pathlib.Path, max_retries: int = 3) -> bool:
        """Download a single image with retries"""
        for attempt in range(max_retries):
            try:
                logger.debug(f"Fetching {url} (attempt {attempt + 1}/{max_retries})")
                
                response = self.session.get(url, timeout=30)
                
                if response.status_code == 200:
                    # Check if it's actually an image
                    if 'image' in response.headers.get('content-type', '').lower():
                        with open(save_path, 'wb') as f:
                            f.write(response.content)
                        
                        # Verify the image can be read
                        try:
                            import cv2
                            img = cv2.imread(str(save_path))
                            if img is not None:
                                logger.info(f"Successfully downloaded: {save_path.name}")
                                return True
                            else:
                                logger.warning(f"Downloaded file is not a valid image: {save_path.name}")
                                os.remove(save_path)
                        except ImportError:
                            # If OpenCV not available, just trust the content-type
                            logger.info(f"Downloaded: {save_path.name}")
                            return True
                    else:
                        logger.warning(f"Not an image: {url}")
                elif response.status_code == 404:
                    logger.debug(f"Image not found (404): {url}")
                    return False
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed for {url}: {e}")
            
            # Wait before retry
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return False
    
    def fetch_instrument_images(self, instrument: str, hours_back: int, step_min: int, max_frames: int) -> List[str]:
        """Fetch images for a specific instrument (C2 or C3)"""
        fetched = []
        times = self.get_available_times(hours_back)
        
        # Apply step_min filtering
        filtered_times = [t for i, t in enumerate(times) if i % (step_min // 12) == 0]
        
        logger.info(f"Looking for {instrument} images from {len(filtered_times)} timestamps")
        
        for timestamp in filtered_times:
            if len(fetched) >= max_frames:
                break
                
            # Build filename and path
            date_str = timestamp.strftime("%Y%m%d")
            time_str = timestamp.strftime("%H%M")
            filename = f"{date_str}_{time_str}_{instrument.lower()}_1024.jpg"
            save_path = self.root_dir / instrument / filename
            
            # Skip if already exists
            if save_path.exists():
                logger.debug(f"Already exists: {filename}")
                fetched.append(str(save_path))
                continue
            
            # Build URL and fetch
            url = self.build_url(timestamp, instrument)
            if self.fetch_image(url, save_path):
                fetched.append(str(save_path))
            
            # Be nice to the server
            time.sleep(0.5)
        
        return fetched
    
    def fetch_window(self, hours_back: int = 24, step_min: int = 30, root: str = "frames", max_frames: int = 12) -> List[str]:
        """Main function to fetch LASCO images for specified window"""
        logger.info(f"Fetching LASCO images: {hours_back}h back, {step_min}min step, max {max_frames} frames")
        
        # Ensure step_min is at least 12 (minimum interval for LASCO)
        if step_min < 12:
            step_min = 12
            logger.warning(f"Step min adjusted to {step_min} (minimum LASCO interval)")
        
        self.root_dir = pathlib.Path(root)
        self.ensure_dirs()
        
        all_fetched = []
        
        # Fetch C2 and C3 images
        for instrument in ["C2", "C3"]:
            logger.info(f"Fetching {instrument} images...")
            fetched = self.fetch_instrument_images(
                instrument=instrument,
                hours_back=hours_back,
                step_min=step_min,
                max_frames=max_frames // 2  # Split between C2 and C3
            )
            all_fetched.extend(fetched)
            logger.info(f"Fetched {len(fetched)} {instrument} images")
        
        # Fallback: If no images found, try to get latest images
        if not all_fetched:
            logger.warning("No images found via timestamp method. Trying latest images...")
            all_fetched = self.fetch_latest_images(max_frames)
        
        logger.info(f"Total fetched: {len(all_fetched)} images")
        return all_fetched
    
    def fetch_latest_images(self, max_frames: int = 6) -> List[str]:
        """Fallback: Fetch the latest available images"""
        fetched = []
        
        # Latest image URLs
        latest_urls = {
            "C2": f"{LASCO_REALTIME_URL}/c2/1024/latest.jpg",
            "C3": f"{LASCO_REALTIME_URL}/c3/1024/latest.jpg"
        }
        
        for instrument, url in latest_urls.items():
            if len(fetched) >= max_frames:
                break
                
            timestamp = datetime.utcnow()
            date_str = timestamp.strftime("%Y%m%d")
            time_str = timestamp.strftime("%H%M")
            filename = f"{date_str}_{time_str}_{instrument.lower()}_latest.jpg"
            save_path = self.root_dir / instrument / filename
            
            if self.fetch_image(url, save_path):
                fetched.append(str(save_path))
        
        return fetched
    
    def clean_old_files(self, keep_hours: int = 48):
        """Clean up old image files"""
        cutoff = datetime.utcnow() - timedelta(hours=keep_hours)
        
        for cam in ["C2", "C3"]:
            cam_dir = self.root_dir / cam
            if not cam_dir.exists():
                continue
            
            for file_path in cam_dir.glob("*.jpg"):
                try:
                    # Parse timestamp from filename
                    filename = file_path.stem
                    time_str = filename.split('_')[1] if '_' in filename else ""
                    
                    if time_str and len(time_str) >= 4:
                        file_time = datetime.strptime(time_str, "%H%M")
                        # We'll just delete based on file modification time instead
                        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        
                        if file_mtime < cutoff:
                            logger.debug(f"Deleting old file: {file_path.name}")
                            file_path.unlink()
                except Exception as e:
                    logger.debug(f"Could not parse timestamp for {file_path}: {e}")


def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch LASCO C2/C3 images from SOHO")
    parser.add_argument("--hours", type=int, default=24, help="Hours to look back")
    parser.add_argument("--step-min", type=int, default=12, help="Minutes between frames")
    parser.add_argument("--max-frames", type=int, default=12, help="Maximum frames to fetch")
    parser.add_argument("--out-dir", type=str, default="frames", help="Output directory")
    parser.add_argument("--clean-old", action="store_true", help="Clean old files (>48h)")
    
    args = parser.parse_args()
    
    fetcher = SohoImageFetcher(root_dir=args.out_dir)
    
    if args.clean_old:
        logger.info("Cleaning old files...")
        fetcher.clean_old_files()
    
    logger.info(f"Starting fetch: {args.hours}h, {args.step_min}min step, max {args.max_frames} frames")
    
    try:
        fetched = fetcher.fetch_window(
            hours_back=args.hours,
            step_min=args.step_min,
            root=args.out_dir,
            max_frames=args.max_frames
        )
        
        logger.info(f"Successfully fetched {len(fetched)} images")
        
        # List what was fetched
        if fetched:
            logger.info("Fetched files:")
            for f in fetched[:10]:  # Show first 10
                logger.info(f"  {os.path.basename(f)}")
            if len(fetched) > 10:
                logger.info(f"  ... and {len(fetched) - 10} more")
        else:
            logger.warning("No images were fetched!")
            
    except Exception as e:
        logger.error(f"Failed to fetch images: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
