import requests
import pathlib
from datetime import datetime, timedelta
import time

def fetch_window(hours_back=12, step_min=30, root="frames", max_frames=12):
    """Fetch LASCO C2/C3 images"""
    fetched = []
    
    # Create directories
    for cam in ["C2", "C3"]:
        pathlib.Path(f"{root}/{cam}").mkdir(parents=True, exist_ok=True)
    
    # Simple mock - replace with actual SOHO fetching
    print(f"Mock fetching {max_frames} frames for last {hours_back} hours")
    
    # Return empty list for now
    return fetched
