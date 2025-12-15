#!/usr/bin/env python3
"""
cleanup_repo.py - Clean up old files in your repository
Run this locally or in a workflow to manage storage
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

def get_file_age_days(filepath):
    """Get the age of a file in days"""
    mtime = os.path.getmtime(filepath)
    file_date = datetime.fromtimestamp(mtime)
    age = datetime.now() - file_date
    return age.days

def cleanup_directory(directory, extensions, days_old, keep_latest=None, exclude_files=None):
    """
    Clean up files in a directory
    
    Args:
        directory: Directory path to clean
        extensions: List of file extensions to target (e.g., ['.png', '.jpg'])
        days_old: Delete files older than this many days
        keep_latest: Number of most recent files to always keep
        exclude_files: List of filenames to never delete
    """
    if not os.path.exists(directory):
        print(f"⚠️  Directory not found: {directory}")
        return
    
    directory = Path(directory)
    exclude_files = exclude_files or []
    
    # Find all matching files
    files = []
    for ext in extensions:
        files.extend(directory.glob(f"*{ext}"))
    
    # Sort by modification time (newest first)
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    deleted_count = 0
    deleted_size = 0
    kept_count = 0
    
    for i, filepath in enumerate(files):
        # Skip excluded files
        if filepath.name in exclude_files:
            print(f"⏭️  Keeping excluded: {filepath.name}")
            kept_count += 1
            continue
        
        # Keep the N most recent files
        if keep_latest and i < keep_latest:
            print(f"⏭️  Keeping recent: {filepath.name}")
            kept_count += 1
            continue
        
        # Check age
        age = get_file_age_days(filepath)
        if age > days_old:
            size = filepath.stat().st_size
            try:
                filepath.unlink()
                deleted_count += 1
                deleted_size += size
                print(f"🗑️  Deleted: {filepath.name} ({age} days old, {size/1024:.1f} KB)")
            except Exception as e:
                print(f"❌ Error deleting {filepath.name}: {e}")
        else:
            kept_count += 1
    
    print(f"\n📊 Summary for {directory}:")
    print(f"   Deleted: {deleted_count} files ({deleted_size/1024/1024:.2f} MB)")
    print(f"   Kept: {kept_count} files")

def main():
    print("🧹 Starting repository cleanup...\n")
    
    # Clean detection images (keep last 50, delete if older than 30 days)
    print("=" * 60)
    print("Cleaning detection images...")
    print("=" * 60)
    cleanup_directory(
        directory="detections",
        extensions=[".png", ".jpg"],
        days_old=30,
        keep_latest=50,
        exclude_files=["overlay_C2.png", "overlay_C3.png", "contact_C2.png", "contact_C3.png"]
    )
    
    # Clean detection JSON files (keep last 100, delete if older than 60 days)
    print("\n" + "=" * 60)
    print("Cleaning detection JSON files...")
    print("=" * 60)
    cleanup_directory(
        directory="detections",
        extensions=[".json"],
        days_old=60,
        keep_latest=100,
        exclude_files=["latest_status.json"]
    )
    
    # Clean temporary frames (delete if older than 2 days)
    print("\n" + "=" * 60)
    print("Cleaning temporary frames...")
    print("=" * 60)
    cleanup_directory(
        directory="frames",
        extensions=[".png", ".jpg", ".fits"],
        days_old=2,
        keep_latest=10
    )
    
    # Report repository size
    print("\n" + "=" * 60)
    print("Repository size summary:")
    print("=" * 60)
    
    for directory in ["detections", "frames"]:
        if os.path.exists(directory):
            total_size = sum(f.stat().st_size for f in Path(directory).rglob('*') if f.is_file())
            file_count = len(list(Path(directory).rglob('*')))
            print(f"   {directory}/: {file_count} files, {total_size/1024/1024:.2f} MB")
    
    print("\n✅ Cleanup complete!")

if __name__ == "__main__":
    main()
