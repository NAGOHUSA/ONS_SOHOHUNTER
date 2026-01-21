#!/usr/bin/env python3
"""Test script to verify LASCO image fetching"""

import os
import sys
sys.path.append('detector')

from fetch_lasco import SohoImageFetcher
import cv2

def main():
    print("=== Testing LASCO Image Fetch ===")
    
    fetcher = SohoImageFetcher(root_dir="test_frames")
    
    # Test 1: Try to fetch a few images
    print("\n1. Testing image fetch...")
    fetched = fetcher.fetch_window(hours_back=24, step_min=60, max_frames=6)
    print(f"Fetched {len(fetched)} images")
    
    # Test 2: Verify images are readable
    print("\n2. Verifying images...")
    for f in fetched:
        try:
            img = cv2.imread(f)
            if img is not None:
                print(f"✓ {os.path.basename(f)}: {img.shape}")
            else:
                print(f"✗ {os.path.basename(f)}: Cannot read")
        except:
            print(f"✗ {os.path.basename(f)}: Error")
    
    # Test 3: Check time parsing
    print("\n3. Testing time parsing...")
    from datetime import datetime
    
    test_files = [
        "20260121_1200_c2_1024.jpg",
        "20260121_1200_c2_latest.jpg",
        "20260121_1200_c2_test.jpg"
    ]
    
    for filename in test_files:
        # Extract timestamp
        parts = filename.split('_')
        if len(parts) >= 2:
            date_str = parts[0]
            time_str = parts[1]
            if len(date_str) == 8 and len(time_str) >= 4:
                time_str = time_str[:6] if len(time_str) >= 6 else time_str.ljust(6, '0')
                iso = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}T{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}Z"
                print(f"{filename} -> {iso}")

if __name__ == "__main__":
    main()
