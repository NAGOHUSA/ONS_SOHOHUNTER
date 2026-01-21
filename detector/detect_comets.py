def is_within_last_hours(filename: str, hours: int = 12) -> bool:
    """Check if frame is within last N hours - FIXED VERSION"""
    iso = parse_frame_iso(filename)
    if not iso:
        # Try to extract timestamp from filename directly
        try:
            # Handle formats like: 20260121_1408_c2_latest.jpg
            parts = filename.split('_')
            if len(parts) >= 2:
                date_str = parts[0]
                time_str = parts[1]
                if len(date_str) == 8 and len(time_str) >= 4:
                    # Pad time to 6 digits if needed
                    time_str = time_str[:6] if len(time_str) >= 6 else time_str.ljust(6, '0')
                    iso = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}T{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}Z"
        except:
            return False
    
    if not iso:
        return False
    
    try:
        # Handle both "Z" and no timezone
        if iso.endswith('Z'):
            frame_time = datetime.fromisoformat(iso[:-1] + "+00:00")
        else:
            frame_time = datetime.fromisoformat(iso)
        
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return frame_time >= cutoff
    except Exception as e:
        print(f"Warning: Could not parse time '{iso}' from {filename}: {e}")
        return False
