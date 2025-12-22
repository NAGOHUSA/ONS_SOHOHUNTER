#!/usr/bin/env python3
"""
Comprehensive disk space cleanup script for GitHub Actions runners.
This script removes common space-consuming files and directories.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def get_disk_usage():
    """Get current disk usage information."""
    try:
        result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
        print("Current disk usage:")
        print(result.stdout)
    except Exception as e:
        print(f"Could not get disk usage: {e}")


def remove_path(path, description):
    """Safely remove a file or directory."""
    try:
        path_obj = Path(path)
        if path_obj.exists():
            if path_obj.is_file():
                path_obj.unlink()
                print(f"✓ Removed {description}: {path}")
            elif path_obj.is_dir():
                shutil.rmtree(path_obj)
                print(f"✓ Removed {description}: {path}")
            return True
        else:
            print(f"⊘ Not found: {description} ({path})")
            return False
    except Exception as e:
        print(f"✗ Failed to remove {description} ({path}): {e}")
        return False


def run_command(cmd, description):
    """Run a shell command safely."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {description}")
            if result.stdout:
                print(f"  Output: {result.stdout.strip()}")
        else:
            print(f"✗ {description} failed: {result.stderr.strip()}")
    except Exception as e:
        print(f"✗ {description} error: {e}")


def clean_apt_packages():
    """Remove unnecessary APT packages and clean cache."""
    print("\n=== Cleaning APT packages ===")
    
    # Packages that are commonly large and not always needed
    packages_to_remove = [
        'ansible',
        'azure-cli',
        'google-cloud-sdk',
        'mono-devel',
        'powershell',
        'firefox',
        'google-chrome-stable',
        'microsoft-edge-stable'
    ]
    
    for package in packages_to_remove:
        run_command(f'sudo apt-get remove -y {package} 2>/dev/null || true', 
                   f"Removing {package}")
    
    run_command('sudo apt-get autoremove -y', "Running apt autoremove")
    run_command('sudo apt-get clean', "Cleaning apt cache")


def clean_docker():
    """Clean Docker images, containers, and build cache."""
    print("\n=== Cleaning Docker ===")
    
    run_command('docker system prune -af --volumes 2>/dev/null || true', 
               "Pruning Docker system")
    run_command('docker image prune -af 2>/dev/null || true', 
               "Pruning Docker images")


def clean_large_directories():
    """Remove large tool directories that consume significant space."""
    print("\n=== Cleaning large directories ===")
    
    directories = [
        ('/usr/local/lib/android', "Android SDK"),
        ('/usr/share/dotnet', ".NET SDK"),
        ('/usr/local/share/boost', "Boost"),
        ('/usr/local/graalvm', "GraalVM"),
        ('/usr/local/.ghcup', "GHCup"),
        ('/usr/local/share/powershell', "PowerShell"),
        ('/usr/local/share/chromium', "Chromium"),
        ('/usr/local/lib/node_modules', "Global npm modules"),
        ('/opt/hostedtoolcache', "Hosted tool cache"),
        ('/imagegeneration', "Image generation"),
    ]
    
    for directory, description in directories:
        remove_path(directory, description)


def clean_caches():
    """Clean various cache directories."""
    print("\n=== Cleaning caches ===")
    
    cache_dirs = [
        '~/.cache',
        '~/.npm',
        '~/.yarn',
        '~/.m2',
        '~/.gradle',
        '~/.cargo',
        '~/.rustup',
        '~/go/pkg',
    ]
    
    for cache_dir in cache_dirs:
        expanded = os.path.expanduser(cache_dir)
        remove_path(expanded, f"Cache directory {cache_dir}")


def clean_temp_files():
    """Clean temporary files and logs."""
    print("\n=== Cleaning temporary files ===")
    
    temp_locations = [
        '/tmp',
        '/var/tmp',
        '/var/log',
    ]
    
    for temp_dir in temp_locations:
        try:
            path = Path(temp_dir)
            if path.exists() and path.is_dir():
                for item in path.iterdir():
                    try:
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                    except Exception as e:
                        pass  # Some files may be in use
                print(f"✓ Cleaned {temp_dir}")
        except Exception as e:
            print(f"⊘ Could not fully clean {temp_dir}: {e}")


def clean_github_workspace():
    """Clean GitHub workspace artifacts."""
    print("\n=== Cleaning GitHub workspace ===")
    
    workspace = os.environ.get('GITHUB_WORKSPACE', '')
    if workspace:
        workspace_path = Path(workspace)
        
        # Common directories to clean in a workspace
        to_clean = [
            'node_modules',
            '.venv',
            'venv',
            'env',
            '__pycache__',
            '.pytest_cache',
            '.tox',
            'dist',
            'build',
            '*.egg-info',
            '.eggs',
            'htmlcov',
            '.coverage',
            'target',  # Rust/Java builds
            'out',
            'bin',
            'obj',
        ]
        
        for pattern in to_clean:
            if '*' in pattern:
                # Handle glob patterns
                for match in workspace_path.glob(f"**/{pattern}"):
                    remove_path(match, f"Build artifact {match.name}")
            else:
                # Handle exact directory names
                for match in workspace_path.glob(f"**/{pattern}"):
                    if match.is_dir():
                        remove_path(match, f"Build directory {pattern}")


def main():
    """Main cleanup function."""
    print("=" * 60)
    print("GitHub Actions Disk Space Cleanup Script")
    print("=" * 60)
    
    # Show initial disk usage
    get_disk_usage()
    
    # Run all cleanup functions
    clean_apt_packages()
    clean_docker()
    clean_large_directories()
    clean_caches()
    clean_temp_files()
    clean_github_workspace()
    
    # Show final disk usage
    print("\n" + "=" * 60)
    print("Cleanup complete!")
    print("=" * 60)
    get_disk_usage()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
