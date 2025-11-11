#!/usr/bin/env python3
"""
Generate a CSV file listing all video files in a directory for V-JEPA training.

The output CSV format is:
    /path/to/video1.mp4 0
    /path/to/video2.avi 0
    ...

Usage:
    python generate_video_csv.py /path/to/video/dir output.csv
    python generate_video_csv.py /path/to/video/dir output.csv --recursive
    python generate_video_csv.py /path/to/video/dir output.csv --extensions mp4 avi mkv
"""

import argparse
import os
from pathlib import Path


# Common video file extensions
DEFAULT_VIDEO_EXTENSIONS = [
    'mp4', 'avi', 'mkv', 'mov', 'webm', 'flv', 'wmv', 'mpg', 'mpeg', 'm4v',
    'jpg', 'jpeg', 'png'  # Images are also supported
]


def find_videos(directory, extensions, recursive=True):
    """
    Find all video files in a directory.
    
    Args:
        directory: Root directory to search
        extensions: List of file extensions to include (without dot)
        recursive: Whether to search subdirectories
        
    Returns:
        List of absolute paths to video files
    """
    directory = Path(directory).resolve()
    video_files = []
    
    # Convert extensions to lowercase for case-insensitive matching
    extensions = [ext.lower().lstrip('.') for ext in extensions]
    
    if recursive:
        # Recursively search all subdirectories
        for ext in extensions:
            video_files.extend(directory.rglob(f'*.{ext}'))
            # Also search for uppercase extensions
            video_files.extend(directory.rglob(f'*.{ext.upper()}'))
    else:
        # Only search immediate directory
        for ext in extensions:
            video_files.extend(directory.glob(f'*.{ext}'))
            video_files.extend(directory.glob(f'*.{ext.upper()}'))
    
    # Remove duplicates and sort
    video_files = sorted(set(video_files))
    
    return [str(f) for f in video_files]


def generate_csv(video_dir, output_csv, label=0, recursive=True, extensions=None):
    """
    Generate a CSV file from a directory of videos.
    
    Args:
        video_dir: Directory containing video files
        output_csv: Output CSV file path
        label: Label to assign to all videos (default: 0 for unsupervised)
        recursive: Whether to search subdirectories recursively
        extensions: List of video extensions to include (default: common video formats)
    """
    if extensions is None:
        extensions = DEFAULT_VIDEO_EXTENSIONS
    
    # Find all video files
    print(f"Searching for videos in: {video_dir}")
    print(f"Extensions: {', '.join(extensions)}")
    print(f"Recursive: {recursive}")
    
    video_files = find_videos(video_dir, extensions, recursive)
    
    if not video_files:
        print(f"Warning: No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Write to CSV
    output_path = Path(output_csv).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for video_file in video_files:
            f.write(f"{video_file} {label}\n")
    
    print(f"Generated CSV file: {output_path}")
    print(f"Total entries: {len(video_files)}")
    
    # Show first few entries as preview
    print("\nPreview (first 5 entries):")
    for video_file in video_files[:5]:
        print(f"  {video_file} {label}")
    
    if len(video_files) > 5:
        print(f"  ... and {len(video_files) - 5} more")


def main():
    parser = argparse.ArgumentParser(
        description='Generate a CSV file listing video files for V-JEPA training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate CSV from a directory (recursive by default)
  python generate_video_csv.py /data/videos train_videos.csv
  
  # Only search immediate directory (not recursive)
  python generate_video_csv.py /data/videos train.csv --no-recursive
  
  # Specify custom extensions
  python generate_video_csv.py /data/videos train.csv --extensions mp4 avi
  
  # Use custom label (for supervised learning)
  python generate_video_csv.py /data/class1 class1.csv --label 1
        """
    )
    
    parser.add_argument(
        'video_dir',
        type=str,
        help='Directory containing video files'
    )
    
    parser.add_argument(
        'output_csv',
        type=str,
        help='Output CSV file path'
    )
    
    parser.add_argument(
        '--label',
        type=int,
        default=0,
        help='Label to assign to all videos (default: 0, for unsupervised learning)'
    )
    
    parser.add_argument(
        '--recursive',
        action='store_true',
        default=False,
        help='Search subdirectories recursively (default: False)'
    )
    
    parser.add_argument(
        '--extensions',
        type=str,
        nargs='+',
        default=None,
        help=f'Video file extensions to include (default: {" ".join(DEFAULT_VIDEO_EXTENSIONS[:5])} ...)'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.video_dir):
        print(f"Error: Directory not found: {args.video_dir}")
        return 1
    
    # Generate CSV
    generate_csv(
        video_dir=args.video_dir,
        output_csv=args.output_csv,
        label=args.label,
        recursive=args.recursive,
        extensions=args.extensions
    )
    
    return 0


if __name__ == '__main__':
    exit(main())

