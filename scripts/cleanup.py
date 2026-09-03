#!/usr/bin/env python3
"""
Cleanup script for the RAG chatbot application.
Removes temporary files, logs, and data directories.
"""

import os
import shutil
import argparse
from pathlib import Path


def cleanup_data_directories():
    """Clean up data directories."""
    data_dirs = ["data/chunks", "data/vectors", "data/temp", "data/logs"]
    
    for directory in data_dirs:
        if os.path.exists(directory):
            print(f"🧹 Cleaning {directory}...")
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            print(f" Cleaned {directory}")
        else:
            print(f" Directory not found: {directory}")


def cleanup_logs():
    """Clean up log files."""
    log_dir = "data/logs"
    if os.path.exists(log_dir):
        print(f"🧹 Cleaning logs in {log_dir}...")
        for file in os.listdir(log_dir):
            if file.endswith('.log'):
                os.remove(os.path.join(log_dir, file))
        print(" Logs cleaned")
    else:
        print(" Log directory not found")


def cleanup_temp_files():
    """Clean up temporary files."""
    temp_dirs = ["data/temp", "__pycache__"]
    
    for directory in temp_dirs:
        if os.path.exists(directory):
            print(f"🧹 Cleaning {directory}...")
            shutil.rmtree(directory)
            os.makedirs(directory, exist_ok=True)
            print(f" Cleaned {directory}")


def cleanup_pycache():
    """Clean up Python cache files."""
    print("🧹 Cleaning Python cache files...")
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                shutil.rmtree(os.path.join(root, dir_name))
    print(" Python cache cleaned")


def main():
    """Main cleanup function."""
    parser = argparse.ArgumentParser(description="Cleanup RAG chatbot data")
    parser.add_argument("--all", action="store_true", help="Clean all data")
    parser.add_argument("--logs", action="store_true", help="Clean logs only")
    parser.add_argument("--temp", action="store_true", help="Clean temp files only")
    parser.add_argument("--cache", action="store_true", help="Clean Python cache only")
    
    args = parser.parse_args()
    
    if not any([args.all, args.logs, args.temp, args.cache]):
        args.all = True
    
    print("🧹 Starting cleanup...")
    print("=" * 40)
    
    if args.all or args.temp:
        cleanup_data_directories()
        cleanup_temp_files()
    
    if args.all or args.logs:
        cleanup_logs()
    
    if args.all or args.cache:
        cleanup_pycache()
    
    print("=" * 40)
    print(" Cleanup completed!")


if __name__ == "__main__":
    main()

