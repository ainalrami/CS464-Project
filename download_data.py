#!/usr/bin/env python3
"""
EuroSAT RGB Dataset Downloader
================================

Downloads the EuroSAT RGB dataset and extracts it to ./data/EuroSAT_RGB/

Usage:
    python download_data.py
"""

import os
import ssl
import sys
import zipfile
import urllib.request
import shutil

# Bypass SSL certificate verification (common issue on macOS with Homebrew Python)
ssl._create_default_https_context = ssl._create_unverified_context

# EuroSAT RGB download URL
EUROSAT_URL = "https://madm.dfki.de/files/sentinel/EuroSAT.zip"
DATA_DIR = "./data"
ZIP_PATH = os.path.join(DATA_DIR, "EuroSAT.zip")
EXTRACT_DIR = os.path.join(DATA_DIR)
FINAL_DIR = os.path.join(DATA_DIR, "EuroSAT_RGB")


def download_with_progress(url, dest):
    """Download a file with a progress indicator."""
    print(f"Downloading from: {url}")
    print(f"Saving to: {dest}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
    print("\n  Download complete!")


def main():
    # Check if dataset already exists
    if os.path.exists(FINAL_DIR) and len(os.listdir(FINAL_DIR)) >= 10:
        print(f"Dataset already exists at {FINAL_DIR} with {len(os.listdir(FINAL_DIR))} classes.")
        print("Skipping download. Delete the folder to re-download.")
        return

    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)

    # Download
    if not os.path.exists(ZIP_PATH):
        download_with_progress(EUROSAT_URL, ZIP_PATH)
    else:
        print(f"ZIP file already exists: {ZIP_PATH}")

    # Extract
    print(f"Extracting to {EXTRACT_DIR}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print("  Extraction complete!")

    # The ZIP extracts to data/2750/ — rename to EuroSAT_RGB
    extracted_dir = os.path.join(DATA_DIR, "2750")
    if os.path.exists(extracted_dir) and not os.path.exists(FINAL_DIR):
        shutil.move(extracted_dir, FINAL_DIR)
        print(f"  Renamed {extracted_dir} -> {FINAL_DIR}")
    elif os.path.exists(os.path.join(DATA_DIR, "EuroSAT")):
        # Some versions extract to EuroSAT/
        src = os.path.join(DATA_DIR, "EuroSAT")
        if not os.path.exists(FINAL_DIR):
            shutil.move(src, FINAL_DIR)
            print(f"  Renamed {src} -> {FINAL_DIR}")

    # Verify
    if os.path.exists(FINAL_DIR):
        classes = sorted([d for d in os.listdir(FINAL_DIR)
                         if os.path.isdir(os.path.join(FINAL_DIR, d))])
        total_images = sum(
            len([f for f in os.listdir(os.path.join(FINAL_DIR, c))
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))])
            for c in classes
        )
        print(f"\n✅ Dataset ready!")
        print(f"   Location: {FINAL_DIR}")
        print(f"   Classes ({len(classes)}): {classes}")
        print(f"   Total images: {total_images}")
    else:
        print(f"\n⚠️  Could not find extracted dataset at {FINAL_DIR}")
        print("   Please check the extracted contents and rename manually.")

    # Clean up ZIP
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
        print(f"   Cleaned up {ZIP_PATH}")


if __name__ == "__main__":
    main()
