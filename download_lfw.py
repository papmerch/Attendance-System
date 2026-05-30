"""
download_lfw.py — Download and enroll the LFW (Labeled Faces in the Wild) dataset.

The LFW dataset contains ~13,000 labeled face images of ~5,749 celebrities.
This script downloads it, extracts it, and places images into known_faces/lfw/
so that face_recognizer.py can enroll them at startup.

Usage:
    python download_lfw.py

Requirements:
    - Internet connection (first download ~230 MB)
"""

import os
import sys
import tarfile
import urllib.request
import urllib.error

KNOWN_FACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "known_faces")

# Mirrors in order of preference (first working one is used)
LFW_MIRRORS = [
    # scikit-learn's figshare mirror (most reliable as of 2026)
    "https://ndownloader.figshare.com/files/5976015",
    # Official UMass server (sometimes down)
    "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz",
]

LFW_TGZ = "/tmp/lfw.tgz"


def download_file(url, dest):
    """Download a file with a progress indicator."""
    def report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, int(downloaded * 100 / total_size))
        bar = "█" * (percent // 5) + "░" * (20 - percent // 5)
        sys.stdout.write(f"\r  [{bar}] {percent}% ({downloaded // 1024 // 1024} MB / {total_size // 1024 // 1024} MB)")
        sys.stdout.flush()

    print(f"[INFO] Downloading from:\n  {url}")
    urllib.request.urlretrieve(url, dest, reporthook=report)
    print()


def extract_lfw(tgz_path, dest_dir):
    """
    Extract the LFW archive into the known_faces directory.

    The archive contains a root folder (either 'lfw_funneled' or 'lfw').
    We rename it to 'lfw' for consistency.
    """
    lfw_target = os.path.join(dest_dir, "lfw")
    if os.path.exists(lfw_target):
        print(f"[INFO] LFW directory already exists at {lfw_target}")
        print("  Delete it if you want to re-download and re-extract.")
        return lfw_target

    print(f"[INFO] Extracting archive (this may take a minute)...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        # Determine the root folder name inside the archive
        members = tar.getmembers()
        root = os.path.commonpath(m.name for m in members)
        tar.extractall(path=dest_dir)

        # Rename root folder to 'lfw' if it has a different name
        extracted_root = os.path.join(dest_dir, root)
        if root != "lfw" and os.path.isdir(extracted_root):
            os.rename(extracted_root, lfw_target)

    print("[INFO] Extraction complete.")

    # Count people and images
    person_count = 0
    image_count = 0
    for person_dir in os.listdir(lfw_target):
        person_path = os.path.join(lfw_target, person_dir)
        if os.path.isdir(person_path):
            person_count += 1
            image_count += len(
                [f for f in os.listdir(person_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            )

    print(f"\n[SUMMARY] Enrolled {person_count} people with {image_count} total images from LFW.")
    return lfw_target


def main():
    """Download LFW and extract it into known_faces/."""
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)

    # Step 1: Download if not already cached locally
    if os.path.exists(LFW_TGZ):
        print(f"[INFO] Using cached download at {LFW_TGZ}")
    else:
        # Try each mirror in order until one works
        downloaded = False
        for url in LFW_MIRRORS:
            try:
                download_file(url, LFW_TGZ)
                downloaded = True
                break
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                print(f"  [WARN] Mirror failed: {e}")
                print(f"  Trying next mirror...\n")
                continue

        if not downloaded:
            print()
            print("[ERROR] All download mirrors failed.")
            print()
            print("To complete setup manually:")
            print("  1. Download LFW dataset from one of these URLs:")
            print("     - https://ndownloader.figshare.com/files/5976015")
            print("     - http://vis-www.cs.umass.edu/lfw/")
            print("  2. Save the file to:", LFW_TGZ)
            print("  3. Re-run this script.")
            print()
            print("Alternatively, skip LFW and add your own images to:")
            print("  ", KNOWN_FACES_DIR)
            print("  (Name them like: alice.jpg, bob.jpg, ...)")
            sys.exit(1)

    # Step 2: Extract
    lfw_path = extract_lfw(LFW_TGZ, KNOWN_FACES_DIR)

    # Step 3: Cleanup the tgz to save space
    os.remove(LFW_TGZ)
    print(f"[INFO] Removed temporary file {LFW_TGZ}")

    print(f"\n[NEXT] Run 'python face_recognizer.py' to start recognition with LFW faces.")
    print(f"  Or add your own images to {KNOWN_FACES_DIR}")


if __name__ == "__main__":
    main()
