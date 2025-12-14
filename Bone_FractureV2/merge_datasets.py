"""
Merge two YOLO datasets into a unified dataset.
Follows the plan: preserves original splits and remaps class IDs.
"""

import os
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict

# Base directory
BASE_DIR = Path(__file__).parent

# Source dataset paths
DATASET2_PATH = BASE_DIR / "BoneFractureYolo8"
DATASET3_PATH = BASE_DIR / "Human Bone Fractures Multi-modal Image Dataset (HBFMID)" / "Bone Fractures Detection"

# Output directory
OUTPUT_DIR = BASE_DIR / "Merged"

# Unified class list (17 classes total)
# Classes 0-6: From dataset 2
CLASSES_0_6 = [
    'elbow positive',
    'fingers positive',
    'forearm fracture',
    'humerus fracture',
    'humerus',
    'shoulder fracture',
    'wrist positive'
]

# Classes 7-16: From dataset 3
CLASSES_7_16 = [
    'Comminuted',
    'Greenstick',
    'Healthy',
    'Linear',
    'Oblique Displaced',
    'Oblique',
    'Segmental',
    'Spiral',
    'Transverse Displaced',
    'Transverse'
]

UNIFIED_CLASSES = CLASSES_0_6 + CLASSES_7_16


def get_file_hash(file_path):
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def remap_label_file(label_path, offset):
    """
    Remap class IDs in a label file by adding offset.
    Returns list of remapped lines.
    """
    remapped_lines = []
    if not os.path.exists(label_path):
        return remapped_lines
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                # Get class ID (first value)
                class_id = int(parts[0])
                new_class_id = class_id + offset
                
                # Reconstruct line with new class ID
                new_line = f"{new_class_id} {' '.join(parts[1:])}"
                remapped_lines.append(new_line)
    except Exception as e:
        print(f"Warning: Could not read {label_path}: {e}")
    
    return remapped_lines


def copy_dataset_split(source_dir, output_dir, split_name, dataset_num, remap_offset=0, seen_hashes=None):
    """
    Copy images and labels from a dataset split.
    
    Args:
        source_dir: Source dataset directory
        output_dir: Output directory (Merged)
        split_name: 'train', 'valid', or 'test'
        dataset_num: Dataset number (1, 2, or 3)
        remap_offset: Offset to add to class IDs (0 for dataset 2, 7 for dataset 3)
        seen_hashes: Set of file hashes already copied (to detect duplicates)
    
    Returns:
        Tuple of (images_copied, labels_copied, duplicates_skipped)
    """
    source_images_dir = source_dir / split_name / "images"
    source_labels_dir = source_dir / split_name / "labels"
    
    output_images_dir = output_dir / split_name / "images"
    output_labels_dir = output_dir / split_name / "labels"
    
    if not source_images_dir.exists():
        print(f"  Warning: {source_images_dir} does not exist, skipping...")
        return 0, 0, 0
    
    images_copied = 0
    labels_copied = 0
    duplicates_skipped = 0
    
    # Get all image files
    image_files = list(source_images_dir.glob("*.jpg")) + list(source_images_dir.glob("*.JPG")) + \
                  list(source_images_dir.glob("*.jpeg")) + list(source_images_dir.glob("*.JPEG")) + \
                  list(source_images_dir.glob("*.png")) + list(source_images_dir.glob("*.PNG"))
    
    for img_path in image_files:
        # Check for duplicates using file hash
        img_hash = get_file_hash(img_path)
        if seen_hashes is not None and img_hash in seen_hashes:
            duplicates_skipped += 1
            continue
        
        if seen_hashes is not None:
            seen_hashes.add(img_hash)
        
        # Generate unique filename with dataset prefix
        img_stem = img_path.stem
        img_ext = img_path.suffix
        new_filename = f"ds{dataset_num}_{img_stem}{img_ext}"
        
        # Copy image
        dest_img_path = output_images_dir / new_filename
        shutil.copy2(img_path, dest_img_path)
        images_copied += 1
        
        # Handle corresponding label file
        label_path = source_labels_dir / f"{img_stem}.txt"
        if label_path.exists():
            # Remap labels if needed
            if remap_offset > 0:
                remapped_lines = remap_label_file(label_path, remap_offset)
                dest_label_path = output_labels_dir / f"ds{dataset_num}_{img_stem}.txt"
                with open(dest_label_path, 'w') as f:
                    f.write('\n'.join(remapped_lines))
                    if remapped_lines:  # Add newline if file has content
                        f.write('\n')
            else:
                # Copy label as-is
                dest_label_path = output_labels_dir / f"ds{dataset_num}_{img_stem}.txt"
                shutil.copy2(label_path, dest_label_path)
            labels_copied += 1
        else:
            # Create empty label file if image has no labels
            dest_label_path = output_labels_dir / f"ds{dataset_num}_{img_stem}.txt"
            dest_label_path.touch()
    
    return images_copied, labels_copied, duplicates_skipped


def create_directory_structure(output_dir):
    """Create the Merged directory structure."""
    for split in ['train', 'valid', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    print(f"Created directory structure in {output_dir}")


def create_data_yaml(output_dir):
    """Create the unified data.yaml file."""
    yaml_content = f"""train: train/images
val: valid/images
test: test/images

nc: {len(UNIFIED_CLASSES)}
names: {UNIFIED_CLASSES}
"""
    
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"Created {yaml_path}")


def validate_merged_dataset(output_dir):
    """Validate the merged dataset."""
    print("\nValidating merged dataset...")
    
    stats = {
        'splits': defaultdict(lambda: {'images': 0, 'labels': 0, 'empty_labels': 0}),
        'class_counts': defaultdict(int),
        'invalid_labels': []
    }
    
    for split in ['train', 'valid', 'test']:
        images_dir = output_dir / split / 'images'
        labels_dir = output_dir / split / 'labels'
        
        # Count images
        image_files = list(images_dir.glob("*.*"))
        stats['splits'][split]['images'] = len(image_files)
        
        # Check labels
        for img_path in image_files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            if label_path.exists():
                stats['splits'][split]['labels'] += 1
                
                # Check label content
                try:
                    with open(label_path, 'r') as f:
                        content = f.read().strip()
                        if not content:
                            stats['splits'][split]['empty_labels'] += 1
                        else:
                            # Validate class IDs
                            for line in content.split('\n'):
                                if line.strip():
                                    parts = line.split()
                                    if len(parts) >= 5:
                                        class_id = int(parts[0])
                                        if 0 <= class_id < len(UNIFIED_CLASSES):
                                            stats['class_counts'][class_id] += 1
                                        else:
                                            stats['invalid_labels'].append((str(label_path), class_id))
                except Exception as e:
                    stats['invalid_labels'].append((str(label_path), str(e)))
            else:
                stats['splits'][split]['empty_labels'] += 1
    
    # Print validation results
    print("\nValidation Results:")
    print("=" * 60)
    for split in ['train', 'valid', 'test']:
        s = stats['splits'][split]
        print(f"\n{split.upper()}:")
        print(f"  Images: {s['images']}")
        print(f"  Labels: {s['labels']}")
        print(f"  Empty labels: {s['empty_labels']}")
    
    print(f"\nClass Distribution:")
    for class_id in sorted(stats['class_counts'].keys()):
        count = stats['class_counts'][class_id]
        class_name = UNIFIED_CLASSES[class_id]
        print(f"  Class {class_id} ({class_name}): {count} annotations")
    
    if stats['invalid_labels']:
        print(f"\nWarnings: {len(stats['invalid_labels'])} invalid labels found")
        for label_path, error in stats['invalid_labels'][:10]:
            print(f"  {label_path}: {error}")
    
    return stats


def main():
    """Main function to merge the datasets."""
    print("=" * 60)
    print("MERGING TWO YOLO DATASETS")
    print("=" * 60)
    
    # Check if source datasets exist
    if not DATASET2_PATH.exists():
        print(f"Error: Dataset 2 not found at {DATASET2_PATH}")
        return
    
    if not DATASET3_PATH.exists():
        print(f"Error: Dataset 3 not found at {DATASET3_PATH}")
        return
    
    # Create output directory structure
    print("\n[Step 1] Creating directory structure...")
    create_directory_structure(OUTPUT_DIR)
    
    # Track file hashes to detect duplicates
    # First, check existing files in Merged folder and add their hashes
    seen_hashes = set()
    if OUTPUT_DIR.exists():
        print("  Checking existing files in Merged folder...")
        existing_count = 0
        for split in ['train', 'valid', 'test']:
            images_dir = OUTPUT_DIR / split / 'images'
            if images_dir.exists():
                for img_path in images_dir.glob("*.*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        img_hash = get_file_hash(img_path)
                        seen_hashes.add(img_hash)
                        existing_count += 1
        if existing_count > 0:
            print(f"  Found {existing_count} existing files in Merged folder (will skip duplicates)")
        else:
            print("  Merged folder is empty, starting fresh")
    
    total_stats = defaultdict(int)
    
    # Process Dataset 2 (BoneFractureYolo8)
    print("\n[Step 2] Processing Dataset 2 (BoneFractureYolo8)...")
    print("  Class IDs: 0-6 (no remapping)")
    
    for split in ['train', 'valid', 'test']:
        images, labels, duplicates = copy_dataset_split(
            DATASET2_PATH, OUTPUT_DIR, split, 2, remap_offset=0, seen_hashes=seen_hashes
        )
        print(f"  {split}: {images} images, {labels} labels, {duplicates} duplicates skipped")
        total_stats['images'] += images
        total_stats['labels'] += labels
        total_stats['duplicates'] += duplicates
    
    # Process Dataset 3 (HBFMID)
    print("\n[Step 3] Processing Dataset 3 (HBFMID)...")
    print("  Class IDs: 0-9 -> 7-16 (remapping by adding 7)")
    
    for split in ['train', 'valid', 'test']:
        images, labels, duplicates = copy_dataset_split(
            DATASET3_PATH, OUTPUT_DIR, split, 3, remap_offset=7, seen_hashes=seen_hashes
        )
        print(f"  {split}: {images} images, {labels} labels, {duplicates} duplicates skipped")
        total_stats['images'] += images
        total_stats['labels'] += labels
        total_stats['duplicates'] += duplicates
    
    # Create data.yaml
    print("\n[Step 4] Creating data.yaml...")
    create_data_yaml(OUTPUT_DIR)
    
    # Validate merged dataset
    print("\n[Step 5] Validating merged dataset...")
    validation_stats = validate_merged_dataset(OUTPUT_DIR)
    
    # Print summary
    print("\n" + "=" * 60)
    print("MERGE SUMMARY")
    print("=" * 60)
    print(f"\nTotal images copied: {total_stats['images']}")
    print(f"Total labels copied: {total_stats['labels']}")
    print(f"Total duplicates skipped: {total_stats['duplicates']}")
    print(f"\nUnified classes: {len(UNIFIED_CLASSES)}")
    print(f"Classes 0-6: {CLASSES_0_6}")
    print(f"Classes 7-16: {CLASSES_7_16}")
    print("\n" + "=" * 60)
    print("Merge completed successfully!")
    print(f"\nMerged dataset location: {OUTPUT_DIR}")
    print(f"YAML file: {OUTPUT_DIR / 'data.yaml'}")


if __name__ == '__main__':
    main()
