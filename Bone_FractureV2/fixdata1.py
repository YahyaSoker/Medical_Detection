"""
Convert polygon segmentation annotations to YOLO bounding box format.
Processes BoneFractureYolo8 dataset label files.
"""

import os
from pathlib import Path
from collections import defaultdict

# Base directory
BASE_DIR = Path(__file__).parent

# Dataset path
DATASET_PATH = BASE_DIR / "BoneFractureYolo8"

# Splits to process
SPLITS = ['train', 'valid', 'test']


def convert_polygon_to_box(class_id, values):
    """
    Convert polygon coordinates to YOLO bounding box format.
    
    Args:
        class_id: Class ID (integer)
        values: List of coordinate values (floats)
    
    Returns:
        String in YOLO format: "class_id x_center y_center width height"
    """
    if len(values) == 4:
        # Already YOLO box format
        return f"{class_id} {values[0]:.6f} {values[1]:.6f} {values[2]:.6f} {values[3]:.6f}"
    
    if len(values) < 6 or len(values) % 2 != 0:
        # Invalid format - need at least 3 points (6 values) and even number
        raise ValueError(f"Invalid polygon format: {len(values)} values (need >= 6 and even)")
    
    # Extract x and y coordinates
    # Polygon format: x1 y1 x2 y2 x3 y3 ...
    xs = values[0::2]  # Even indices: 0, 2, 4, ...
    ys = values[1::2]  # Odd indices: 1, 3, 5, ...
    
    # Calculate bounding box
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    
    # Clamp coordinates to [0, 1] range to handle floating point precision issues
    x_min = max(0.0, min(1.0, x_min))
    x_max = max(0.0, min(1.0, x_max))
    y_min = max(0.0, min(1.0, y_min))
    y_max = max(0.0, min(1.0, y_max))
    
    # Ensure min < max after clamping
    if x_min >= x_max:
        x_max = min(1.0, x_min + 0.001)  # Add small width if needed
    if y_min >= y_max:
        y_max = min(1.0, y_min + 0.001)  # Add small height if needed
    
    # Convert to YOLO format (normalized center, width, height)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    # Validate bounds
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid bounding box: width={width}, height={height}")
    
    # Clamp center and dimensions to ensure they're within valid range
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    # Ensure box fits within [0, 1] bounds
    if x_center - width/2 < 0:
        x_center = width/2
    if x_center + width/2 > 1:
        x_center = 1 - width/2
    if y_center - height/2 < 0:
        y_center = height/2
    if y_center + height/2 > 1:
        y_center = 1 - height/2
    
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def process_label_file(label_path):
    """
    Process a single label file, converting polygon annotations to YOLO boxes.
    
    Args:
        label_path: Path to label file
    
    Returns:
        Tuple of (new_lines, was_converted, error_message)
        - new_lines: List of converted lines
        - was_converted: Boolean indicating if any conversion happened
        - error_message: Error message if any, None otherwise
    """
    new_lines = []
    was_converted = False
    
    if not label_path.exists():
        return [], False, f"File does not exist: {label_path}"
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Handle empty files
        if not lines or all(line.strip() == '' for line in lines):
            return [], False, None
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                # Preserve empty lines
                new_lines.append('')
                continue
            
            try:
                # Parse line
                parts = line.split()
                if len(parts) < 5:
                    # Skip lines with insufficient values
                    continue
                
                # Get class ID and values
                class_id = int(parts[0])
                values = list(map(float, parts[1:]))
                
                # Convert to YOLO box format
                converted_line = convert_polygon_to_box(class_id, values)
                new_lines.append(converted_line)
                
                # Check if conversion happened (more than 4 values means polygon)
                if len(values) > 4:
                    was_converted = True
                    
            except ValueError as e:
                error_msg = f"Line {line_num}: {str(e)}"
                return [], False, error_msg
            except Exception as e:
                error_msg = f"Line {line_num}: Unexpected error - {str(e)}"
                return [], False, error_msg
        
        return new_lines, was_converted, None
        
    except Exception as e:
        return [], False, f"File read error: {str(e)}"


def process_split(split_name, stats):
    """
    Process all label files in a split (train/valid/test).
    
    Args:
        split_name: Name of the split ('train', 'valid', or 'test')
        stats: Dictionary to update with statistics
    """
    labels_dir = DATASET_PATH / split_name / "labels"
    
    if not labels_dir.exists():
        print(f"  Warning: {labels_dir} does not exist, skipping...")
        return
    
    # Get all label files
    label_files = list(labels_dir.glob("*.txt"))
    
    if not label_files:
        print(f"  No label files found in {split_name}")
        return
    
    print(f"\n  Processing {split_name} split ({len(label_files)} files)...")
    
    converted_count = 0
    already_yolo_count = 0
    error_count = 0
    empty_count = 0
    
    for label_file in label_files:
        new_lines, was_converted, error = process_label_file(label_file)
        
        if error:
            error_count += 1
            print(f"    Error in {label_file.name}: {error}")
            continue
        
        if not new_lines:
            empty_count += 1
            # Write empty file
            with open(label_file, 'w') as f:
                f.write('')
            continue
        
        # Write converted labels
        with open(label_file, 'w') as f:
            f.write('\n'.join(new_lines))
            if new_lines:  # Add newline at end if file has content
                f.write('\n')
        
        if was_converted:
            converted_count += 1
        else:
            already_yolo_count += 1
    
    # Update statistics
    stats[split_name] = {
        'total': len(label_files),
        'converted': converted_count,
        'already_yolo': already_yolo_count,
        'empty': empty_count,
        'errors': error_count
    }
    
    print(f"    Converted: {converted_count}")
    print(f"    Already YOLO: {already_yolo_count}")
    print(f"    Empty: {empty_count}")
    print(f"    Errors: {error_count}")


def validate_conversion():
    """
    Validate that all converted files have valid YOLO format.
    """
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    
    validation_stats = {
        'valid_files': 0,
        'invalid_files': 0,
        'total_annotations': 0,
        'invalid_annotations': []
    }
    
    for split_name in SPLITS:
        labels_dir = DATASET_PATH / split_name / "labels"
        
        if not labels_dir.exists():
            continue
        
        label_files = list(labels_dir.glob("*.txt"))
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                file_valid = True
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    
                    # Check format: should have exactly 5 values (class_id + 4 box values)
                    if len(parts) != 5:
                        validation_stats['invalid_annotations'].append(
                            f"{label_file.name}: Line {line_num} has {len(parts)} values (expected 5)"
                        )
                        file_valid = False
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Validate ranges
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                            validation_stats['invalid_annotations'].append(
                                f"{label_file.name}: Line {line_num} center out of bounds"
                            )
                            file_valid = False
                        
                        if width <= 0 or height <= 0:
                            validation_stats['invalid_annotations'].append(
                                f"{label_file.name}: Line {line_num} invalid dimensions"
                            )
                            file_valid = False
                        
                        if x_center - width/2 < 0 or x_center + width/2 > 1:
                            validation_stats['invalid_annotations'].append(
                                f"{label_file.name}: Line {line_num} x bounds out of range"
                            )
                            file_valid = False
                        
                        if y_center - height/2 < 0 or y_center + height/2 > 1:
                            validation_stats['invalid_annotations'].append(
                                f"{label_file.name}: Line {line_num} y bounds out of range"
                            )
                            file_valid = False
                        
                        validation_stats['total_annotations'] += 1
                        
                    except ValueError:
                        validation_stats['invalid_annotations'].append(
                            f"{label_file.name}: Line {line_num} invalid number format"
                        )
                        file_valid = False
                
                if file_valid:
                    validation_stats['valid_files'] += 1
                else:
                    validation_stats['invalid_files'] += 1
                    
            except Exception as e:
                validation_stats['invalid_files'] += 1
                validation_stats['invalid_annotations'].append(
                    f"{label_file.name}: Read error - {str(e)}"
                )
    
    # Print validation results
    print(f"\nValid files: {validation_stats['valid_files']}")
    print(f"Invalid files: {validation_stats['invalid_files']}")
    print(f"Total annotations: {validation_stats['total_annotations']}")
    
    if validation_stats['invalid_annotations']:
        print(f"\nInvalid annotations found: {len(validation_stats['invalid_annotations'])}")
        print("First 10 errors:")
        for error in validation_stats['invalid_annotations'][:10]:
            print(f"  {error}")
    else:
        print("\nAll annotations are valid!")
    
    return validation_stats


def main():
    """Main function."""
    print("=" * 60)
    print("CONVERTING POLYGON ANNOTATIONS TO YOLO BOXES")
    print("=" * 60)
    
    # Check if dataset exists
    if not DATASET_PATH.exists():
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return
    
    # Statistics
    stats = {}
    
    # Process each split
    print("\n[Step 1] Processing label files...")
    for split_name in SPLITS:
        process_split(split_name, stats)
    
    # Print summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    
    total_files = 0
    total_converted = 0
    total_already_yolo = 0
    total_empty = 0
    total_errors = 0
    
    for split_name in SPLITS:
        if split_name in stats:
            s = stats[split_name]
            total_files += s['total']
            total_converted += s['converted']
            total_already_yolo += s['already_yolo']
            total_empty += s['empty']
            total_errors += s['errors']
            
            print(f"\n{split_name.upper()}:")
            print(f"  Total files: {s['total']}")
            print(f"  Converted (polygon -> box): {s['converted']}")
            print(f"  Already YOLO format: {s['already_yolo']}")
            print(f"  Empty files: {s['empty']}")
            print(f"  Errors: {s['errors']}")
    
    print(f"\nOVERALL:")
    print(f"  Total files processed: {total_files}")
    print(f"  Converted: {total_converted}")
    print(f"  Already YOLO: {total_already_yolo}")
    print(f"  Empty: {total_empty}")
    print(f"  Errors: {total_errors}")
    
    # Validate conversion
    print("\n[Step 2] Validating converted labels...")
    validation_stats = validate_conversion()
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
