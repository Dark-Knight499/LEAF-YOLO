import os
import glob
from PIL import Image

def convert_visdrone_to_yolo(dataset_path):
    """
    Converts VisDrone format annotations to YOLO format.

    This script reads annotations from the 'labels' directory, which are expected
    to be in the VisDrone format (bbox_left,bbox_top,width,height,score,category,truncation,occlusion),
    and converts them to YOLO format (class_id x_center y_center width height), normalized by image size.

    The original 'labels' directory will be renamed to 'labels_backup'.
    A new 'labels' directory will be created with the YOLO formatted annotation files.

    Args:
        dataset_path (str): The path to the dataset directory (e.g., 'C:/.../VisDrone2019-DET-val/VisDrone2019-DET-val').
    """
    labels_dir = os.path.join(dataset_path, 'labels')
    images_dir = os.path.join(dataset_path, 'images')

    # --- 1. Validate paths ---
    if not os.path.exists(labels_dir):
        print(f"Error: 'labels' directory not found at: {labels_dir}")
        print("Please ensure the 'annotations' folder has been renamed to 'labels'.")
        return
    if not os.path.exists(images_dir):
        print(f"Error: 'images' directory not found at: {images_dir}")
        return

    # --- 2. Backup original labels ---
    backup_dir = os.path.join(dataset_path, 'labels_backup')
    if os.path.exists(backup_dir):
        print(f"Backup directory '{backup_dir}' already exists. Skipping backup.")
    else:
        os.rename(labels_dir, backup_dir)
        print(f"Original labels backed up to: {backup_dir}")

    # --- 3. Create new labels directory ---
    os.makedirs(labels_dir, exist_ok=True)

    # --- 4. Get list of annotation files ---
    annotation_files = glob.glob(os.path.join(backup_dir, '*.txt'))
    if not annotation_files:
        print(f"Warning: No .txt files found in {backup_dir}")
        return

    print(f"Found {len(annotation_files)} annotation files. Starting conversion...")

    # --- 5. Process each annotation file ---
    for ann_file in annotation_files:
        file_name = os.path.basename(ann_file)
        image_name = os.path.splitext(file_name)[0] + '.jpg'
        image_path = os.path.join(images_dir, image_name)

        if not os.path.exists(image_path):
            print(f"Warning: Corresponding image not found for {file_name}. Skipping.")
            continue

        # Get image dimensions
        with Image.open(image_path) as img:
            img_w, img_h = img.size

        yolo_annotations = []
        with open(ann_file, 'r') as f:
            for line in f.readlines():
                try:
                    parts = [int(p) for p in line.strip().split(',')]
                    # VisDrone format: <bbox_left>,<bbox_top>,<width>,<height>,<score>,<category>,<truncation>,<occlusion>
                    
                    # Skip ignored regions (category 0) and others (category 11)
                    visdrone_class_id = parts[5]
                    if visdrone_class_id in [0, 11]:
                        continue

                    # Convert to YOLO format
                    # YOLO class IDs are 0-indexed. VisDrone 'pedestrian' is 1, so we subtract 1.
                    yolo_class_id = visdrone_class_id - 1

                    x, y, w, h = parts[0], parts[1], parts[2], parts[3]

                    # Normalize
                    x_center = (x + w / 2) / img_w
                    y_center = (y + h / 2) / img_h
                    norm_w = w / img_w
                    norm_h = h / img_h

                    yolo_annotations.append(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping corrupted or malformed line in {file_name}: '{line.strip()}' -> {e}")
                    continue
        
        # Write new annotation file
        if yolo_annotations:
            new_ann_path = os.path.join(labels_dir, file_name)
            with open(new_ann_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))

    print("Conversion complete!")
    print(f"New YOLO-formatted labels are in: {labels_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert VisDrone annotations to YOLO format.')
    parser.add_argument('dataset_path', help='Path to the dataset directory containing "images" and "labels" folders.')
    args = parser.parse_args()
    
    convert_visdrone_to_yolo(args.dataset_path)
