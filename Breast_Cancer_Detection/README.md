# YOLO Prediction Pipeline

This script runs YOLO object detection predictions on a dataset of mammography images and saves the results to the `pred` directory.

## Features

- **Automatic Processing**: Processes all images in the sample dataset
- **Comprehensive Outputs**: Saves prediction images, results JSON, and summary reports
- **Error Handling**: Robust error handling with detailed logging
- **Visualization**: Creates annotated images with bounding boxes and confidence scores
- **Statistics**: Generates detailed summary reports with detection statistics

## Directory Structure

```
├── main.py                                    # Main prediction script
├── requirements.txt                           # Python dependencies
├── Yolo_Detection_Mamografi_31.08.2025.pt   # YOLO model file
├── mamografi_sample_dataset_31.08.2025/      # Input images directory
└── pred/                                     # Output directory
    ├── images/                               # Prediction images with boxes
    ├── results/                              # JSON results and metadata
    └── reports/                              # Summary reports
```

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify files exist:**
   - YOLO model: `Yolo_Detection_Mamografi_31.08.2025.pt`
   - Sample dataset: `mamografi_sample_dataset_31.08.2025/`
   - Output directory: `pred/` (will be created automatically)

## Usage

### Run the complete pipeline:
```bash
python main.py
```

### What the script does:

1. **Loads the YOLO model** from the `.pt` file
2. **Scans the input directory** for all supported image formats (PNG, JPG, JPEG, BMP, TIFF)
3. **Runs predictions** on each image using the loaded model
4. **Saves annotated images** with bounding boxes and labels to `pred/images/`
5. **Stores detailed results** in JSON format to `pred/results/`
6. **Generates summary reports** with statistics to `pred/reports/`

## Output Files

### 1. Prediction Images (`pred/images/`)
- Original images with drawn bounding boxes
- Labels showing class names and confidence scores
- Filename format: `pred_[original_name].png`

### 2. Results JSON (`pred/results/predictions.json`)
- Complete prediction data for all images
- Bounding box coordinates, confidence scores, class IDs
- Processing metadata and timing information

### 3. Summary Report (`pred/reports/summary_report.txt`)
- Processing statistics (total images, success rate)
- Detection counts by class
- Error summaries for failed images
- Performance metrics

## Supported Image Formats

- PNG (`.png`)
- JPEG (`.jpg`, `.jpeg`)
- BMP (`.bmp`)
- TIFF (`.tiff`, `.tif`)

## Model Information

- **Model File**: `Yolo_Detection_Mamografi_31.08.2025.pt`
- **Type**: YOLO object detection model
- **Domain**: Mammography image analysis
- **Classes**: Automatically detected from the model

## Error Handling

The script includes comprehensive error handling:
- Continues processing if individual images fail
- Logs all errors with detailed information
- Reports success/failure rates in summary
- Graceful degradation for corrupted images

## Performance

- **Batch Processing**: Processes images sequentially for memory efficiency
- **Progress Tracking**: Shows progress for each image being processed
- **Timing Information**: Reports processing time for each image
- **Memory Management**: Closes matplotlib figures to prevent memory leaks

## Customization

To modify the script for different datasets:

1. **Change input directory**: Modify `INPUT_DIR` in the `main()` function
2. **Change model path**: Modify `MODEL_PATH` in the `main()` function
3. **Change output directory**: Modify `OUTPUT_DIR` in the `main()` function
4. **Adjust confidence thresholds**: Modify the prediction logic in `predict_single_image()`
5. **Change visualization style**: Modify the plotting code in `save_prediction_image()`

## Troubleshooting

### Common Issues:

1. **Model loading fails**: Check if the `.pt` file exists and is not corrupted
2. **Memory errors**: Reduce image batch size or use smaller images
3. **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
4. **Permission errors**: Check write permissions for the output directory

### Debug Mode:

The script provides detailed logging. Check console output for:
- Model loading status
- Image processing progress
- Error messages and stack traces
- Final summary statistics

## Example Output

```
============================================================
YOLO PREDICTION PIPELINE
============================================================
Loading YOLO model from: Yolo_Detection_Mamografi_31.08.2025.pt
Model loaded successfully!

Starting prediction on all images in: mamografi_sample_dataset_31.08.2025
Found 30 image files

[1/30] Processing: Kategori5Sol_12664_LMLO.png
Processing: Kategori5Sol_12664_LMLO.png
Completed: Kategori5Sol_12664_LMLO.png
...

Pipeline completed successfully!
Total Duration: 0:02:15
Results saved to: pred
============================================================
```

## License

This script is provided as-is for research and development purposes.
