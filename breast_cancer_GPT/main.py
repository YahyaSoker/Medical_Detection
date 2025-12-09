import os
import cv2
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from ultralytics import YOLO
import shutil

# Optional: local LLM for explanations
try:
    import llm_local
    _LLM_AVAILABLE = True
except Exception:
    _LLM_AVAILABLE = False

class YOLOPredictor:
    def __init__(self, model_path, input_dir, output_dir):
        """
        Initialize YOLO Predictor
        
        Args:
            model_path (str): Path to the .pt YOLO model file
            input_dir (str): Directory containing input images
            output_dir (str): Directory to save prediction results
        """
        self.model_path = model_path
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different output types
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Load the YOLO model
        print(f"Loading YOLO model from: {model_path}")
        try:
            self.model = YOLO(model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Get supported image extensions
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Initialize results storage
        self.all_results = []
        
    def get_image_files(self):
        """Get all image files from input directory"""
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(self.input_dir.glob(f"*{ext}"))
            image_files.extend(self.input_dir.glob(f"*{ext.upper()}"))
        return sorted(image_files)
    
    def predict_single_image(self, image_path):
        """
        Run prediction on a single image
        
        Args:
            image_path (Path): Path to the image file
            
        Returns:
            dict: Prediction results and metadata
        """
        try:
            print(f"Processing: {image_path.name}")
            
            # Run prediction
            results = self.model(str(image_path))
            
            # Get the first result (assuming single image input)
            result = results[0]
            
            # Extract prediction data
            boxes = result.boxes
            if boxes is not None:
                # Get coordinates, confidence scores, and class IDs
                xyxy = boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                conf = boxes.conf.cpu().numpy()  # confidence scores
                cls = boxes.cls.cpu().numpy()    # class IDs
                
                # Get class names
                class_names = [result.names[int(c)] for c in cls]
                
                # Create detection list
                detections = []
                for i in range(len(xyxy)):
                    detection = {
                        'bbox': xyxy[i].tolist(),  # [x1, y1, x2, y2]
                        'confidence': float(conf[i]),
                        'class_id': int(cls[i]),
                        'class_name': class_names[i]
                    }
                    detections.append(detection)
            else:
                detections = []
            
            # Create result dictionary
            result_data = {
                'image_name': image_path.name,
                'image_path': str(image_path),
                'total_detections': len(detections),
                'detections': detections,
                'processing_time': getattr(result, 'speed', {}).get('inference', 0),
                'image_size': {
                    'width': result.orig_shape[1],
                    'height': result.orig_shape[0]
                }
            }
            
            # If there are detections, optionally generate a user-facing explanation via local LLM
            if detections and _LLM_AVAILABLE:
                try:
                    prompt = self._build_explanation_prompt(result_data)
                    explanation = llm_local.generate(
                        prompt=prompt,
                        model_full_path=getattr(llm_local, 'DEFAULT_MODEL_PATH', ''),
                        max_tokens=300,
                        temperature=0.2,
                        top_p=0.9,
                        top_k=40,
                        repeat_penalty=1.05,
                        system_prompt=(
                            "You are a concise, careful assistant. Explain model detections in plain,"
                            " non-diagnostic terms. Avoid medical advice; describe what the model found"
                            " and suggest consulting a specialist for interpretation."
                        ),
                        stream=False,
                    )
                    result_data['llm_explanation'] = explanation.strip()
                except Exception as e:
                    # Do not fail the pipeline if LLM isn't available/working
                    result_data['llm_explanation_error'] = str(e)

            return result_data
            
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            return {
                'image_name': image_path.name,
                'image_path': str(image_path),
                'error': str(e),
                'total_detections': 0,
                'detections': []
            }
    
    def save_prediction_image(self, image_path, result_data):
        """
        Save image with prediction boxes drawn
        
        Args:
            image_path (Path): Path to the original image
            result_data (dict): Prediction results
        """
        try:
            # Read the original image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Could not read image: {image_path}")
                return
            
            # Convert BGR to RGB for matplotlib
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(image_rgb)
            
            # Draw bounding boxes
            for detection in result_data['detections']:
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                
                # Draw rectangle
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, color='red', linewidth=2)
                ax.add_patch(rect)
                
                # Add label
                label = f"{detection['class_name']} ({detection['confidence']:.2f})"
                ax.text(x1, y1-10, label, color='red', fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Set title
            ax.set_title(f"Predictions: {image_path.name}\n"
                        f"Total Detections: {result_data['total_detections']}")
            ax.axis('off')
            
            # Save the image
            output_path = self.images_dir / f"pred_{image_path.stem}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved prediction image: {output_path}")
            
        except Exception as e:
            print(f"Error saving prediction image for {image_path.name}: {e}")
    
    def process_all_images(self):
        """Process all images in the input directory"""
        print(f"\nStarting prediction on all images in: {self.input_dir}")
        
        # Get all image files
        image_files = self.get_image_files()
        print(f"Found {len(image_files)} image files")
        
        if not image_files:
            print("No image files found!")
            return
        
        # Process each image
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
            
            # Run prediction
            result_data = self.predict_single_image(image_path)
            
            # Store results
            self.all_results.append(result_data)
            
            # Save prediction image
            if 'error' not in result_data:
                self.save_prediction_image(image_path, result_data)
            
            print(f"Completed: {image_path.name}")
        
        print(f"\nCompleted processing {len(image_files)} images!")
    
    def save_results_json(self):
        """Save all results to JSON file"""
        output_path = self.results_dir / "predictions.json"
        
        # Add metadata
        results_data = {
            'model_path': str(self.model_path),
            'input_directory': str(self.input_dir),
            'output_directory': str(self.output_dir),
            'processing_date': datetime.now().isoformat(),
            'total_images_processed': len(self.all_results),
            'results': self.all_results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved results to: {output_path}")
        return output_path
    
    def generate_summary_report(self):
        """Generate a summary report of all predictions"""
        if not self.all_results:
            print("No results to generate report from!")
            return
        
        # Calculate statistics
        total_images = len(self.all_results)
        successful_images = len([r for r in self.all_results if 'error' not in r])
        failed_images = total_images - successful_images
        
        total_detections = sum(r.get('total_detections', 0) for r in self.all_results if 'error' not in r)
        
        # Count detections by class
        class_counts = {}
        for result in self.all_results:
            if 'error' not in result:
                for detection in result['detections']:
                    class_name = detection['class_name']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Generate report
        report_lines = [
            "YOLO PREDICTION SUMMARY REPORT",
            "=" * 50,
            f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model: {self.model_path}",
            f"Input Directory: {self.input_dir}",
            f"Output Directory: {self.output_dir}",
            "",
            "PROCESSING STATISTICS",
            "-" * 30,
            f"Total Images: {total_images}",
            f"Successfully Processed: {successful_images}",
            f"Failed: {failed_images}",
            f"Success Rate: {(successful_images/total_images)*100:.1f}%",
            "",
            "DETECTION STATISTICS",
            "-" * 30,
            f"Total Detections: {total_detections}",
            f"Average Detections per Image: {total_detections/successful_images:.2f}" if successful_images > 0 else "N/A",
            "",
            "DETECTIONS BY CLASS",
            "-" * 30
        ]
        
        for class_name, count in sorted(class_counts.items()):
            report_lines.append(f"{class_name}: {count}")
        
        # Add failed images if any
        if failed_images > 0:
            report_lines.extend([
                "",
                "FAILED IMAGES",
                "-" * 30
            ])
            for result in self.all_results:
                if 'error' in result:
                    report_lines.append(f"{result['image_name']}: {result['error']}")
        
        # Save report
        report_path = self.reports_dir / "summary_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Saved summary report to: {report_path}")
        
        # Print report to console
        print("\n" + "\n".join(report_lines))
        
        return report_path

    def _build_explanation_prompt(self, result_data: dict) -> str:
        """Build a concise prompt for the local LLM to explain detections."""
        image_name = result_data.get('image_name', 'image')
        width = result_data.get('image_size', {}).get('width')
        height = result_data.get('image_size', {}).get('height')
        lines = [
            f"Image: {image_name} ({width}x{height})",
            f"Total detections: {result_data.get('total_detections', 0)}",
            "Detections:",
        ]
        for det in result_data.get('detections', []):
            cls = det.get('class_name', 'object')
            conf = det.get('confidence', 0.0)
            x1, y1, x2, y2 = det.get('bbox', [None, None, None, None])
            lines.append(f"- {cls} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}] (conf {conf:.2f})")

        lines.append(
            "You are an assistant that supports clinicians and radiologists by summarizing image- "
            "and data-driven tumor-detection results. You are not a clinician and must never give "
            "definitive diagnoses or medical advice. Your role is to interpret algorithm outputs, "
            "summarize findings clearly, explain them in simple, non-diagnostic language, describe "
            "uncertainty and limitations, and encourage consultation with qualified specialists.\n"
        )

        lines.append(
            "Begin with a short factual summary of what the algorithm detected (1–2 sentences). "
            "Then provide a simple explanation for a non-specialist reader.\n"
        )

        lines.append(
            "Explain in 3-6 sentences what the model found, in simple, non-diagnostic language. "
            "Avoid medical advice; encourage consulting a qualified specialist for interpretation.\n"
        )

        lines.append(
            "Include a confidence score (0–100%), reasons for uncertainty, and 2–4 possible explanations "
            "(e.g., benign cyst vs neoplasm). List model limitations, and suggest non-prescriptive next steps "
            "such as 'recommend specialist review' or 'further imaging may be considered.'\n"
        )

        lines.append(
            "If any urgent or red-flag features are detected (e.g., rapid growth, compression of vital structures), "
            "clearly flag them and state that immediate clinical review is warranted.\n"
        )

        lines.append(
            "Use clear, professional tone; avoid jargon in patient-facing summaries. Never include or infer "
            "personal information. Do not instruct the reader to take specific medical actions.\n"
        )
        return "\n".join(lines)
    
    def run_complete_pipeline(self):
        """Run the complete prediction pipeline"""
        print("=" * 60)
        print("YOLO PREDICTION PIPELINE")
        print("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # Process all images
            self.process_all_images()
            
            # Save results
            self.save_results_json()
            
            # Generate report
            self.generate_summary_report()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print(f"\n" + "=" * 60)
            print(f"PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"Total Duration: {duration}")
            print(f"Results saved to: {self.output_dir}")
            print("=" * 60)
            
            # Optional: start an interactive doctor's assistant session with LLM context
            if getattr(self, 'enable_doctor_assistant', False) and _LLM_AVAILABLE:
                try:
                    self.start_doctor_assistant_chat()
                except Exception as e:
                    print(f"Doctor assistant chat could not start: {e}")

        except Exception as e:
            print(f"\nPipeline failed with error: {e}")
            raise

    def start_doctor_assistant_chat(self):
        """Start an interactive chat where a doctor can ask about results."""
        if not self.all_results:
            print("No results available for assistant context.")
            return

        # Build concise context summary for the assistant
        context_lines = [
            "Context: YOLO detections summary for current study.",
            f"Total images processed: {len(self.all_results)}",
        ]
        for r in self.all_results:
            if 'error' in r:
                context_lines.append(f"- {r.get('image_name')}: ERROR - {r.get('error')}")
                continue
            image_name = r.get('image_name')
            total = r.get('total_detections', 0)
            context_lines.append(f"- {image_name}: {total} detections")
            for d in r.get('detections', [])[:5]:  # keep concise
                cls = d.get('class_name')
                conf = d.get('confidence', 0.0)
                x1, y1, x2, y2 = d.get('bbox', [None, None, None, None])
                context_lines.append(
                    f"    • {cls} at [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] (conf {conf:.2f})"
                )
        context_lines.append(
            "\nYou are Doctor's Assistant. Answer the doctor's questions about these detections in"
            " clear, non-diagnostic language. Describe what the model found, likely anatomical"
            " region if inferable from names only, and limitations. Avoid medical advice and"
            " do not provide diagnosis. Encourage correlation with imaging and specialist review."
        )
        system_prompt = "\n".join(context_lines)

        print("\nStarting Doctor's Assistant chat. Type /exit to quit.\n")
        llm = llm_local.load_model(getattr(llm_local, 'DEFAULT_MODEL_PATH', ''))
        with llm.chat_session(system_prompt=system_prompt):
            while True:
                try:
                    user_q = input("Doctor: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting chat.")
                    break
                if not user_q:
                    continue
                if user_q.lower() in {"/exit", ":q", "quit", "exit"}:
                    print("Goodbye.")
                    break
                print("Assistant: ", end="", flush=True)
                for token in llm.generate(user_q, max_tokens=400, temp=0.2, top_p=0.9, top_k=40, repeat_penalty=1.05, streaming=True):
                    print(token, end="", flush=True)
                print()

def main():
    """Main function to run the YOLO prediction pipeline"""
    
    # Configuration
    MODEL_PATH = os.path.join("models", "Yolo_Detection_Mamografi_31.08.2025.pt")
    INPUT_DIR = "target"
    OUTPUT_DIR = "pred"
    
    # Feature flag: launch doctor's assistant chat after pipeline
    ENABLE_DOCTOR_ASSISTANT = True

    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found: {MODEL_PATH}")
        return
    
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory not found: {INPUT_DIR}")
        return
    
    try:
        # Create predictor instance
        predictor = YOLOPredictor(MODEL_PATH, INPUT_DIR, OUTPUT_DIR)
        predictor.enable_doctor_assistant = ENABLE_DOCTOR_ASSISTANT and _LLM_AVAILABLE
        
        # Run the complete pipeline
        predictor.run_complete_pipeline()
        
    except Exception as e:
        print(f"Error running pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
