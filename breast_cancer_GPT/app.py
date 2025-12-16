import os
import json
from pathlib import Path
from typing import List, Dict

import streamlit as st
from PIL import Image

# Local modules
import llm_local
from main import YOLOPredictor


# ============================================================================
# CONFIGURATION - Modify these paths to match your setup
# ============================================================================

# YOLO Model Configuration
# Option 1: Use relative path (default - looks for models/ folder in project root)
MODELS_DIR = Path("models")
DEFAULT_MODEL_NAME = "Yolo_Detection_Mamografi_v1.pt"
DEFAULT_MODEL_PATH = MODELS_DIR / DEFAULT_MODEL_NAME

# Option 2: Use absolute path (uncomment and modify if your model is elsewhere)
# DEFAULT_MODEL_PATH = Path(r"C:\path\to\your\model\Yolo_Detection_Mamografi_v1.pt")

# Or use environment variable (set YOLO_MODEL_PATH before running)
if "YOLO_MODEL_PATH" in os.environ:
	DEFAULT_MODEL_PATH = Path(os.environ["YOLO_MODEL_PATH"])

# Input/Output Directories
INPUT_DIR = Path("target")
OUTPUT_DIR = Path("pred")
RESULTS_JSON = OUTPUT_DIR / "results" / "predictions.json"
PRED_IMAGES_DIR = OUTPUT_DIR / "images"


st.set_page_config(page_title="Breast Cancer Assistant", layout="wide")


def ensure_dirs() -> None:
	OUTPUT_DIR.mkdir(exist_ok=True)
	(OUTPUT_DIR / "images").mkdir(exist_ok=True)
	(OUTPUT_DIR / "results").mkdir(exist_ok=True)
	(OUTPUT_DIR / "reports").mkdir(exist_ok=True)


def load_predictions() -> Dict:
	if RESULTS_JSON.exists():
		with open(RESULTS_JSON, "r", encoding="utf-8") as f:
			return json.load(f)
	return {}


def list_pred_images() -> List[Path]:
	if PRED_IMAGES_DIR.exists():
		return sorted([p for p in PRED_IMAGES_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
	return []


def build_context_from_results(results: Dict) -> str:
	if not results:
		return "No results available."
	lines: List[str] = [
		"Context: YOLO detections summary for current study.",
		f"Model: {results.get('model_path','')}",
		f"Total images processed: {results.get('total_images_processed', 0)}",
		"",
	]
	for r in results.get("results", [])[:30]:
		if "error" in r:
			lines.append(f"- {r.get('image_name')}: ERROR - {r.get('error')}")
			continue
		image_name = r.get("image_name")
		total = r.get("total_detections", 0)
		lines.append(f"- {image_name}: {total} detections")
		for d in r.get("detections", [])[:5]:
			cls = d.get("class_name")
			conf = d.get("confidence", 0.0)
			x1, y1, x2, y2 = d.get("bbox", [None, None, None, None])
			lines.append(
				f"    • {cls} at [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] (conf {conf:.2f})"
			)
	lines.append(
		"\nYou are Doctor's Assistant. Answer the doctor's questions about these detections in"
		" clear, diagnostic language. Describe what the model found and limitations."
		" Give medical advice or diagnoses. Because User is already a doctor."
	)
	return "\n".join(lines)


def build_explanation_prompt_from_record(rec: Dict) -> str:
	image_name = rec.get('image_name', 'image')
	img_w = rec.get('image_size', {}).get('width')
	img_h = rec.get('image_size', {}).get('height')
	lines: List[str] = [
		f"Image: {image_name} ({img_w}x{img_h})",
		f"Total detections: {rec.get('total_detections', 0)}",
		"Detections:",
	]
	for det in rec.get('detections', []):
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


@st.cache_resource(show_spinner=False)
def get_llm():
	try:
		return llm_local.load_model(getattr(llm_local, "DEFAULT_MODEL_PATH", ""))
	except Exception as e:
		st.warning(f"LLM unavailable: {e}")
		return None


def ui_sidebar():
	st.sidebar.title("Controls")

	st.sidebar.subheader("Detection")
	run_now = st.sidebar.button("Run detection now", help="Run YOLO detection on all images in target/ directory")
	
	st.sidebar.subheader("LLM Explanations")
	st.sidebar.caption("Generate AI explanations for detection results")
	generate_all_explanations = st.sidebar.button("Generate explanations for all images", help="Generate LLM explanations for all detected images")

	st.sidebar.subheader("Model")
	st.sidebar.write(f"YOLO: {DEFAULT_MODEL_PATH.name}")
	st.sidebar.caption(str(DEFAULT_MODEL_PATH))

	return run_now, generate_all_explanations


def run_detection_pipeline():
	if not DEFAULT_MODEL_PATH.exists():
		st.error(f"Model not found: {DEFAULT_MODEL_PATH}")
		return
	if not INPUT_DIR.exists():
		st.error(f"Input directory not found: {INPUT_DIR}")
		return

	# Initialize progress bar and status
	progress_bar = st.progress(0)
	status_text = st.empty()
	status_text.text("Initializing detection pipeline...")
	
	try:
		# Create predictor
		predictor = YOLOPredictor(str(DEFAULT_MODEL_PATH), str(INPUT_DIR), str(OUTPUT_DIR))
		predictor.enable_doctor_assistant = False
		
		# Get image count for progress tracking
		image_files = predictor.get_image_files()
		total_images = len(image_files)
		
		if total_images == 0:
			st.warning("No images found in target/ directory.")
			progress_bar.empty()
			status_text.empty()
			return
		
		status_text.text(f"Found {total_images} image(s). Starting detection...")
		
		# Process images with progress updates
		predictor.process_all_images(
			progress_callback=lambda p: progress_bar.progress(p),
			status_callback=lambda msg: status_text.text(msg)
		)
		
		# Save results and generate report
		status_text.text("Saving results...")
		predictor.save_results_json()
		
		status_text.text("Generating summary report...")
		predictor.generate_summary_report()
		
		# Clean up progress indicators
		progress_bar.empty()
		status_text.empty()
		
		st.success(f"Detection completed! Processed {total_images} image(s).")
		
	except Exception as e:
		progress_bar.empty()
		status_text.empty()
		st.error(f"Detection failed: {e}")
		raise


def render_results_panel(results: Dict):
	st.subheader("Results")
	if not results:
		st.info("No predictions found. Click 'Run detection now' or place results in pred/results/predictions.json")
		return

	col1, col2 = st.columns([2, 2])

	with col1:
		pred_images = list_pred_images()
		selected_image_name = None
		if not pred_images:
			st.info("No annotated images found in pred/images")
		else:
			names = [p.name for p in pred_images]
			choice = st.selectbox("Select annotated image", names, key="image_selector")
			chosen = pred_images[names.index(choice)]
			# Display image with max width constraint for better screen fit
			st.image(Image.open(chosen), caption=choice, width=600)
			# Extract original image name from annotated image name (remove "pred_" prefix, keep extension)
			# Annotated images are named like "pred_Kategori5Sag_12321_RMLO.png"
			# Original images in results are like "Kategori5Sag_12321_RMLO.png"
			if choice.startswith("pred_"):
				selected_image_name = choice[5:]  # Remove "pred_" prefix, keep extension
			else:
				selected_image_name = choice

	with col2:
		st.write("Summary")
		st.json({
			"model": results.get("model_path"),
			"total_images": results.get("total_images_processed"),
		})
		if st.checkbox("Show full JSON results"):
			st.json(results)

		# Automatically match selected image to its result entry
		rec = None
		if selected_image_name:
			# Find matching result entry by image name
			rec = next((r for r in results.get("results", []) if r.get("image_name") == selected_image_name), None)
			# If exact match not found, try partial match (in case of extension differences)
			if not rec:
				rec = next((r for r in results.get("results", []) if selected_image_name in r.get("image_name", "")), None)
		
		if rec:
			# Show image name and detection info
			st.markdown(f"**Image:** {rec.get('image_name', 'Unknown')}")
			st.markdown(f"**Detections:** {rec.get('total_detections', 0)}")
			
			# On-demand explanation button (keyed per selection for dynamic updates)
			col_a, col_b = st.columns([1, 1])
			with col_a:
				explain_now = st.button("Generate explanation for this image", key=f"explain_{rec.get('image_name', 'unknown')}")
			with col_b:
				overwrite = st.checkbox("Overwrite existing explanation", value=False)

			if explain_now:
				llm = get_llm()
				if llm is None:
					st.warning("LLM unavailable. Ensure model is accessible and gpt4all installed.")
				else:
					# If no cancer detected or effectively 0 confidence, provide generic message
					cancer_confs = [d.get("confidence", 0.0) for d in rec.get("detections", []) if str(d.get("class_name", "")).lower() == "cancer"]
					max_cancer_conf = max(cancer_confs) if cancer_confs else 0.0
					if rec.get("total_detections", 0) == 0 or max_cancer_conf < 0.01:
						answer = "No cancer findings detected by the model in this image. This is not a diagnosis; clinical correlation is recommended."
					else:
						prompt = build_explanation_prompt_from_record(rec)
						with st.spinner("Generating explanation..."):
							try:
								with llm.chat_session(system_prompt="You are a concise clinical assistant for imaging results. Avoid medical advice."):
									answer = llm.generate(
										prompt,
										max_tokens=400,
										temp=0.2,
										top_p=0.9,
										top_k=40,
										repeat_penalty=1.05,
										streaming=False,
									)
							except Exception as e:
								answer = f"Assistant unavailable: {e}"
					# Persist back to JSON
					try:
						if overwrite or not rec.get("llm_explanation"):
							rec["llm_explanation"] = answer
							# Update the results dict
							for i, r in enumerate(results.get("results", [])):
								if r.get("image_name") == rec.get("image_name"):
									results["results"][i] = rec
									break
							with open(RESULTS_JSON, "w", encoding="utf-8") as f:
								json.dump(results, f, indent=2, ensure_ascii=False)
						st.success("Explanation saved to results.")
						st.rerun()
					except Exception as e:
						st.warning(f"Could not save explanation: {e}")

			# Show explanation (new or existing) for the selected image
			st.markdown("**Assistant explanation**")
			if rec.get("llm_explanation"):
				st.info(rec.get("llm_explanation"))
			else:
				st.write("No explanation available for this image. Click 'Generate explanation' to create one.")
		else:
			if selected_image_name:
				st.info(f"No result entry found for image: {selected_image_name}")


def generate_explanations_for_all(results: Dict):
	"""Generate LLM explanations for all images that have detections but no explanation yet"""
	if not results or not results.get("results"):
		st.warning("No results available. Run detection first.")
		return
	
	llm = get_llm()
	if llm is None:
		st.error("LLM unavailable. Ensure model is accessible and gpt4all installed.")
		return
	
	# Find all images that need explanations
	images_to_process = []
	for rec in results.get("results", []):
		if rec.get("total_detections", 0) > 0 and not rec.get("llm_explanation"):
			images_to_process.append(rec)
	
	if not images_to_process:
		st.info("All images with detections already have explanations.")
		return
	
	# Process each image
	progress_bar = st.progress(0)
	status_text = st.empty()
	
	for idx, rec in enumerate(images_to_process):
		status_text.text(f"Processing {idx + 1}/{len(images_to_process)}: {rec.get('image_name', 'Unknown')}")
		progress_bar.progress((idx + 1) / len(images_to_process))
		
		try:
			# Check for cancer detections
			cancer_confs = [d.get("confidence", 0.0) for d in rec.get("detections", []) 
				if str(d.get("class_name", "")).lower() == "cancer"]
			max_cancer_conf = max(cancer_confs) if cancer_confs else 0.0
			
			if rec.get("total_detections", 0) == 0 or max_cancer_conf < 0.01:
				answer = "No cancer findings detected by the model in this image. This is not a diagnosis; clinical correlation is recommended."
			else:
				prompt = build_explanation_prompt_from_record(rec)
				with llm.chat_session(system_prompt="You are a concise clinical assistant for imaging results. Avoid medical advice."):
					answer = llm.generate(
						prompt,
						max_tokens=400,
						temp=0.2,
						top_p=0.9,
						top_k=40,
						repeat_penalty=1.05,
						streaming=False,
					)
			
			# Update the result
			rec["llm_explanation"] = answer.strip()
			
			# Update in results list
			for i, r in enumerate(results.get("results", [])):
				if r.get("image_name") == rec.get("image_name"):
					results["results"][i] = rec
					break
		except Exception as e:
			st.warning(f"Failed to generate explanation for {rec.get('image_name', 'Unknown')}: {e}")
			continue
	
	# Save updated results
	try:
		with open(RESULTS_JSON, "w", encoding="utf-8") as f:
			json.dump(results, f, indent=2, ensure_ascii=False)
		status_text.empty()
		progress_bar.empty()
		st.success(f"Generated explanations for {len(images_to_process)} image(s).")
	except Exception as e:
		st.error(f"Failed to save explanations: {e}")


def render_chat_panel(results: Dict):
	st.subheader("Doctor's Assistant Chat")

	llm = get_llm()
	if llm is None:
		st.info("Local LLM not available. Ensure gpt4all is installed and model path is correct in llm_local.py.")
		return

	if "chat_history" not in st.session_state:
		st.session_state.chat_history = []

	context = build_context_from_results(results)

	with st.expander("Context (read-only)", expanded=False):
		st.code(context)

	# Render chat history with chat bubbles
	for role, content in st.session_state.chat_history:
		with st.chat_message("user" if role == "user" else "assistant"):
			st.markdown(content)

	# Chat input at the bottom
	user_input = st.chat_input("Ask a question")
	if user_input:
		st.session_state.chat_history.append(("user", user_input.strip()))
		with st.spinner("Generating response..."):
			try:
				with llm.chat_session(system_prompt=context):
					answer = llm.generate(
						user_input.strip(),
						max_tokens=400,
						temp=0.2,
						top_p=0.9,
						top_k=40,
						repeat_penalty=1.05,
						streaming=False,
					)
			except Exception as e:
				answer = f"Assistant unavailable: {e}"
		st.session_state.chat_history.append(("assistant", answer))
		# Immediately display the last exchange
		with st.chat_message("assistant"):
			st.markdown(answer)


def main():
	ensure_dirs()

	st.title("Breast Cancer Doctor's Assistant")
	st.caption("Run detection, review annotated images and explanations, and chat with the assistant.")

	run_now, generate_all_explanations = ui_sidebar()

	if run_now:
		run_detection_pipeline()
		# Reload results after detection
		results = load_predictions()
	else:
		results = load_predictions()

	if generate_all_explanations:
		generate_explanations_for_all(results)
		# Reload results after generating explanations
		results = load_predictions()

	tab1, tab2 = st.tabs(["Results", "Chat"])
	with tab1:
		render_results_panel(results)
	with tab2:
		
		render_chat_panel(results)


if __name__ == "__main__":
	main()