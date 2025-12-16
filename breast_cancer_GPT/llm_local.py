import argparse
import os
import sys
from typing import Optional


def ensure_gpt4all_installed() -> None:
	try:
		import gpt4all  # noqa: F401
	except Exception:  # pragma: no cover
		print(
			"The 'gpt4all' package is required. Install with:"
			"\n  pip install gpt4all",
			file=sys.stderr,
		)
		sys.exit(1)


def resolve_model_parts(full_path: str) -> tuple[str, str]:
	full_path = os.path.expandvars(os.path.expanduser(full_path))
	directory = os.path.dirname(full_path)
	filename = os.path.basename(full_path)
	if not filename:
		raise ValueError("Model filename is empty; provide a full path to the .gguf file.")
	return directory, filename


_LLM_CACHE: dict[str, object] = {}


def load_model(model_full_path: str):
	ensure_gpt4all_installed()
	from gpt4all import GPT4All

	model_dir, model_name = resolve_model_parts(model_full_path)
	key = os.path.join(model_dir, model_name)

	# Return cached, opened instance if available
	cached = _LLM_CACHE.get(key)
	if cached is not None:
		return cached

	# Try GPU first for speed! Fall back to CPU only if GPU fails
	# Priority: 1) User-specified device, 2) "gpu" (CUDA), 3) "kompute" (Vulkan), 4) "cpu"
	user_device = os.environ.get("GPT4ALL_DEVICE", None)
	
	# Device priority list - try GPU options first
	if user_device:
		device_priority = [user_device, "gpu", "kompute", "cpu"]
	else:
		device_priority = ["gpu", "kompute", "cpu"]

	def _create_with_device(device_name: str):
		# GPT4All loads the model on instantiation, no need for explicit open()
		# If backend initialization fails (GGML_ASSERT), this may raise or crash
		try:
			return GPT4All(model_name=model_name, model_path=model_dir, device=device_name)
		except Exception as e:
			raise RuntimeError(f"Failed to initialize GPT4All with device '{device_name}': {e}") from e

	# Try devices in priority order
	last_error = None
	for device_name in device_priority:
		try:
			print(f"Attempting to load model on device: {device_name}", file=sys.stderr)
			llm = _create_with_device(device_name)
			# Warm-up tiny generate to ensure fully loaded
			# This may trigger backend initialization that could fail with GGML_ASSERT
			try:
				_ = llm.generate(".", max_tokens=1, temp=0.0, streaming=False)
				print(f"Successfully loaded model on device: {device_name}", file=sys.stderr)
				_LLM_CACHE[key] = llm
				return llm
			except Exception as warmup_err:
				# If warm-up fails, the backend might not be ready - try next device
				print(f"Warning: Warm-up failed on {device_name}: {warmup_err}. Trying next device...", file=sys.stderr)
				last_error = warmup_err
				continue
		except Exception as e:
			print(f"Failed to load on {device_name}: {e}. Trying next device...", file=sys.stderr)
			last_error = e
			continue
	
	# If we get here, all devices failed
	raise RuntimeError(
		f"Failed to load model on all attempted devices: {device_priority}. "
		f"Last error: {last_error}. "
		"Check GPU drivers, CUDA/Vulkan installation, or model file path."
	) from last_error


def generate(
	prompt: str,
	model_full_path: str,
	max_tokens: int = 512,
	temperature: float = 0.2,
	top_p: float = 0.9,
	top_k: int = 40,
	repeat_penalty: float = 1.1,
	system_prompt: Optional[str] = None,
	stream: bool = True,
) -> str:
	llm = load_model(model_full_path)

	# Use a chat session for better instruction following
	output_chunks: list[str] = []
	with llm.chat_session(system_prompt=system_prompt or "You are a helpful assistant."):
		if stream:
			for token in llm.generate(
				prompt,
				max_tokens=max_tokens,
				temp=temperature,
				top_p=top_p,
				top_k=top_k,
				repeat_penalty=repeat_penalty,
				streaming=True,
			):
				print(token, end="", flush=True)
				output_chunks.append(token)
			print()  # newline after stream
			return "".join(output_chunks)
		else:
			text = llm.generate(
				prompt,
				max_tokens=max_tokens,
				temp=temperature,
				top_p=top_p,
				top_k=top_k,
				repeat_penalty=repeat_penalty,
				streaming=False,
			)
			print(text)
			return text


def interactive_chat(model_full_path: str, system_prompt: Optional[str] = None) -> None:
	llm = load_model(model_full_path)
	print("Interactive chat started. Type /exit to quit, /clear to reset context.\n")
	with llm.chat_session(system_prompt=system_prompt or "You are a helpful assistant."):
		while True:
			try:
				user = input("You: ").strip()
			except (EOFError, KeyboardInterrupt):
				print("\nExiting.")
				break

			if user.lower() in {"/exit", ":q", "quit", "exit"}:
				print("Goodbye!")
				break
			if user.lower() in {"/clear", "/reset"}:
				llm.reset_chat()
				print("Context cleared.")
				continue
			if not user:
				continue

			print("Assistant: ", end="", flush=True)
			for token in llm.generate(user, max_tokens=512, temp=0.2, top_p=0.9, top_k=40, repeat_penalty=1.1, streaming=True):
				print(token, end="", flush=True)
			print()


# ============================================================================
# LLM MODEL CONFIGURATION - Modify this path to match your LLM model location
# ============================================================================

# Option 1: Use default GPT4All location (Windows)
# Uses %LOCALAPPDATA% environment variable to avoid hardcoding username
_default_gpt4all_path = os.path.join(
	os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
	"nomic.ai", "GPT4All", "qwen2.5-coder-7b-instruct-q4_0.gguf"
)
DEFAULT_MODEL_PATH = _default_gpt4all_path

# Option 2: Use custom path (uncomment and modify)
# DEFAULT_MODEL_PATH = r"C:\path\to\your\llm\model.gguf"

# Option 3: Use environment variable (set LLM_MODEL_PATH before running)
if "LLM_MODEL_PATH" in os.environ:
	DEFAULT_MODEL_PATH = os.environ["LLM_MODEL_PATH"]


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Run a local GGUF model via GPT4All (one-shot or chat).",
	)
	parser.add_argument(
		"--model",
		default=DEFAULT_MODEL_PATH,
		help="Full path to the .gguf model file. Defaults to your provided local model.",
	)
	parser.add_argument(
		"--prompt",
		help="One-shot prompt to generate a completion. If omitted, starts interactive chat.",
	)
	parser.add_argument("--system", help="Optional system prompt for better steering.")
	parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate.")
	parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
	parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling.")
	parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling.")
	parser.add_argument(
		"--repeat-penalty",
		type=float,
		default=1.1,
		help="Penalty for repeated tokens (higher = less repetition).",
	)
	parser.add_argument(
		"--no-stream",
		action="store_true",
		help="Disable token streaming in one-shot mode.",
	)
	return parser


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()

	model_path = args.model
	if args.prompt:
		generate(
			prompt=args.prompt,
			model_full_path=model_path,
			max_tokens=args.max_tokens,
			temperature=args.temperature,
			top_p=args.top_p,
			top_k=args.top_k,
			repeat_penalty=args.repeat_penalty,
			system_prompt=args.system,
			stream=not args.no_stream,
		)
	else:
		interactive_chat(model_full_path=model_path, system_prompt=args.system)


if __name__ == "__main__":
	main()
