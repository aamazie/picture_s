#!/usr/bin/env python3
"""Generate or modify an image with Stable Diffusion v1.5.

Examples:
    python picture_s_fixed.py --prompt "A lighthouse in a thunderstorm"
    python picture_s_fixed.py --image input.jpg --prompt "Turn this into an oil painting"

With no arguments, the program prompts interactively.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_OUTPUT = Path("pix/output.png")


def choose_device(torch_module: Any, requested: str = "auto") -> tuple[str, Any]:
    """Return a supported device and an appropriate tensor dtype."""
    if requested == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA-capable GPU is available.")
        return "cuda", torch_module.float16

    if requested == "mps":
        mps = getattr(torch_module.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested, but Apple Metal acceleration is unavailable.")
        return "mps", torch_module.float16

    if requested == "cpu":
        return "cpu", torch_module.float32

    if torch_module.cuda.is_available():
        return "cuda", torch_module.float16

    mps = getattr(torch_module.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps", torch_module.float16

    return "cpu", torch_module.float32


def load_input_image(image_path: str | Path, max_side: int = 768) -> Image.Image:
    """Open an image safely and resize it to Stable Diffusion-friendly dimensions."""
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Input image was not found: {path}")

    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"The input is not a readable image: {path}") from exc

    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    width = max(8, int(width * scale) // 8 * 8)
    height = max(8, int(height * scale) // 8 * 8)

    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)

    return image


def build_pipeline(mode: str, device: str, dtype: Any):
    """Load exactly one pipeline, avoiding the original script's duplicate model load."""
    try:
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
    except ImportError as exc:
        raise RuntimeError(
            "Diffusers is not installed. Install the packages listed in requirements.txt."
        ) from exc

    pipeline_class = (
        AutoPipelineForImage2Image if mode == "img2img" else AutoPipelineForText2Image
    )

    load_options: dict[str, Any] = {
        "torch_dtype": dtype,
        "use_safetensors": True,
    }
    if device in {"cuda", "mps"}:
        load_options["variant"] = "fp16"

    pipeline = pipeline_class.from_pretrained(MODEL_ID, **load_options)
    pipeline = pipeline.to(device)

    # Lowers peak memory use, especially on smaller GPUs and CPU systems.
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()

    return pipeline


def generate_image(
    prompt: str,
    image_path: str | None = None,
    output_path: str | Path = DEFAULT_OUTPUT,
    device_request: str = "auto",
    strength: float = 0.75,
    steps: int = 30,
    seed: int | None = None,
) -> Path:
    """Run text-to-image or image-to-image generation and save the result."""
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("The prompt cannot be empty.")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("Strength must be between 0.0 and 1.0.")
    if steps < 1:
        raise ValueError("Inference steps must be at least 1.")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed.") from exc

    device, dtype = choose_device(torch, device_request)
    mode = "img2img" if image_path else "text2img"
    pipeline = build_pipeline(mode, device, dtype)

    call_options: dict[str, Any] = {
        "prompt": prompt,
        "num_inference_steps": steps,
    }

    if seed is not None:
        generator_device = "cuda" if device == "cuda" else "cpu"
        call_options["generator"] = torch.Generator(device=generator_device).manual_seed(seed)

    if image_path:
        call_options["image"] = load_input_image(image_path)
        call_options["strength"] = strength
    else:
        call_options["height"] = 512
        call_options["width"] = 512

    with torch.inference_mode():
        result = pipeline(**call_options)

    if not getattr(result, "images", None):
        raise RuntimeError("The model returned no image.")

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(output)
    return output.resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or modify an image with Stable Diffusion v1.5."
    )
    parser.add_argument("--prompt", help="Text prompt for the image")
    parser.add_argument("--image", help="Optional input image for image-to-image mode")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output image path")
    parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto"
    )
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    image_path = args.image
    if image_path is None and args.prompt is None:
        image_path = input(
            "Enter an image path to modify, or press Enter for a new image: "
        ).strip() or None

    prompt = args.prompt
    if prompt is None:
        prompt = input("Enter your prompt: ").strip()

    try:
        output = generate_image(
            prompt=prompt,
            image_path=image_path,
            output_path=args.output,
            device_request=args.device,
            strength=args.strength,
            steps=args.steps,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Image saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
