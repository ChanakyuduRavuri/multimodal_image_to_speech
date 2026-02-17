"""
Multimodal Image-to-Speech Pipeline — Approach 2
==================================================
Input Image → Image Captioning (Google GenAI) → Text Processing → TTS → Audio Output

Models used:
- Image Captioning: gemini-3-flash-preview (Google GenAI)
- Text-to-Speech:   suno/bark-small (HuggingFace Transformers)
"""

import sys
import os
import gc
import re

import torch
import numpy as np
from PIL import Image
from transformers import pipeline
from scipy.io.wavfile import write as write_wav

from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful AI Assistant. Given an image perform object detection "
    "and provide a text output which contains the information about the labels "
    "detected and their counts."
)

MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def _resolve_mime(image_path: str) -> str:
    """Return MIME type based on file extension."""
    ext = os.path.splitext(image_path)[1].lower()
    mime = MIME_MAP.get(ext)
    if mime is None:
        raise ValueError(f"Unsupported image format: {ext}")
    return mime


# ---------------------------------------------------------------------------
# Step 0: Load API key
# ---------------------------------------------------------------------------

def load_api_key(key_path: str = os.path.join("keys", ".gemini.txt")) -> str:
    """Read the Gemini API key from a local file."""
    if not os.path.exists(key_path):
        raise FileNotFoundError(
            f"API key file not found at '{key_path}'. "
            "Create the file and paste your Gemini API key inside."
        )

    with open(key_path, "r") as f:
        key = f.read().strip()

    if not key or key == "ENTER_YOUR_GEMINI_API_KEY_HERE":
        raise ValueError(
            f"Please replace the placeholder in '{key_path}' with your actual Gemini API key."
        )

    return key


# ---------------------------------------------------------------------------
# Step 1: Image Captioning using Google GenAI
# ---------------------------------------------------------------------------

def caption_image(image_path: str, client: genai.Client) -> str:
    """Generate an accessibility-focused caption for the image using Gemini."""
    print(f"\n[Step 1] Captioning image with Gemini: {image_path}")

    mime_type = _resolve_mime(image_path)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
    ]

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
        ),
    )

    caption = response.text
    print(f"  Gemini response:\n    {caption}")
    return caption


# ---------------------------------------------------------------------------
# Step 2: Text Processing (optional enhancement)
# ---------------------------------------------------------------------------

def process_text(raw_caption: str) -> str:
    """Clean up and enhance the generated caption for TTS."""
    print("\n[Step 2] Processing text for speech synthesis...")

    # Remove markdown-style formatting symbols
    text = re.sub(r"[*_#`>]", "", raw_caption)

    # Collapse multiple whitespace / newlines into a single space
    text = re.sub(r"\s+", " ", text).strip()

    # Add an introductory phrase if not already present
    if not text.lower().startswith("here is"):
        text = "Here is what I see in the image: " + text

    # Truncate to ~500 chars to keep TTS output manageable
    if len(text) > 500:
        text = text[:497] + "..."

    print(f"  Processed text: {text}")
    return text


# ---------------------------------------------------------------------------
# Step 3: Text-to-Speech
# ---------------------------------------------------------------------------

def text_to_speech(text: str, output_path: str) -> str:
    """Convert text to speech using suno/bark-small and save as WAV."""
    print("\n[Step 3] Converting text to speech...")

    synthesizer = pipeline(
        task="text-to-speech",
        model="suno/bark-small",
    )

    result = synthesizer(text)

    # Extract audio data and sampling rate
    audio = np.array(result["audio"][0])
    sampling_rate = result["sampling_rate"]

    # Normalize audio to 16-bit PCM range
    audio = audio / np.max(np.abs(audio))
    audio_16bit = (audio * 32767).astype(np.int16)

    # Save as WAV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_wav(output_path, sampling_rate, audio_16bit)

    print(f"  Audio saved to: {output_path}")
    print(f"  Sampling rate: {sampling_rate} Hz")

    # Clean up
    del synthesizer
    gc.collect()

    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = os.path.join("images", "image_1.jpg")

    if not os.path.exists(image_path):
        print(f"Error: Image not found at '{image_path}'")
        sys.exit(1)

    output_path = os.path.join("output", "output_v2.wav")

    print("=" * 60)
    print("  Multimodal Image-to-Speech Pipeline — Approach 2")
    print("  (Google GenAI + HuggingFace Transformers TTS)")
    print("=" * 60)

    # Step 0: Load API key & create Gemini client
    api_key = load_api_key()
    client = genai.Client(api_key=api_key)

    # Step 1: Image Captioning (Gemini)
    raw_caption = caption_image(image_path, client)

    # Step 2: Text Processing
    text = process_text(raw_caption)

    # Step 3: Text-to-Speech
    text_to_speech(text, output_path)

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
