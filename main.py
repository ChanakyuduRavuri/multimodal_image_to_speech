"""
Multimodal Image-to-Speech Pipeline
====================================
Input Image → Object Detection → Label Extraction & Counting → Text Generation → TTS → Audio Output

Models used:
- Object Detection: facebook/detr-resnet-50
- Text-to-Speech:   suno/bark-small
"""

import sys
import os
import gc
from collections import Counter

import torch
import numpy as np
from PIL import Image
from transformers import pipeline
from scipy.io.wavfile import write as write_wav


def detect_objects(image_path):
    """Step 1: Detect objects in the image using DETR."""
    print(f"\n[Step 1] Running object detection on: {image_path}")

    image = Image.open(image_path)

    detector = pipeline(
        task="object-detection",
        model="facebook/detr-resnet-50",
        torch_dtype=torch.bfloat16,
    )

    results = detector(image)

    # Clean up to free memory
    del detector
    gc.collect()

    print(f"  Detected {len(results)} object(s):")
    for det in results:
        print(f"    - {det['label']} ({det['score']:.2f})")

    return results


def generate_text(detections):
    """Step 2: Count detected objects and generate a descriptive sentence."""
    print("\n[Step 2] Generating descriptive text...")

    # Count occurrences of each label
    label_counts = Counter(det["label"] for det in detections)

    if not label_counts:
        text = "No objects were detected in the image."
    else:
        # Build parts like "2 cars", "1 person"
        parts = []
        for label, count in label_counts.items():
            if count == 1:
                parts.append(f"1 {label}")
            else:
                parts.append(f"{count} {label}s")

        # Join with commas and "and"
        if len(parts) == 1:
            items_text = parts[0]
        elif len(parts) == 2:
            items_text = f"{parts[0]} and {parts[1]}"
        else:
            items_text = ", ".join(parts[:-1]) + f", and {parts[-1]}"

        text = f"The image contains {items_text}."

    print(f"  Generated text: {text}")
    return text


def text_to_speech(text, output_path):
    """Step 3: Convert text to speech and save as WAV file."""
    print(f"\n[Step 3] Converting text to speech...")

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


def main():
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = os.path.join("images", "image_1.jpg")

    if not os.path.exists(image_path):
        print(f"Error: Image not found at '{image_path}'")
        sys.exit(1)

    output_path = os.path.join("output", "output.wav")

    print("=" * 50)
    print("  Multimodal Image-to-Speech Pipeline")
    print("=" * 50)

    # Step 1: Object Detection
    detections = detect_objects(image_path)

    # Step 2: Label Extraction & Text Generation
    text = generate_text(detections)

    # Step 3: Text-to-Speech
    text_to_speech(text, output_path)

    print("\n" + "=" * 50)
    print("  Pipeline complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
