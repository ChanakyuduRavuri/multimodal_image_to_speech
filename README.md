# Multimodal Image-to-Speech Pipeline

A multimodal AI system that analyzes an image and converts the analysis into speech audio. Two approaches are provided.

---

## Approach 1 — DETR Object Detection + Python Text Generation + HF TTS

### System Workflow

```
Input Image → Object Detection (DETR) → Label Extraction & Counting → Text Generation → TTS → Audio Output
```

### Models Used

| Task | Model |
|------|-------|
| Object Detection | `facebook/detr-resnet-50` |
| Text-to-Speech | `suno/bark-small` |

Text generation uses pure Python logic (label counting + sentence construction).

### Usage

```bash
python main.py                      # uses default image (images/image_1.jpg)
python main.py images/image_2.jpeg  # specify a custom image
```

Output audio is saved to `output/output.wav`.

---

## Approach 2 — Google GenAI (Gemini) + HuggingFace Transformers TTS

### System Workflow

```
Input Image → Image Captioning (Gemini) → Text Processing → TTS → Audio Output
```

Instead of detecting isolated objects, the Gemini vision model:
- Understands the entire scene
- Identifies relationships between objects
- Describes actions and context
- Produces natural, human-like language

### Models Used

| Task | Model | Provider |
|------|-------|----------|
| Image Captioning | `gemini-3-flash-preview` | Google GenAI |
| Text-to-Speech | `suno/bark-small` | HuggingFace Transformers |

### API Key Setup

Place your Gemini API key in `keys/.gemini.txt`:

```bash
echo "YOUR_API_KEY" > keys/.gemini.txt
```

### Usage

```bash
python main_v2.py                      # uses default image (images/image_1.jpg)
python main_v2.py images/image_2.jpeg  # specify a custom image
```

Output audio is saved to `output/output_v2.wav`.

---

## Setup

```bash
pip install -r requirements.txt
```

## Jupyter Notebooks

| Notebook | Approach |
|----------|----------|
| `multimodal_pipeline.ipynb` | Approach 1 — DETR + Python text + Bark TTS |
| `multimodal_pipeline_v2.ipynb` | Approach 2 — Gemini captioning + Bark TTS |

Open the desired notebook and run all cells. Audio plays inline at the end.

## Project Structure

```
multimodal_image_to_speech/
├── main.py                        # CLI script — Approach 1
├── main_v2.py                     # CLI script — Approach 2
├── multimodal_pipeline.ipynb      # Interactive notebook — Approach 1
├── multimodal_pipeline_v2.ipynb   # Interactive notebook — Approach 2
├── requirements.txt               # Dependencies
├── README.md
├── .gitignore
├── images/                        # Input images
│   ├── image_1.jpg
│   └── image_2.jpeg
├── keys/                          # API keys (gitignored)
│   └── .gemini.txt
└── output/                        # Generated audio (gitignored)
    ├── output.wav
    └── output_v2.wav
```

## Example Output

### Approach 1

```
[Step 1] Running object detection on: images/image_1.jpg
  Detected 5 object(s):
    - car (0.98)
    - car (0.95)
    - person (0.92)

[Step 2] Generating descriptive text...
  Generated text: The image contains 2 cars and 1 person.

[Step 3] Converting text to speech...
  Audio saved to: output/output.wav
```

### Approach 2

```
[Step 1] Captioning image with Gemini: images/image_1.jpg
  Gemini response:
    The image contains 2 cars, 1 person, and 1 traffic light.

[Step 2] Processing text for speech synthesis...
  Processed text: Here is what I see in the image: The image contains 2 cars, 1 person, and 1 traffic light.

[Step 3] Converting text to speech...
  Audio saved to: output/output_v2.wav
```

## Requirements

- Python 3.8+
- ~2GB disk space for model downloads
- GPU recommended but not required
- Gemini API key (for Approach 2 only)
