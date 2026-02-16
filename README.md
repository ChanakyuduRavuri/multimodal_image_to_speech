# Multimodal Image-to-Speech Pipeline

A simple multimodal AI system that detects objects in an image, generates a descriptive sentence, and converts it to speech audio.

## System Workflow

```
Input Image → Object Detection → Label Extraction & Counting → Text Generation → Text-to-Speech → Audio Output
```

## Models Used

| Task | Model |
|------|-------|
| Object Detection | `facebook/detr-resnet-50` |
| Text-to-Speech | `suno/bark-small` |

Text generation uses pure Python logic (label counting + sentence construction).

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Python Script

```bash
python main.py                      # uses default image (images/image_1.jpg)
python main.py images/image_2.jpeg  # specify a custom image
```

Output audio is saved to `output/output.wav`.

### Jupyter Notebook

Open `multimodal_pipeline.ipynb` and run all cells. Audio plays inline at the end.

## Project Structure

```
multimodal_image_to_speech/
├── main.py                    # CLI script
├── multimodal_pipeline.ipynb  # Interactive notebook
├── requirements.txt           # Dependencies
├── README.md
├── .gitignore
├── images/                    # Input images
│   ├── image_1.jpg
│   └── image_2.jpeg
└── output/                    # Generated audio (gitignored)
    └── output.wav
```

## Example Output

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

## Requirements

- Python 3.8+
- ~2GB disk space for model downloads
- GPU recommended but not required
