# Face Recognition with OpenCV DNN

Real-time face recognition that runs in your **browser**. Detects faces from your webcam, identifies them using MobileFaceNet embeddings, and displays names — no OpenCV GUI needed.

## How It Works

```
Webcam frame  →  Detect face (Caffe SSD)  →  Extract embedding (MobileFaceNet ONNX)  →  Match against known people  →  Show name in browser
```

The script starts an MJPEG HTTP server on `http://localhost:8080`. Your browser opens automatically and shows the live annotated feed.

## Requirements

- **Python 3.6+**
- **Webcam** (built-in or USB)
- **~200 MB free disk space** (for model files)

## Setup

### 1. Clone and enter the project

```bash
git clone <your-repo-url> face_recognition
cd face_recognition
```

### 2. Create virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 3. Download the face detection model

```bash
wget -O models/deploy.prototxt \
  https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
wget -O models/res10_300x300_ssd_iter_140000.caffemodel \
  https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
```

### 4. Download the recognition model (MobileFaceNet)

```bash
curl -L -o /tmp/buffalo_sc.zip \
  https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip
unzip -j /tmp/buffalo_sc.zip w600k_mbf.onnx -d models/
rm /tmp/buffalo_sc.zip
```

### 5. Add your face

Place a photo of yourself (well-lit, facing forward) in `known_faces/`:

```bash
cp ~/Desktop/your_photo.jpg known_faces/your_name.jpg
```

**Tip:** Add multiple photos (different angles, expressions) for better reliability:

```bash
mkdir known_faces/your_name
cp photo1.jpg known_faces/your_name/
cp photo2.jpg known_faces/your_name/
```

## Run

```bash
source venv/bin/activate      # if not already activated
python face_recognizer.py
```

Your browser should open to `http://localhost:8080` with the live webcam feed. Press **Ctrl+C** in the terminal to quit.

## CLI Options

| Flag | Description |
|------|-------------|
| `--match` | Capture one frame and print top-5 match scores (no live feed) |
| `--threshold 0.25` | Set recognition threshold (default: 0.35) |

Examples:

```bash
python face_recognizer.py --match
python face_recognizer.py --threshold 0.30
```

## Adding More People

Place their photos in `known_faces/` and re-run the script:

```
known_faces/
├── alice.jpg
├── bob.jpg
├── Charlie/
│   ├── charlie_smiling.jpg
│   └── charlie_serious.jpg
└── .gitkeep
```

Subdirectories let you add multiple photos per person. The cache (`_embeddings_cache.npz`) auto-updates — delete it only if you want a fresh scan.

## Understanding the Labels

| Label | Color | Meaning |
|-------|-------|---------|
| `NAME: Alice (72.3%)` | Green | Match above threshold |
| `GUESS: Alice? (25.0%)` | Yellow | Weak match below threshold |
| `NO MATCH (5.0%)` | Red | No one recognized |

## Threshold Guide

| Value | Behavior |
|-------|----------|
| 0.60+ | Only very confident matches |
| **0.35** | **Default — good balance** |
| 0.20 | More matches, more wrong guesses |
| 0.10 | Everyone gets a name (mostly wrong) |

## Project Structure

```
├── face_recognizer.py          # Main script (detection + recognition)
├── face_detector.py            # Detection only (redirects to recognizer)
├── download_lfw.py             # (Optional) LFW dataset downloader
├── requirements.txt            # Python dependencies
├── .gitignore
├── README.md
├── known_faces/                # Put your face photos here
│   └── .gitkeep
└── models/                     # Download models into here
    ├── deploy.prototxt
    ├── res10_300x300_ssd_iter_140000.caffemodel
    └── w600k_mbf.onnx
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No module named cv2` | Run `pip install -r requirements.txt` |
| `Could not open webcam` | Check webcam connection, try `ls /dev/video*` |
| `localhost:8080` not loading | Restart the script, check port with `lsof -i :8080` |
| Low match scores | Add more photos of yourself with different expressions |
| Wrong person matched | Lower threshold with `--threshold 0.25` |

## Models

| Model | Type | Input | Output | Size |
|-------|------|-------|--------|------|
| Detection | Caffe SSD (ResNet-10) | 300×300 | Bounding boxes | 11 MB |
| Recognition | MobileFaceNet ONNX | 112×112 | 512-d embedding | 13.6 MB |

- Detection: trained on WIDER FACE
- Recognition: trained on WebFace (InsightFace zoo)
