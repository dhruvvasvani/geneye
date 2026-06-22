👁️ GenEye — Synthetic Ocular Biometric Generator
Simulate realistic eye diseases from any face photo. Built for medical AI research, training datasets, and ophthalmology education — without touching a single real patient record.

Python FastAPI OpenCV License

What is this?
Medical AI needs thousands of labelled eye images to train on. The problem? Real patient scans are private, rare, and legally restricted. You can't just grab them off the internet.

GenEye solves that.

You upload or capture a face photo → it finds your eye → simulates a disease on it → returns a synthetic medical image. The output looks clinically plausible but belongs to no real person and carries zero privacy risk.

It's not a filter. It's not a cartoon effect. It uses iris boundary detection (HoughCircles) to locate the anatomically correct regions — cornea, lens, sclera, limbus — and applies disease-specific optical transformations calibrated to how these conditions actually present in clinical photography.

Supported Conditions
Anomaly	What it simulates
Corneal Opacity	Stromal/nuclear scarring — white-grey haze with limbal sparing and Haab's striae texture
Cataract	Nuclear sclerotic opacity — amber-to-brunescent core, cortical spoke opacities, PSC highlight
Glaucoma	Elevated IOP signs — steel-blue corneal oedema, peripheral field darkening, Descemet striae
Retinal Detachment	Subconjunctival haemorrhage — scleral injection, blood patch noise, conjunctival chemosis, radial fold distortion
Each condition has a severity slider (0–100%) so you can generate Grade I → Grade IV progressions.

How it works
Face photo (upload or webcam)
         │
         ▼
  Face detection (Haar cascade)
         │
         ▼
  Eye region extracted
         │
         ▼
  Iris + pupil boundary detected (HoughCircles)
         │
         ▼
  Anomaly applied to exact anatomical region
  (cornea / lens / sclera — not the whole image)
         │
         ▼
  Synthetic eye image returned as JPEG
No AI model downloads. No GPU required. Runs entirely on CPU. Each generation takes under 200 ms.

Tech stack
Backend — FastAPI + Uvicorn
Computer vision — OpenCV (iris/pupil detection, image processing)
Image synthesis — NumPy + Pillow (anatomically-grounded rendering)
Frontend — Vanilla HTML/CSS/JS (dark UI, webcam support, drag & drop)
Getting Started
Prerequisites
Python 3.9 or higher
pip
A webcam (optional — you can also upload images)
Setup in VS Code
Step 1 — Clone the repo

git clone https://github.com/YOUR_USERNAME/geneye.git
cd geneye
Step 2 — Create a virtual environment

# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
Step 3 — Install dependencies

pip install -r requirements.txt
Step 4 — Run the server

python main.py
Step 5 — Open in browser

http://127.0.0.1:8000
That's it. No model downloads, no API keys, no configuration files.

VS Code recommended extensions
Install these for the best experience:

Extension	ID
Python	ms-python.python
Pylance	ms-python.vscode-pylance
REST Client (optional, for API testing)	humao.rest-client
Or install all at once from the terminal:

code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
API
The app exposes a simple REST endpoint if you want to integrate it into your own pipeline.

POST /api/v1/generate-from-face
Form fields:

Field	Type	Description
face_image	File	JPG or PNG — any face photo
anomaly	string	none / corneal_opacity / cataract / glaucoma / retinal_detachment
severity	float	0.0 (minimal) → 1.0 (severe)
Response: JPEG image — the synthesised eye with the requested anomaly.

Example with curl:

curl -X POST http://127.0.0.1:8000/api/v1/generate-from-face \
  -F "face_image=@photo.jpg" \
  -F "anomaly=cataract" \
  -F "severity=0.6" \
  --output result.jpg
Example with Python:

import requests

with open("photo.jpg", "rb") as f:
    resp = requests.post(
        "http://127.0.0.1:8000/api/v1/generate-from-face",
        files={"face_image": f},
        data={"anomaly": "glaucoma", "severity": 0.75},
    )

with open("result.jpg", "wb") as out:
    out.write(resp.content)
GET /health
Returns {"status": "ok"} — useful for uptime checks.

Project structure
geneye/
├── main.py              # FastAPI app + API endpoints
├── generator.py         # Anomaly simulation engine (iris detection + rendering)
├── face_extractor.py    # Face/eye region extraction (Haar cascade)
├── validation_engine.py # Iris structure validator (blur + circle check)
├── requirements.txt     # Python dependencies
├── static/
│   └── index.html       # Frontend UI (no framework, pure HTML/CSS/JS)
└── .gitignore
Privacy & ethics
No real patient data is used or stored. The face photo you upload is processed in memory and immediately discarded — it is never saved to disk.
The output is fully synthetic. It doesn't represent any real person's medical condition.
Intended for research and education only. Not a medical diagnostic tool. Do not use outputs for clinical decisions.
Why no AI model?
Earlier versions used Stable Diffusion (instruct-pix2pix) for generation. That required a 6+ GB download and took 10–20 minutes per image on CPU.

The current approach — anatomically-grounded OpenCV rendering — produces medically plausible results in under 200 ms with no downloads. The key insight is that realism comes from applying effects to the right anatomical region (detected via HoughCircles), not from running a billion-parameter model.

Contributing
Pull requests are welcome. If you add a new anomaly, please include a brief comment citing the clinical reference you used for the colour/texture calibration.

