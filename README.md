# AR Math Equation Solver

An intelligent, camera-enabled or marker-based Augmented Reality (AR) application built using **FastAPI**, **AR.js**, and **Pix2Tex OCR**, designed to detect, interpret, and solve mathematical equations in real time from either uploaded images or through the webcam using printed AR markers.

---

## Features

- Upload-based or camera-based equation solving
- Pix2Tex OCR for high-accuracy LaTeX extraction
- Local LLM powered by `wizard-math:13b` via [Ollama](https://ollama.com/) for step-by-step math solutions
- Supports both raw LaTeX and corrected equations
- AR visualization using [AR.js](https://ar-js-org.github.io/AR.js-Docs/) to project equations in the real world
- Live solution streaming via FastAPI using `StreamingResponse`
- Clean UI with Bootstrap 5 and MathJax rendering
- Displays both uploaded and preprocessed (denoised) images
- Supports both marker-based and markerless AR modes

---

## Project Structure

```
app/
├── main.py                  # FastAPI entrypoint
├── routes/
│   └── equation.py          # API logic for OCR, solving, streaming
├── services/
│   ├── image_preprocessing.py
│   └── ocr_engine.py        # Pix2Tex or Tesseract-based OCR
├── static/
│   ├── css/style.css
│   ├── js/
│   │   ├── api-client.js
│   │   └── ar-render.js
│   └── markers/
│       └── diff-triggerr.patt
├── templates/
│   ├── index.html           # Upload mode
│   ├── ar-marker.html       # Marker-based AR
│   └── ar-markerless.html   # Markerless AR
requirements.txt
run.py
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ar-math-solver.git
cd ar-math-solver
```

### 2. Set up virtual environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

Ensure Python 3.9+ is installed.

```bash
pip install -r requirements.txt
```

### 4. Start your local LLM (Ollama)

Make sure the `wizard-math:13b` model is available.

```bash
ollama run wizard-math:13b
```

If it's not installed yet:

```bash
ollama pull wizard-math:13b
ollama run wizard-math:13b
```

### 5. Launch the application

```bash
python run.py
```

Visit in browser: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## Usage Modes

| Mode        | URL              | Description                               |
|-------------|------------------|-------------------------------------------|
| Upload Only | `/`              | Upload an image and receive live solution |
| Marker AR   | `/ar-marker`     | Show printed AR marker to webcam          |
| Markerless  | `/ar-markerless` | Tap camera view to scan and solve         |

---

## Sample Marker

Use this marker image to trigger marker-based AR mode:

```
/static/markers/diff-triggerr.patt
```

Print it or display it clearly on screen for detection.

---

## Tech Stack

- FastAPI + Uvicorn
- Pix2Tex, Tesseract, SymPy, OpenCV
- AR.js, A-Frame, Bootstrap 5
- MathJax, Ollama, wizard-math:13b

---

## Future Improvements

- Add mobile-optimized markerless AR support
- Camera pause/play control for better scanning
- Export extracted LaTeX and answers as PDF

---

## Contributors

- **Vamshi Krishna Madhavan** – Project Creator  
- **ChatGPT** – Assistant for development, optimization, and debugging

---

## Sample AR Math Solving Equation:

## DEMO:

[![Watch the Demo](https://is1-ssl.mzstatic.com/image/thumb/Purple112/v4/b1/fc/3d/b1fc3d96-a69d-acba-2ec2-e3460f185368/AppIcon-1x_U007emarketing-0-0-0-7-0-0-85-220.png/1200x630wa.png)]([https://drive.google.com/uc?id=1-GuAkBnskkBW-FC4DKYmFsbOpTIUVpdl](https://drive.google.com/file/d/1-GuAkBnskkBW-FC4DKYmFsbOpTIUVpdl/view))

---

## License

Copyright © 2025 Vamshi Krishna Madhavan

All rights reserved.

This software and all associated files are proprietary and confidential. No part of this project may be copied, reproduced, distributed, modified, or transmitted in any form or by any means — electronic, mechanical, photocopying, recording, or otherwise — without the prior written permission of the author.

Unauthorized use, publication, or redistribution is strictly prohibited. Viewing the code in a public repository does not constitute a grant of license or permission to use the code in any manner.

Violation of these terms may result in legal action.
