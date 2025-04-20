import io
import json

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from meta_ai_api import MetaAI

from app.services.image_preprocessing import preprocess_image
from app.services.ocr_engine import extract_text, extract_ai_text, is_gibberish, correct_latex_ocr

router = APIRouter(prefix="/api/equation", tags=["Equation"])


def process_image(image_bytes: bytes):
    raw_img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    raw_np = np.array(raw_img)

    print("Trying OCR on original image...")
    raw_text = extract_text(image_bytes, raw_np, source="raw")

    if is_gibberish(raw_text):
        print("OCR not good. Preprocessing...")
        preprocessed = preprocess_image(image_bytes)
        raw_text = extract_text(image_bytes, preprocessed, source="preprocessed")
    else:
        preprocessed = cv2.cvtColor(raw_np, cv2.COLOR_RGB2GRAY)

    print("Final OCR:", raw_text)
    eqn, qn = extract_ai_text(raw_text)
    return preprocessed, raw_text, eqn, qn


def build_prompt(eqn: str, qn: str, stream=False):
    if stream:
        return f"""
You are a highly intelligent math tutor AI.

A student has submitted a math problem extracted from an image. Your job is to:
1. Detect and understand the math expression.
2. Solve it step-by-step.
3. Conclude with the final answer.

If unsolvable, reply: "No clear math problem was detected."

---

**Extracted Math Expression**:
{eqn}
{f'**Related Question:** {qn}' if qn else ''}
"""
    else:
        return f"""
You are a math solving assistant.

Evaluate the following LaTeX-style math expression and return ONLY the final numeric or simplified symbolic result.

DO NOT provide steps.
DO NOT explain.

---

Problem:
{eqn}
"""


@router.post("/preprocessed-image")
async def get_preprocessed_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    pre, _, _, _ = process_image(image_bytes)
    _, buffer = cv2.imencode(".png", pre)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")


@router.post("/stream")
async def stream_equation_solution(file: UploadFile = File(...)):
    image_bytes = await file.read()
    _, raw_text, eqn, qn = process_image(image_bytes)
    corrected_eqn = correct_latex_ocr(raw_text).splitlines()[0].strip()
    prompt = build_prompt(corrected_eqn, qn, stream=True)

    def stream_llm():
        with requests.post("http://localhost:11434/api/generate", json={
            "model": "wizard-math:13b", "prompt": prompt, "stream": True
        }, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    try:
                        yield json.loads(line.decode("utf-8")).get("response", "")
                    except:
                        continue

    return StreamingResponse(stream_llm(), media_type="text/plain")


@router.post("/query")
async def extract_equation(file: UploadFile = File(...)):
    image_bytes = await file.read()
    _, raw_text, eqn, qn = process_image(image_bytes)
    corrected_eqn = correct_latex_ocr(raw_text).splitlines()[0].strip()
    return {
        "query": f"{qn}: {eqn}" if qn else eqn,
        "raw": raw_text,
        "corrected": corrected_eqn
    }


@router.post("/ar-query")
async def extract_equation_answer_only(file: UploadFile = File(...)):
    image_bytes = await file.read()
    _, raw_text, eqn, qn = process_image(image_bytes)
    corrected_eqn = correct_latex_ocr(raw_text).splitlines()[0].strip()
    ai = MetaAI()
    prompt = build_prompt(corrected_eqn, qn)
    print(prompt)
    result = ai.prompt(message=prompt)["message"]
    return {
        "query": f"{qn}: {eqn}" if qn else eqn,
        "raw": raw_text,
        "corrected": f"{eqn}, {qn}" if qn else eqn,
        "answer": result
    }
