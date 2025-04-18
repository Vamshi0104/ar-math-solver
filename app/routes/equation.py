import base64
import io
import json
import re
from io import StringIO

import cv2
import requests
from PIL import Image
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from meta_ai_api import MetaAI

from app.services.image_preprocessing import preprocess_image
from app.services.ocr_engine import extract_text, clean_equation, extract_ai_text

router = APIRouter(prefix="/api/equation", tags=["Equation"])


@router.post("/preprocessed-image")
async def get_preprocessed_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    preprocessed_image, _, _ = prepare_image_for_processing(image_bytes)

    _, buffer = cv2.imencode(".png", preprocessed_image)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")


def image_to_base64_jpeg(image_np):
    pil_img = Image.fromarray(image_np).convert("RGB")
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    return encoded_image


def extract_text_from_llava(image_np):
    base64_image = image_to_base64_jpeg(image_np)
    print(base64_image)
    payload = {
        "model": "llava:7b",
        "prompt": "Extract only the math expression in plain text from this image. Do NOT explain anything. No punctuation. Just return the math.",
        "images": [base64_image]
    }
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        if response.ok:
            full = response.text.strip()

            # Heuristic cleanup: remove known noisy prefixes
            full = full.replace("The math equation is:", "").strip()
            full = re.sub(r'^.*?\n\n', '', full)  # remove any intro sentences before double newline
            return full
        else:
            print("❌ LLaVA API error:", response.text)
            return ""
    except Exception as e:
        print("❌ Failed to connect to LLaVA API:", str(e))
        return ""


def prepare_image_for_processing(image_bytes: bytes):
    preprocessed_image = preprocess_image(image_bytes)
    raw_text = extract_text(preprocessed_image)
    # raw_text = extract_text_from_llava(preprocessed_image)
    print("raw_text : ", raw_text)
    corrected_raw_text = extract_ai_text(raw_text)
    print("corrected_raw_text : ", corrected_raw_text)
    equation, question = clean_equation(corrected_raw_text)
    print("equation : ", equation)
    print("question : ", question)
    return preprocessed_image, equation, question


@router.post("/stream")
async def stream_equation_solution(file: UploadFile = File(...)):
    image_bytes = await file.read()
    preprocessed_image, equation, question = prepare_image_for_processing(image_bytes)

    prompt = f"""
    You are an expert math tutor helping a student understand and solve problems step by step.

    Below is some text that was extracted from an image. It may contain a math problem, a question, or both.

    Your task is to:
    1. Identify any math expression, equation, or problem within the text.
    2. Solve it step by step, clearly explaining each stage of the process.
    3. Conclude with the final answer in a clear and friendly way.

    If the text doesn't contain any math at all, respond politely by saying no math problem was found.

    ---

    Extracted Text:
    {equation}

    {f"Additional Question: {question}" if question else ""}
    """

    buffer = StringIO()  # Optional: still collect if needed for later parsing

    def stream_llm():
        with requests.post("http://localhost:11434/api/generate", json={
            "model": "wizard-math:7b",
            "prompt": prompt,
            "stream": True
        }, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        token = chunk.get("response", "")
                        buffer.write(token)
                        yield token
                    except:
                        continue

    return StreamingResponse(
        stream_llm(),
        media_type="text/plain"
    )


@router.post("/query")
async def extract_equation_query(file: UploadFile = File(...)):
    image_bytes = await file.read()
    preprocessed_image = preprocess_image(image_bytes)
    raw_text = extract_text(preprocessed_image)
    corrected_raw_text = extract_ai_text(raw_text)
    equation, question = clean_equation(corrected_raw_text)

    return {
        "query": f"{question}: {equation}" if question else equation,
        "raw": raw_text,
        "corrected": corrected_raw_text
    }


@router.post("/ar-query")
async def extract_equation_query(file: UploadFile = File(...)):
    image_bytes = await file.read()

    # Step 1: Preprocess and extract raw OCR text
    preprocessed_image = preprocess_image(image_bytes)
    raw_text = extract_text(preprocessed_image)

    # Step 2: Correct OCR noise using AI
    corrected_raw_text = extract_ai_text(raw_text)
    equation, question = clean_equation(corrected_raw_text)

    # Step 3: Prompt for final answer only
    full_prompt = f"""
    You are a math solving assistant.

    Evaluate the following LaTeX-style math expression and return ONLY the final numeric or simplified symbolic result.

    DO NOT provide steps.
    DO NOT explain.
    DO NOT rephrase or introduce the answer.

    Just return the final answer as short as possible.

    ---

    Problem:
    {raw_text}
    """
    ai = MetaAI()
    print(full_prompt)
    response = ai.prompt(message=full_prompt)
    cleansed_response = response["message"]

    return {
        "query": f"{question}: {equation}" if question else equation,
        "raw": raw_text,
        "corrected": corrected_raw_text,
        "answer": cleansed_response
    }
