import re

import pytesseract
from PIL import Image
from meta_ai_api import MetaAI
from pix2tex.cli import LatexOCR

latex_model = LatexOCR()


def is_gibberish(text: str) -> bool:
    """
    Heuristic: if 40%+ of characters are NOT valid math tokens, it's likely gibberish.
    """
    if not text or len(text.strip()) < 5:
        return True
    allowed = re.findall(r'[a-zA-Z0-9=+\-*/^().\[\]{}\\ ]', text)
    return len(allowed) / len(text) < 0.6  # If <60% are valid, it's noise


def extract_ai_text(dirty_equation):
    query = f"""
You are a highly intelligent mathematical equation corrector. 
Your job is to take a noisy, poorly formatted, or corrupted mathematical expression (which may include LaTeX-like syntax, incorrect operators, typos, or natural language words mixed in), and return the **best-guess corrected** equation in standard mathematical notation.

- Correct syntax, symbols, and structure.
- Ignore irrelevant words or malformed LaTeX commands.
- If a question is implied (e.g., “what is x”), include it at the end in natural form.
- Do **not** explain your corrections. Only output the corrected equation and the final question, if present.

Example:
Input: 2X^{{\\times}}2+6\\times+3=0,\\,\\mathrm{{what}}\\,\\mathrm{{ts}}\\times Z,?
Output: 2x^2 + 6x + 3 = 0, what is x?

Now correct this:
{dirty_equation}
    """

    ai = MetaAI()
    response = ai.prompt(message=query)
    cleansed_response = response["message"]
    print("cleansed_response : ", cleansed_response)
    lines = cleansed_response.splitlines()
    for line in lines:
        if re.search(r"[0-9a-zA-Z=+\-*/^().\\]+", line) and not line.strip().lower().startswith("i am"):
            return line.strip()
    return cleansed_response.strip()


def extract_text(image_np):
    try:
        pil_img = Image.fromarray(image_np).convert("RGB")
        latex = latex_model(pil_img).strip()

        if is_gibberish(latex):
            print("⚠️ Pix2Tex produced gibberish. Using Tesseract fallback...")
            return pytesseract.image_to_string(pil_img, config='--psm 6').strip()
        return latex
    except Exception as e:
        print(f"❌ OCR extraction failed: {e}")
        return ""


def split_equation_and_question(text):
    # Normalize spaces and remove newlines
    text = re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()

    # Expanded list of natural language question triggers
    question_keywords = [
        'what', 'solve', 'find', 'calculate', 'determine', 'evaluate',
        'roots', 'value', 'when', 'if', 'how', 'show', 'give', 'derive',
        'compute', 'simplify', 'prove', 'integrate', 'differentiate', 'result'
    ]

    # Lowercase version for keyword search only
    lowered = text.lower()

    # Try to locate where the question likely starts
    question_start_idx = -1
    for kw in question_keywords:
        match = re.search(rf'\b{kw}\b', lowered)
        if match:
            question_start_idx = match.start()
            break

    if question_start_idx != -1:
        equation_raw = text[:question_start_idx].strip(" ,")
        question_raw = text[question_start_idx:].strip()
    else:
        equation_raw = text.strip()
        question_raw = ""

    return equation_raw, question_raw.lower()


def clean_equation(raw_text: str):
    """
    Cleans OCR-generated LaTeX/math expression to make it more readable and processable.
    Returns: (equation_part, question_part)
    """

    if not raw_text:
        return "", ""

    # 1. Fix common OCR/LaTeX misreads
    ocr_corrections = {
        r"\\chi": "x",  # Chi → x
        r"\\theta": "x",  # Theta → x
        r"\\pi": "pi",
        r"\\infty": "oo",
        r"\\cdot": "*",
        r"\\times": "*",
        r"\\div": "/",
        r"\\sqrt": "sqrt",
        r"\\sum": "sum",
        r"\\int": "integrate",
        r"\\left": "", r"\\right": "",

        r"\\begin{.*?}": "",  # Remove LaTeX envs
        r"\\end{.*?}": "",
        r"\\[a-zA-Z]+": "",  # Remove other LaTeX commands
        r"\\mathrm{.*?}": "",  # Remove noisy \mathrm
        r"[a-zA-Z]{3,}": "",  # Remove gibberish like 'what', 'ts', 'is'
    }
    for pattern, replacement in ocr_corrections.items():
        raw_text = re.sub(pattern, replacement, raw_text)

    # 2. Symbolic cleanup
    symbol_map = {
        "−": "-",  # minus
        "×": "*",
        "÷": "/",
        "^": "**",
        "∑": "sum",
        "∫": "integrate",
        "∞": "oo",
        "√": "sqrt",
        "∂": "d",
        "≠": "!=",
        "≤": "<=",
        "≥": ">=",
        "…": "...",
    }
    for symbol, replacement in symbol_map.items():
        raw_text = raw_text.replace(symbol, replacement)

    # 3. General cleanup
    raw_text = raw_text.replace("**{", "**(").replace("}", ")").replace("{", "(")
    raw_text = raw_text.replace(" ", "").replace(",,", ",")
    raw_text = re.sub(r"(?<=[^\d)])\*\*(?=[^\d(])", "*", raw_text)

    # 4. Smart separation of equation and question part
    split_match = re.search(
        r"(.*?)([,;:\.\?]?\s*(solve|what|find|evaluate|compute|determine|is|calculate)\b.*)",
        raw_text,
        re.IGNORECASE
    )
    if split_match:
        equation_part = split_match.group(1).strip()
        question_part = split_match.group(2).strip()
    else:
        equation_part, question_part = raw_text, ""

    return equation_part, question_part
