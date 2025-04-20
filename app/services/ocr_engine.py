import pytesseract
from PIL import Image
from meta_ai_api import MetaAI
from pix2tex.cli import LatexOCR

latex_model = LatexOCR()


def correct_latex_ocr(raw_text: str):
    prompt = f"""
You are a world-class mathematical OCR corrector.

Your ONLY job is to:
- Receive noisy or malformed math expressions extracted via OCR.
- Output a single clean LaTeX or plain math equation or expression.
- DO NOT include any explanation, comments, or extra text — only the corrected math.

Rules:
- Fix common OCR mistakes (e.g., "4 2n^3" → "4(2n^3)", "S" → "5", "l" → "1")
- Correct invalid syntax or formatting (e.g., unmatched brackets, misplaced operators)
- Remove any non-math text or natural language (e.g., "solve", "evaluate", "answer is")

Examples (bad input → corrected output):

Input: 4(2ns 3)n=2  
Output: 4(2n^3)n = 2

Input: solve ∫x^2dx  
Output: \\int x^2 \\, dx

Input: 2 x + = 7  
Output: 2x + 0 = 7

Input: lim x→0 sinx/x  
Output: \\lim_{{x \\to 0}} \\frac{{\\sin x}}{{x}}

Input: 9x+7/2 - (x-2/7) = 36Iamstillimproving...
Output: \\frac{{9x+7}}{2} - \\left(x - \\frac{{2}}{7}\\right) = 36

Now correct this (return only the fixed math expression, nothing else):
{raw_text}
""".strip()

    ai = MetaAI()
    response = ai.prompt(message=prompt)
    return response["message"].strip()


def extract_ai_text(dirty_equation: str):
    is_clean = len(dirty_equation) > 4 and (
            "\\" in dirty_equation and any(op in dirty_equation.lower() for op in ["sum", "int", "frac", "lim"])
    )

    if is_clean:
        query = f"""
You are a LaTeX math interpreter.

The input below is a clean LaTeX math expression or formula. Your task is to:
- Evaluate or simplify it (if it's a summation, integral, or algebraic equation).
- If a variable is involved, and a question is implied (e.g., "solve for n"), ask the question.
- Do NOT reformat or restate the LaTeX — just solve or interpret.

Example:
Input: \\sum_{{n=2}}^{{4}} (2n^2 + 3)
Output: 53, what is the value of the summation?

Now interpret or evaluate:
{dirty_equation}
"""
    else:
        query = f"""
You are a highly intelligent mathematical equation corrector and interpreter.

Your task is to take a noisy or corrupted mathematical expression — possibly extracted from OCR or handwriting — and output a cleaned, accurate mathematical equation and the associated question if it exists.

The input might contain:
- Typos, malformed LaTeX syntax, and invalid operators
- Wrong symbols (e.g., 'Z' instead of 'X' from OCR)
- Extraneous natural language (e.g., "solve this", "what is", etc.)

Your job is to:
- Fix syntax and structure (e.g., '^{{\\times}}' → '^2', 'Z' → 'x')
- Format as clean standard math (e.g., 2x^2 + 3x + 4 = 0)
- Append a clear natural question if one is implied (e.g., "what is x?")
- Output **only** the cleaned equation and question (if present), separated by a comma.

Now correct this:
{dirty_equation}
"""

    ai = MetaAI()
    response = ai.prompt(message=query)
    cleansed_response = response["message"]
    print("AI Response:", cleansed_response)

    parts = [part.strip() for part in cleansed_response.split(",", 1)]
    if len(parts) == 2:
        return parts[0], parts[1]
    return parts[0], ""


def extract_text(image_bytes, image_np, source="raw"):
    try:
        pil_img = Image.fromarray(image_np).convert("RGB")

        # --- Try LaTeX-based OCR (e.g., Pix2Tex) ---
        latex = latex_model(pil_img).strip()

        if not latex or is_gibberish(latex):
            print(f"{source.upper()} Pix2Tex output gibberish. Switching to Tesseract...")

            # --- Use Tesseract with multiple configs if needed ---
            raw_text = pytesseract.image_to_string(pil_img, config='--psm 6').strip()
            cleaned_text = filter_noise(raw_text)

            if not is_valid_math(cleaned_text):
                # Try another psm mode if the first output is weak
                print("Trying Tesseract with alternate config (psm 11)...")
                raw_text_alt = pytesseract.image_to_string(pil_img, config='--psm 11').strip()
                cleaned_text_alt = filter_noise(raw_text_alt)

                # Choose the better of the two cleaned texts
                return choose_better_output(cleaned_text, cleaned_text_alt)

            return cleaned_text

        # If Pix2Tex output looks okay
        return postprocess_latex(latex)

    except Exception as e:
        print(f"OCR extraction failed ({source}): {e}")
        return ""


def is_gibberish(text):
    return len(text) < 3 or not any(c.isalnum() for c in text)


def is_valid_math(text):
    return any(op in text for op in ['=', '^', '+', '-', '*', '/'])


def filter_noise(text):
    # Clean known OCR errors
    replacements = {
        # Smart quotes & dashes
        "’": "'",
        "“": "\"", "”": "\"",
        "−": "-",  # OCR dash
        "—": "-",  # em dash
        "–": "-",  # en dash

        # Basic math operators
        "×": "*",  # Multiplication
        "÷": "/",  # Division
        "^": "**",  # Exponentiation

        # Calculus symbols
        "∫": "integrate",
        "∂": "d",
        "∇": "nabla",  # Gradient symbol

        # Summation & roots
        "∑": "sum",
        "√": "sqrt",
        "∘": "o",  # Composition

        # Constants & numbers
        "π": "pi",
        "∞": "oo",
        "ε": "epsilon",
        "′": "'",  # Prime notation
        "″": "''",  # Double prime
        "°": "deg",  # Degree symbol

        # Logic & set notation
        "≈": "~",
        "≅": "~=",
        "≠": "!=",
        "≤": "<=",
        "≥": ">=",
        "→": "->",
        "⇒": "=>",
        "⇔": "<=>",
        "∝": "proportional_to",
        "∈": "in",
        "∉": "notin",
        "∅": "emptyset",
        "∩": "intersect",
        "∪": "union",

        # Sets and number types
        "ℝ": "R",
        "ℤ": "Z",
        "ℕ": "N",
        "ℚ": "Q",

        # Misc
        "…": "...",  # Ellipsis

        # Digits ↔ Letters
        "S": "5",  # Curved 5 misread as capital S
        "s": "5",  # Lowercase s → 5
        "O": "0",  # Capital O → Zero
        "o": "0",  # Lowercase o → Zero
        "D": "0",  # Capital D → Zero in sloppy fonts
        "Q": "0",  # Looped Q → Zero
        "Z": "2",  # Capital Z → 2
        "z": "2",  # Lowercase z → 2
        "I": "1",  # Capital i → 1
        "l": "1",  # Lowercase L → 1
        "i": "1",  # Dotted i in OCR → 1
        "B": "8",  # Capital B → 8
        "G": "6",  # Capital G → 6
        "g": "9",  # Sloppy g → 9
        "A": "4",  # A in weird fonts → 4

        # Operators ↔ Symbols
        "|": "1",  # Vertical bar → 1
        "~": "-",  # OCR fuzzy tilde → minus
        "_": "-",  # Underscore → dash
        "—": "-",  # em dash → minus
        "–": "-",  # en dash → minus

        # Parentheses & brackets
        "[": "(",  # Left square bracket → left paren
        "]": ")",  # Right square bracket → right paren
        "{": "(",  # Curly bracket → paren
        "}": ")",  # Close curly → paren
    }

    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    return text.strip()


def postprocess_latex(latex):
    # Clean LaTeX if needed (optional)
    return latex.replace(" ", "")


def choose_better_output(text1, text2):
    # Pick the text that looks more mathematical
    score1 = sum(text1.count(op) for op in "=^*/+-")
    score2 = sum(text2.count(op) for op in "=^*/+-")
    return text1 if score1 >= score2 else text2
