import shutil
import subprocess
import time

import uvicorn


def start_ollama_model():
    try:
        if not shutil.which("ollama"):
            print("Ollama is not installed. Please install it from https://ollama.com/")
            return
        check_model = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if "wizard-math:13b" not in check_model.stdout:
            print("Model not found. Pulling wizard-math:13b...")
            subprocess.run(["ollama", "pull", "wizard-math:13b"], check=True)
        else:
            print("Model wizard-math:13b is already available.")
        print("Starting Ollama model: wizard-math:13b ...")
        subprocess.Popen(["ollama", "run", "wizard-math:13b"])
        time.sleep(5)

    except Exception as e:
        print("Failed to start Ollama model:", e)


if __name__ == "__main__":
    start_ollama_model()
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
