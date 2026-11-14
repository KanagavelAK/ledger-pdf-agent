"""Entry point for a Hugging Face Space (Gradio SDK): the Space runs `python app.py` and serves port 7860.
We don't use Gradio's UI; this just starts our own server on the port the Space expects."""
import os
import sys

os.environ.setdefault("HOST", "0.0.0.0")
os.environ.setdefault("PORT", "7860")

import server  # noqa: E402  (reads HOST/PORT above)

if __name__ == "__main__":
    sys.argv = sys.argv[:1]  # ignore any arguments the platform passes
    server.main()
