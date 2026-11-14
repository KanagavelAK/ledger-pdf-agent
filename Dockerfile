# Ledger as a container, e.g. a Hugging Face Docker Space.
# API keys and DEMO_ACCESS_CODE are NOT baked in: set them as secrets on the host.
FROM python:3.11-slim

# Hugging Face runs Spaces as user 1000; the app writes traces/ in its own folder, so the user owns it.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Listen on all interfaces inside the container; 7860 is the port Hugging Face Spaces expects.
ENV HOST=0.0.0.0 PORT=7860
EXPOSE 7860
CMD ["python", "server.py"]
