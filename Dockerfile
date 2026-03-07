# 1. Use an official Python base image
FROM python:3.10-slim

# 2. Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Create non-root user required by Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 4. Set working directory
WORKDIR $HOME/app

# 5. Copy and install requirements — install everything in one pass so the
#    dependency resolver won't re-pull a CUDA torch when ultralytics is installed.
#    --index-url sets PyTorch CPU wheel index as primary;
#    --extra-index-url adds PyPI for all other packages.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple/ \
    torch torchvision \
    -r requirements.txt

# 6. Copy the rest of the project
COPY --chown=user . .

# 7. Expose Streamlit port
EXPOSE 7860

# 8. Run the app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
