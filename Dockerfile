# 1. Use an official Python base image
FROM python:3.10-slim

# 2. Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
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

# 6. Pre-download backbone weights at BUILD time so they are baked into the
#    image layer and never re-fetched at runtime.
#    Weights are cached to $HOME/.cache/{torch,huggingface} inside the image.
RUN python - <<'EOF'
import torchvision.models as m
print("Downloading ResNet-18...")
m.resnet18(weights=m.ResNet18_Weights.DEFAULT)
print("Downloading MobileNetV3-Small...")
m.mobilenet_v3_small(weights=m.MobileNet_V3_Small_Weights.DEFAULT)
print("Downloading MobileViT-XXS...")
import timm
timm.create_model("mobilevit_xxs.cvnets_in1k", pretrained=True, num_classes=0)
print("All backbone weights cached.")
EOF

# 7. Copy the rest of the project
COPY --chown=user . .

# 7. Expose Streamlit port
EXPOSE 7860

# 8. Run the app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
