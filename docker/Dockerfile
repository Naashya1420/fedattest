# Client/server runtime image for the Docker port of BouquetFL.
# CUDA runtime libraries come bundled with the torch wheel; the host driver is
# injected by the NVIDIA container runtime (--gpus).
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0 \
 && pip install numpy==2.2.6 cryptography==44.0.2 scikit-learn==1.6.1
WORKDIR /app
RUN pip install pyarrow==19.0.1 matplotlib==3.10.1
