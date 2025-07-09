# CUDA-enabled PyTorch 2.3 image (Python 3.10, CUDA 12.1, cuDNN 8)
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# 1 — install extra Python packages
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2 — copy project code and set default entry-point
COPY . .
RUN chmod +x run_pubmed.sh
CMD ["bash", "run_pubmed.sh"]
