# HF Space entry point — serves the demo (Reachy Mini conversation UI)
# on $PORT (default 7860). The pipeline Dockerfiles (CUDA + uv sync) live
# at Dockerfile.pipeline and Dockerfile.pipeline.arm64 and are unrelated
# to the Space; this file exists because HF Spaces builds the root
# Dockerfile. The demo's own Dockerfile at demo/Dockerfile is equivalent.
FROM python:3.11-slim

WORKDIR /app

COPY demo/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY demo/ ./

EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
