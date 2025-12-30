FROM python:3.10-slim

# System libs some Python packages (TF/Pillow/matplotlib) expect
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Headless matplotlib and unbuffered logs
ENV MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Install Python deps first (better Docker caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copy your app, model, and Streamlit config
COPY fashion_predict_app.py .
COPY fashion_cnn.keras ./fashion_cnn.keras
COPY .streamlit ./.streamlit

EXPOSE 8501

# Respect cloud $PORT if present; default to 8501 locally
CMD ["bash", "-lc", "streamlit run fashion_predict_app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
