FROM python:3.10-slim

WORKDIR /app

# Install dependencies for OpenCV and other libraries
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary files
COPY custom_runtime.py ./custom_runtime.py
COPY model-settings.json ./model-settings.json
COPY runs/detect/train/weights/best.pt ./runs/detect/train/weights/best.pt

EXPOSE 8080

CMD ["mlserver", "start", "."]
