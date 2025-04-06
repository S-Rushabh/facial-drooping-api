FROM python:3.10

# Install OS packages required by dlib and OpenCV
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    libboost-all-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
