FROM python:3.11-slim

WORKDIR /code

# 1. Install system-level computer vision and PDF tools
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# 3. Install all the Python packages (THIS IS WHAT WAS MISSING!)
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your application code
COPY . /code