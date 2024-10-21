# Use an official Python image as the base
FROM python:3.11-slim

# Install Tesseract OCR and OpenCV dependencies
RUN apt-get update && apt-get install -y tesseract-ocr \
    libgl1 libglib2.0-0

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app’s source code
COPY . .

# Specify the command to run the app
CMD ["gunicorn", "-b", "0.0.0.0:$PORT", "test:app"]
