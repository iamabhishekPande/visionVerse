# Use the official Python image from the Docker Hub
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app


# Copy the requirements.txt first, to leverage Docker cache
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8503

# Command to run the Streamlit app
#CMD ["streamlit", "run", "app.py", "--server.port=8503", "--server.enableCORS=false"]
CMD ["streamlit", "run", "FaceRecognition.py", "--server.port=8503"]