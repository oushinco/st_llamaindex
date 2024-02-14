# Start from a CUDA-enabled base image
FROM nvidia/cuda:12.1-base

# Set a working directory
WORKDIR /app

# Install Python, pip and necessary build tools
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as the default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Set environment variables to ensure Python outputs are sent straight to terminal without being first buffered
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Optionally, set the Transformers cache directory
ENV TRANSFORMERS_CACHE /app/cache

# Set environment variables for cmake to enable CUDA/cuBLAS
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on"
ENV FORCE_CMAKE=1

# Upgrade pip to its latest version
RUN pip install --no-cache-dir --upgrade pip

# Copy the requirements.txt file into the container
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install llama-cpp-python with verbose output for troubleshooting
RUN pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --verbose

# Copy the rest of the application code into the container
COPY . .

# Expose the port your app runs on
EXPOSE 40000

# Command to run your application
CMD ["python", "your_application.py"]