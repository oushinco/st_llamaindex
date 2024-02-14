# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Set the Transformers cache directory to /app/cache (or any other writable path)
ENV TRANSFORMERS_CACHE /tmp/cache

# # Install system dependencies required for building C/C++ extensions
# RUN apt-get update && apt-get install -y \
#     build-essential \  # Includes GCC/G++ compilers
#     cmake \            # CMake for building C/C++ extensions
#     # git \              # Git, in case your dependencies need to fetch code
#     && apt-get clean && rm -rf /var/lib/apt/lists/*  # Clean up


# Copy the local Debian packages
COPY ./packages/*.deb /tmp/packages/

# Install the Debian packages
RUN dpkg -i /tmp/packages/*.deb || apt-get update && apt-get install -yf && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/packages


# Copy the requirements.txt file into the container
COPY requirements.txt ./

# Install project dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install poetry (Python package manager)
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --verbose


# Copy the rest of the application code into the container
COPY . .

# Change ownership and permissions of the /app directory
RUN chmod -R 777 /app

# Set permissions for specific directories (adjust as needed)
RUN chmod -R 777 /tmp

# Copy the script to /tmp
# COPY start_service.sh /tmp/start_service.sh

# Make the start_service.sh script executable if needed
# RUN chmod +x start_service.sh

# Expose the port on which your service runs (7860)
EXPOSE 40000

# Run the start_service.sh script or your application's entry point
CMD ["./start_service.sh"]
