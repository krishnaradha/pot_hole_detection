FROM mcr.microsoft.com/azureml/minimal-ubuntu20.04:latest

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Copy conda environment and install packages
COPY environment.yml /tmp/environment.yml
RUN conda env update -n base -f /tmp/environment.yml
