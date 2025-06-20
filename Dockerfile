# Use NVIDIA CUDA runtime image as the base
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04
LABEL authors="R.Feo"
# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Download and install Miniconda!
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

# Add Conda to PATH
ENV PATH /opt/conda/bin:$PATH

# Update Conda
RUN conda update -n base -c defaults conda -y

# Create a Conda environment
RUN conda create -n myenv scipy scikit-learn=1.5.2 fastapi pydantic stockwell -c conda-forge -y

# Activate the environment and set CUDA variables
ENV PATH /opt/conda/envs/myenv/bin:$PATH \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# LSL
RUN apt-get -y update  && apt-get install -y zip unzip p7zip-full git

COPY ./liblsl-1.16.2-jammy_amd64.deb /liblsl-1.16.2-jammy_amd64.deb
RUN apt-get install -y /liblsl-1.16.2-jammy_amd64.deb && rm /liblsl-1.16.2-jammy_amd64.deb

RUN pip install pylsl

RUN mkdir /app
#VOLUME /model
COPY . /app
WORKDIR /app
#ENTRYPOINT ["top", "-b"] 10.16.78.6
#EXPOSE 16571-16604
CMD ["fastapi", "run", "app/main.py", "--port", "8000"]
