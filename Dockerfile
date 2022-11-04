# TensorFlow with GPU support
FROM tensorflow/tensorflow:latest-gpu

# Common environment variables that Python uses
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Update the underlying system
RUN apt update -y --fix-missing && \
    apt clean && \
    apt autoremove && \
    rm -rf /var/lib/apt/lists/*

# Add new package repository and install additional software
RUN add-apt-repository universe && \
    apt update -y --fix-missing && \
    apt install -y graphviz && \
    apt clean && \
    apt autoremove && \
    rm -rf /var/lib/apt/lists/*

# Add dependencies required for opencv-python package
RUN apt update -y --fix-missing && \
    apt install -y ffmpeg libsm6 libxext6  -y && \
    apt clean && \
    apt autoremove && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements, upgrade pip and install Python dependencies
ARG A_REQUIREMENTS_FILE=requirements.txt
COPY ${A_REQUIREMENTS_FILE} .
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r ${A_REQUIREMENTS_FILE}
RUN rm ${A_REQUIREMENTS_FILE}

# Install additional information to help mypy resolve types
RUN mypy --install-types

# Working directory
ARG SRC_DIR
WORKDIR ${SRC_DIR}
