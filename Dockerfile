FROM python:3.6.3-jessie

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PATH /usr/local/nvidia/bin/:$PATH

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64

WORKDIR /stage

# Install base packages.
RUN apt-get update --fix-missing && apt-get install -y \
    bzip2 \
    ca-certificates \
    curl \
    gcc \
    git \
    libc-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    libevent-dev \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -q http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl

# Copy select files needed for installing requirements.
# We only copy what we need here so small changes to the repository does not trigger re-installation of the requirements.
COPY requirements.txt .
COPY scripts/install_requirements.sh scripts/install_requirements.sh
RUN ./scripts/install_requirements.sh

COPY arc_solvers/ arc_solvers/
COPY scripts/ scripts/
RUN ./scripts/download_data.sh

# If you want to run the solver on a new question file, copy them to the image here
# COPY question_file question_file

CMD ["/bin/bash"]
