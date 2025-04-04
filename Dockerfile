FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app
COPY . /app

# Install git using conda instead of apt-get
RUN conda install -y git

# Create and activate the conda environment
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "One-DM", "/bin/bash", "-c"]

# Install additional Python packages
RUN pip install gcloud

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "One-DM", "python", "train_finetune.py"]