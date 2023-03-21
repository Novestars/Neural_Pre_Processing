# Base image
FROM --platform=linux/amd64 ubuntu:20.04

ENV TZ=Europe/Minsk\
    DEBIAN_FRONTEND=noninteractive
# Copy local data
COPY . /

# Set working directory

# Upgrade pip and install Python packages
#RUN python3 -m pip install --upgrade pip
RUN apt-get update && \
    apt-get install -y build-essential python3 python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -r /requirements.txt
# Clean up
RUN rm -rf /root/.cache/pip

# Set the entrypoint
ENTRYPOINT ["python3", "npp.py"]