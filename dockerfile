# Use Alpine Linux 3.14 as the base image due to its minimal size and security. Specify the AMD64 architecture for compatibility.
FROM --platform=linux/amd64 alpine:3.14

# Set Python to run in unbuffered mode which is recommended in Docker containers. This avoids Python buffering stdout and stderr.
ENV PYTHONUNBUFFERED=1

# Install build dependencies. These are temporary for building Python and other dependencies and will be removed later to keep the image size small.
RUN apk add --no-cache --virtual .build-deps \
    gcc \
    libc-dev \
    linux-headers \
    make \
    openssl-dev \
    zlib-dev \
    libffi-dev

# Install sqlite and its development headers as a permanent dependency.
RUN apk add --no-cache sqlite sqlite-dev

# Install build-base which includes essential development tools.
RUN apk add --no-cache build-base

# Download the Python source code for the specific version you want to install.
RUN wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tar.xz

# Extract the Python source code archive.
RUN tar -xf Python-3.12.0.tar.xz

# Change the working directory to the extracted Python source directory.
WORKDIR /Python-3.12.0

# Configure Python build with optimizations enabled for better performance.
RUN ./configure --enable-optimizations

# Compile Python source code using make, utilizing four cores for speed.
RUN make -j 4

# Install the newly built Python binary into the system.
RUN make altinstall

# Change back to the root directory.
WORKDIR /

# Clean up the Python source directory and tar file to keep the image size small.
RUN rm -rf Python-3.12.0
RUN rm Python-3.12.0.tar.xz

# Remove the build dependencies as they are no longer needed after the build.
RUN apk del .build-deps

# Create symbolic links for python and pip to ensure the correct version is used.
RUN ln -sf /usr/local/bin/python3.12 /usr/bin/python \
 && ln -sf /usr/local/bin/python3.12 /usr/bin/python3 \
 && ln -sf /usr/local/bin/pip3.12 /usr/bin/pip \
 && ln -sf /usr/local/bin/pip3.12 /usr/bin/pip3

# Install openssh-client, potentially for accessing private repositories or other secure communications.
RUN apk update && apk add openssh-client

# Install gcc and other development tools and libraries, which are permanent dependencies for compiling Python packages.
RUN apk add --no-cache gcc musl-dev python3-dev libffi-dev

# Install poetry, a dependency management and packaging tool in Python, at a specific version.
RUN pip install poetry==1.6.1

# Set environment variables for Poetry to avoid user interaction, use virtual environments inside the project, and define cache directory.
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Indicate that the application is running inside a Docker container.
ENV RUNNING_IN_CONTAINER=yes

# Copy project specification files into the image.
COPY pyproject.toml poetry.lock ./

# Install project dependencies with Poetry, excluding development dependencies, and remove cache to reduce image size.
RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR 

# Copy the application source code into the image.
COPY ./src /src

# Create a directory for logs to ensure it exists and can be written into by the application.
RUN mkdir /logs
