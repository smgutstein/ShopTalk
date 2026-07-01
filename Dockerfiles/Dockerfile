# Project GPU runtime:
#   PyTorch 2.7.1
#   CUDA 12.8
#   cuDNN 9
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

# Avoid interactive apt prompts during Docker builds.
# Disable pip's package cache to keep the image smaller.
# Prevent Python from writing __pycache__/.pyc files.
# Force Python logs/output to appear immediately in Docker logs.
# Use a non-GUI Matplotlib backend suitable for containers.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# Use /workspace as the default project directory inside the container.
WORKDIR /workspace

# Install small OS-level dependencies.
# ca-certificates is needed for HTTPS access by pip, Python, git, etc.
# gosu lets the entrypoint drop from root to a host-matching non-root user.
# --no-install-recommends avoids unnecessary optional packages.
# Removing /var/lib/apt/lists keeps the final image smaller.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      gosu \
      git \
      bash \
  && rm -rf /var/lib/apt/lists/*

# Update Python packaging tools before installing project dependencies.
# This helps avoid package build/install issues later.
# Imagebind imports pkg_resources, so keep setuptools below 81.
RUN pip install --upgrade pip wheel \
 && pip install "setuptools<81"


# Install ImageBind.
ARG IMAGEBIND_COMMIT=53680b02d7e37b19b124fa37bae4b6c98c38f5be

WORKDIR /opt
RUN git clone https://github.com/facebookresearch/ImageBind.git /opt/ImageBind \
    && cd /opt/ImageBind \
    && git checkout ${IMAGEBIND_COMMIT} \
    && pip install .

# Verify expected ImageBind source commit and installed package version.
RUN cd /opt/ImageBind && python - <<PY
import subprocess
import importlib.metadata as md
import imagebind

expected_commit = "${IMAGEBIND_COMMIT}"
actual_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    text=True,
).strip()

if actual_commit != expected_commit:
    raise RuntimeError(
        f"Unexpected ImageBind commit: expected {expected_commit}, got {actual_commit}"
    )

expected_version = "0.1.0"
actual_version = md.version("imagebind")

if actual_version != expected_version:
    raise RuntimeError(
        f"Unexpected ImageBind version: expected {expected_version}, got {actual_version}"
    )

print(f"imagebind commit: {actual_commit}")
print(f"imagebind distribution version: {actual_version}")
print(f"imagebind module path: {imagebind.__file__}")
PY


# Install project dependencies as root. The entrypoint will drop privileges
# at container startup, after the Python environment has already been built.
WORKDIR /workspace
COPY Dockerfiles/requirements-docker.txt .
RUN pip install -r requirements-docker.txt
 
# Runtime user mapping is handled by the entrypoint. The image starts as root,
# creates/reuses a user matching the host UID/GID, then runs the requested
# command through gosu as that non-root user.
COPY Dockerfiles/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
 
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["tail", "-f", "/dev/null"]




