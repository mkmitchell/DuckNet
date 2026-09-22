# syntax=docker/dockerfile:1
# The torch cu130 wheels bundle the CUDA runtime, so a plain Python image is
# enough; GPU access comes from the NVIDIA container toolkit (--gpus all)
# and a host driver that supports CUDA 13.

FROM python:3.14-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

# soft_nms is a C++ torch extension with no published wheel; build it once
# here against the pinned torch and copy only the wheel into the runtime.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --no-build-isolation --no-deps -w /wheels \
    git+https://github.com/MrParosk/soft_nms.git@446ee47f34a269bdb72a2bb63617c64c74633a73


FROM python:3.14-slim

ENV INSTANCE_PATH="/app" \
    ROOT_PATH="/app" \
    CONFIG_PATH="/app/settings.json" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED="1"

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /app/requirements.txt

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

COPY . /app

# touch first: zip cannot encode pre-1980 mtimes, which checkouts may carry.
RUN mkdir -p /app/models/detection \
    && cd /app/models_src/2024-10-11 \
    && find basemodel.pt -type f -exec touch {} + \
    && python -m zipfile -c /app/models/detection/basemodel.pt.zip basemodel.pt

EXPOSE 5050
CMD ["python", "mainwaitress.py"]
