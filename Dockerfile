# basic-kb as a served instance.
#
# Deliberately installs `[serve]` and not `[local]`: an instance that embeds through an API
# needs no on-device model, and skipping fastembed keeps onnxruntime's ~60 MB of native
# wheels out of the image. An instance that embeds locally should build with
# `--build-arg EXTRAS=serve,local` instead.
#
# The container reads its sources and writes its store through bind mounts. Mount them at
# the same absolute paths the host uses, so one config works in both places and nothing
# needs a container-specific override.
FROM python:3.12-slim

ARG EXTRAS=serve

# Owner of the bind-mounted store on the host. The writer lock and the SQLite file are
# shared with any CLI on the host, so the uid has to match or both break.
ARG UID=1001
ARG GID=1001

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY basic_kb ./basic_kb
RUN pip install ".[${EXTRAS}]"

RUN groupadd -g "${GID}" app && useradd -u "${UID}" -g "${GID}" -m app
USER app

# No CMD arguments beyond the subcommand: host, port, auth and watch come from the
# instance's `serve:` block, so the image stays identical across instances.
ENTRYPOINT ["basic-kb"]
CMD ["serve", "--watch"]
