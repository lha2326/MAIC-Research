# Reproduction image for hoangvanphi2004/MAIC (pinned commit) + MADDPG baseline.
# Build:  docker build -t maic-repro .
# Run:    docker run --gpus all -v $PWD/runs:/work/runs -v $PWD/reports:/work/reports maic-repro bash scripts/run_all.sh 6
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /work
COPY . /work
RUN rm -rf third_party runs runs_smoke reports reports_smoke && bash scripts/setup.sh
CMD ["bash", "scripts/smoke_test.sh"]
