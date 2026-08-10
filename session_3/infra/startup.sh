#!/usr/bin/env bash
# infra/startup.sh — runs as root on the VM's first boot (and on every boot).
#
# This is the automated version of the manual "SSH in and type these" steps:
#   curl -fsSL https://get.docker.com | sh
#   docker pull  yourusername/ride-api:latest
#   docker stop/rm ride-api
#   docker run -d --name ride-api --restart unless-stopped -p 8000:8000 ...
#
# Terraform renders the $${...} placeholders below via templatefile() before the
# VM ever sees this file — so a literal dollar-brace in this script must be
# escaped as $$ or Terraform tries to evaluate it as an expression.
#
# Logs land in the serial console:
#   gcloud compute instances get-serial-port-output <vm> --zone <zone>
set -euxo pipefail

# ── Install Docker (idempotent — skipped on reboots) ──
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
  systemctl enable --now docker
fi

# ── Deploy / redeploy the container ───────────────────
docker pull "${docker_image}"

docker stop "${container_name}" 2>/dev/null || true
docker rm   "${container_name}" 2>/dev/null || true

docker run -d \
  --name "${container_name}" \
  --restart unless-stopped \
  -p "${app_port}:${app_port}" \
  -e MODEL_PATH="${model_path}" \
  "${docker_image}"

echo "startup-script: ${container_name} is up on port ${app_port}"
