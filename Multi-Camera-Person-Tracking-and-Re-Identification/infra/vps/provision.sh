#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y \
  ca-certificates \
  curl \
  git \
  gnupg \
  lsb-release \
  nginx \
  certbot \
  python3-certbot-nginx \
  docker.io \
  docker-compose-plugin

systemctl enable --now docker
systemctl enable --now nginx

echo "Provisioning completed. Docker and nginx are running."
