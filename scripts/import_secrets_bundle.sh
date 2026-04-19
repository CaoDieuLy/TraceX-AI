#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/secrets_bundle_*.tar.gz.enc" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUNDLE_PATH="$1"
TMP_DIR="$(mktemp -d)"
ARCHIVE_PATH="${TMP_DIR}/bundle.tar.gz"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

if [[ ! -f "${BUNDLE_PATH}" ]]; then
  echo "Bundle not found: ${BUNDLE_PATH}" >&2
  exit 1
fi

if [[ -z "${SECRET_BUNDLE_PASSPHRASE:-}" ]]; then
  read -rsp "Secret bundle passphrase: " SECRET_BUNDLE_PASSPHRASE
  echo
fi

if [[ -z "${SECRET_BUNDLE_PASSPHRASE}" ]]; then
  echo "Passphrase is required." >&2
  exit 1
fi

openssl enc -d -aes-256-cbc -pbkdf2 \
  -in "${BUNDLE_PATH}" \
  -out "${ARCHIVE_PATH}" \
  -pass "pass:${SECRET_BUNDLE_PASSPHRASE}"

tar -xzf "${ARCHIVE_PATH}" -C "${REPO_ROOT}"

echo "Secrets restored into:"
echo "  ${REPO_ROOT}"
