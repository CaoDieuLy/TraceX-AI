#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${1:-${REPO_ROOT}/secret_backups}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BUNDLE_PREFIX="secrets_bundle_${TIMESTAMP}"
TMP_DIR="$(mktemp -d)"
MANIFEST_PATH="${OUTPUT_DIR}/${BUNDLE_PREFIX}.manifest.txt"
ARCHIVE_PATH="${TMP_DIR}/${BUNDLE_PREFIX}.tar.gz"
ENCRYPTED_PATH="${OUTPUT_DIR}/${BUNDLE_PREFIX}.tar.gz.enc"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

mkdir -p "${OUTPUT_DIR}"

if [[ -z "${SECRET_BUNDLE_PASSPHRASE:-}" ]]; then
  read -rsp "Secret bundle passphrase: " SECRET_BUNDLE_PASSPHRASE
  echo
fi

if [[ -z "${SECRET_BUNDLE_PASSPHRASE}" ]]; then
  echo "Passphrase is required." >&2
  exit 1
fi

manifest_tmp="${TMP_DIR}/manifest.txt"

SECRETS_ROOT="${REPO_ROOT}/secrets"

if [[ ! -d "${SECRETS_ROOT}" ]]; then
  echo "Secrets folder not found: ${SECRETS_ROOT}" >&2
  exit 1
fi

find "${SECRETS_ROOT}" -type f \
  ! -name '*.example' \
  ! -name '*.md' \
  | sed "s#^${REPO_ROOT}/##" \
  | sort -u > "${manifest_tmp}"

if [[ ! -s "${manifest_tmp}" ]]; then
  echo "No secret files found to export." >&2
  exit 1
fi

{
  echo "# Secret bundle manifest"
  echo "# Generated at: ${TIMESTAMP}"
  echo "# Repo root: ${REPO_ROOT}"
  cat "${manifest_tmp}"
} > "${MANIFEST_PATH}"

tar -czf "${ARCHIVE_PATH}" -C "${REPO_ROOT}" -T "${manifest_tmp}"

openssl enc -aes-256-cbc -pbkdf2 -salt \
  -in "${ARCHIVE_PATH}" \
  -out "${ENCRYPTED_PATH}" \
  -pass "pass:${SECRET_BUNDLE_PASSPHRASE}"

echo "Encrypted bundle created:"
echo "  ${ENCRYPTED_PATH}"
echo "Manifest created:"
echo "  ${MANIFEST_PATH}"
echo
echo "Bundle contains only the canonical secrets folder."
