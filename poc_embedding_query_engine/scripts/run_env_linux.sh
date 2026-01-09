#!/usr/bin/env bash
# ------------------------------------------------------------
# pdf_to_parse_poc — Linux runtime environment loader
# ------------------------------------------------------------
# Purpose:
#   Set all environment variables required to run the pipeline
#   on Linux (DB, embeddings, paths).
#
# Usage:
#   source scripts/run_env_linux.sh
#
# Notes:
#   - Must be *sourced*, not executed, so env vars persist.
#   - Assumes conda env `pdf_to_parse_poc` already exists.
#   - Does NOT activate conda automatically (by design).
# ------------------------------------------------------------


# NOTE:
# This file is meant to be *sourced* into an interactive shell.
# Do NOT use `set -e`, `set -u`, or `set -o pipefail` here, because
# they will leak into the caller shell and can terminate the SSH session
# (e.g., during TAB completion or non-zero-returning helper functions).


# ------------------------
# Project paths
# ------------------------
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src"

# ------------------------
# Database (Postgres)
# ------------------------
export PDF_POC_SCHEMA="pdf_emergency"

 # verify-full requires the host name to match a SAN on the live server certificate.
 # The postgres server cert on llm-db includes DNS:llm-db (and related names), not 127.0.0.1.
export PGHOST="llm-db"
export PGPORT="5432"
export PGDATABASE="medical_documentation_storage"
export PGUSER="joe_admin"
export PGSSLMODE="verify-full"


# SSL client cert auth (canonical shared PKI mount)
# Expected layout (symlinks are fine):
#   /mnt/auth_share/pki/postgres/root.crt
#   /mnt/auth_share/pki/postgres/user/joe_admin/client.crt
#   /mnt/auth_share/pki/postgres/user/joe_admin/client.key
#   /mnt/auth_share/pki/postgres/user/joe_admin/root.crt  (optional, often symlink to root)
#
# Note: libpq requires the private key to be readable by the invoking user and typically 0600.
export PGSSLROOTCERT="/mnt/auth_share/pki/postgres/root.crt"
export PGSSLCERT="/mnt/auth_share/pki/postgres/gui_access_certs/joe_admin_client.crt"
export PGSSLKEY="/mnt/auth_share/pki/postgres/gui_access_certs/joe_admin_client.key"

# ------------------------
# Embeddings (LiteLLM → Infinity)
# ------------------------
export LITELLM_BASE_URL="http://127.0.0.1:4000"
export LITELLM_API_KEY="POMPY"
export EMBEDDINGS_MODEL="local-embed"

# ------------------------
# Runtime behavior
# ------------------------
# Avoid noisy pip warnings
export PIP_DISABLE_PIP_VERSION_CHECK=1

# Ensure predictable unicode handling
export PYTHONUTF8=1

# ------------------------
# Sanity output
# ------------------------
echo "✅ pdf_to_parse_poc Linux environment loaded"
echo "  PROJECT_ROOT       = ${PROJECT_ROOT}"
echo "  PYTHONPATH         = ${PYTHONPATH}"
echo "  PGDATABASE         = ${PGDATABASE}"
echo "  PGHOST             = ${PGHOST}"
echo "  PDF_POC_SCHEMA     = ${PDF_POC_SCHEMA}"
echo "  LITELLM_BASE_URL   = ${LITELLM_BASE_URL}"
echo "  EMBEDDINGS_MODEL   = ${EMBEDDINGS_MODEL}"
echo
echo "ℹ️  Remember to activate conda:"
echo "    conda activate pdf_to_parse_poc"
