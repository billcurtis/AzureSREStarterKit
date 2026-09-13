#!/usr/bin/env bash
set -euo pipefail

# scripts/validate_sre_agent.sh
# Lightweight, non-destructive validation for the SRE Agent Terraform configuration.
# Usage:
#   ./scripts/validate_sre_agent.sh [path-to-terraform-dir]
# Defaults to current directory if no path provided.

TF_DIR="${1:-.}"

echo "Validating Terraform in: ${TF_DIR}"

# Tool checks
missing_tools=()
for tool in terraform; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    missing_tools+=("$tool")
  fi
done

if [ ${#missing_tools[@]} -gt 0 ]; then
  echo "Error: required tools missing: ${missing_tools[*]}"
  echo "Install 'terraform' and re-run. Optional tools: tflint, tfsec (if present they will be run)."
  exit 2
fi

# 1) Formatting check
echo "Running: terraform fmt -check -recursive ${TF_DIR}"
if ! terraform fmt -check -recursive "${TF_DIR}"; then
  echo "terraform fmt check failed. Run 'terraform fmt -recursive ${TF_DIR}' to fix formatting."
  exit 3
fi

# 2) Init without backend to avoid touching remote state
echo "Running: terraform init -backend=false in ${TF_DIR}"
pushd "${TF_DIR}" >/dev/null
terraform init -input=false -backend=false >/dev/null

# 3) Static validate
echo "Running: terraform validate"
if ! terraform validate -no-color; then
  echo "terraform validate failed. Fix configuration before applying."
  popd >/dev/null
  exit 4
fi

# 4) Optional static analyzers (best-effort, non-blocking if not installed)
if command -v tflint >/dev/null 2>&1; then
  echo "Running: tflint"
  if ! tflint --init >/dev/null 2>&1 && ! tflint >/dev/null 2>&1; then
    echo "tflint reported issues. Review tflint output above."
    popd >/dev/null
    exit 5
  else
    echo "tflint passed (or had no actionable output)."
  fi
else
  echo "tflint not installed — skipping. (Recommended for provider/best-practice checks)"
fi

if command -v tfsec >/dev/null 2>&1; then
  echo "Running: tfsec"
  if ! tfsec --exclude-checks=AVD* .; then
    echo "tfsec reported issues. Review tfsec output above."
    popd >/dev/null
    exit 6
  else
    echo "tfsec passed (or had no actionable output)."
  fi
else
  echo "tfsec not installed — skipping. (Recommended for security scanning)"
fi

popd >/dev/null

echo "Validation completed successfully for ${TF_DIR}."
