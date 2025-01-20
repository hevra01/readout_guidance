#!/bin/bash

# Define paths
CONDA_ROOT=$HOME/miniconda3
CONDA=${CONDA_ROOT}/bin/conda
ENV_NAME="readout_new"
YAML_FILE="$HOME/readout_guidance/env.yaml"  # Explicit path to the YAML file

# Check if PROJECT_ROOT is set
if [[ -z "${PROJECT_ROOT}" ]]; then
    echo "'PROJECT_ROOT' is not set. Ensure the submit file contains the line 'environment = PROJECT_ROOT=\$ENV(PWD)'"
    exit 1
else
    echo "PROJECT_ROOT=$PROJECT_ROOT"
fi

# Ensure Conda is available
if [ ! -f ${CONDA} ]; then
    echo "Conda is not installed at ${CONDA}. Install miniconda3 and try again."
    exit 1
fi

# Check if the YAML file exists
if [[ ! -f "${YAML_FILE}" ]]; then
    echo "YAML file not found at ${YAML_FILE}. Ensure it exists and try again."
    exit 1
fi

# Setup the Conda environment
echo "Setting up Conda environment $ENV_NAME using $YAML_FILE..."
cd ${PROJECT_ROOT}
${CONDA} env create -f ${YAML_FILE} --prefix ${CONDA_ROOT}/envs/${ENV_NAME} || ${CONDA} env update -f ${YAML_FILE} --prefix ${CONDA_ROOT}/envs/${ENV_NAME}
echo "Environment setup complete."

