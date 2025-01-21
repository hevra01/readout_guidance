#!/bin/bash

# Activate the Conda environment
CONDA_ROOT=$HOME/miniconda3
CONDA=${CONDA_ROOT}/bin/conda
ENV_NAME="readout_new"

echo "home is ${HOME}"
echo "CONDA path is set to: ${CONDA}"

echo "Attempting to run conda from: ${CONDA}"



# Ensure Conda is available
if [ ! -f ${CONDA} ]; then
    echo "Conda is not installed at ${CONDA}. Exiting..."
fi

echo "Installing GLib in the Conda environment: $ENV_NAME"
${CONDA} run -n ${ENV_NAME} bash -c "

conda list 
"
echo "numpy installation completed."

