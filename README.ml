# Vision Transformer Analysis with FSII

This project analyzes Vision Transformer models and produces FSII files, heatmaps, UpSet plots, and other visualizations.  
Results are saved in `output/results`.

## Requirements
- Python 3.12 or compatible
- Git
- NVIDIA GPU with CUDA 11.8 optional for acceleration

## Installation

### 1. Clone the repository
    git clone https://github.com/saitamaIsReal/VisionTransformer_analysis.git
    cd VisionTransformer_analysis

### 2. Create and activate a virtual environment
Windows PowerShell or VS Code Terminal:
    
    python -m venv venv
    venv\Scripts\activate

macOS or Linux:
    
    python3 -m venv venv
    source venv/bin/activate

### 3. Install PyTorch
For CUDA 11.8:
    
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

For CPU only:
    
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

### 4. Install the remaining dependencies
    pip install -r requirements.txt

Current `requirements.txt`:
    
    numpy
    pillow
    matplotlib
    seaborn
    transformers
    shapiq
    ipython
    requests
    overrides

## Verify the installation
Start Python and check versions and CUDA availability:
    
    python

Inside Python:
    
    import torch, transformers
    print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
    print(transformers.__version__)

`torch.cuda.is_available()` should print `True` if a GPU is detected.

## Run
    python visionTransformerFinal.py

Outputs will appear in:
    
    output/results

## Notes
- Install the correct PyTorch build for your hardware. If unsure, see the official selector on pytorch.org.

- If a required package is missing during execution, add it to `requirements.txt`, then run:
    
        pip install -r requirements.txt
