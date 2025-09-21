# Data Folder Contents

This folder contains the complete implementation and dataset for PyTorch object detection using RetinaNet with ResNet-50 backbone.

## Directory Structure

### Code
- **`RetinaNet_ResNet50_PyTorch_CustomDataset.ipynb`** - Complete Jupyter notebook for PyTorch object detection on custom datasets, including:
  - Image and annotation data preprocessing
  - Hyperparameter tuning with Bayesian Optimization
  - Gradient accumulation-enabled fine-tuning of RetinaNet (COCO pre-trained) and ResNet-50 (ImageNet pre-trained)
  - Model inference and evaluation on test dataset
  - Test set prediction visualization
  - Training/validation metrics plotting
  - Confusion matrix generation

- **`coco_eval.py`** - COCO-style dataset evaluation tools
- **`coco_utils.py`** - COCO-style dataset utilities
- **`engine_gradientAccumulation.py`** - Gradient accumulation-enabled training and evaluation engines
- **`environment.yml`** - Conda environment configuration for reproducing the analysis
- **`transforms.py`** - PyTorch transformation functions for object detection
- **`utils.py`** - Utility functions for training and evaluation

### Data
- **`annotations/`** - Annotation files in Darwin JSON 2.0 format (polygon and XYWH bounding boxes)
- **`images/`** - UAV image dataset

### Model
- **`model/RetinaNet_ResNet50_FPN_DuckNet.pth`** - Trained model weights

### License
- **`LICENSE.pdf`** - CC BY-NC license agreement with Wiley (applies to all folder contents)

## Quick Start

1. Clone the Conda environment:
   ```bash
   conda env create -f publication-data/environment.yml
   ```

2. Open and run the Jupyter notebook

## License

The work in this folder is licensed under CC BY-NC (see `LICENSE.pdf` for details).
