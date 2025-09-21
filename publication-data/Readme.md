# Publication Data - DuckNet Research Implementation

This folder contains the complete implementation and dataset for PyTorch object detection using RetinaNet with ResNet-50 backbone, as described in Loken et al. (2025).

**Citation:** Loken, Z. J., Ringelman, K. M., Mini, A., James, D., & Mitchell, M. (2025). *DuckNet: an open-source deep learning tool for waterfowl species identification in UAV imagery.* Remote Sensing in Ecology and Conservation. https://doi.org/10.1002/rse2.70028

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

**Download Images:** [Google Drive Link - DuckNet Images](https://drive.google.com/drive/folders/1MQ3BE6evpqfCSM1Fhc0M9UnJYUX_SIuu?usp=sharing)

- **`images/`** - UAV image dataset (2,142 images) - hosted on Google Drive due to file size

### Model
**Download Model Weights:** [Google Drive Link - DuckNet Model Weights](https://drive.google.com/file/d/1GSg8q944VyuujwB1n1AXVQMyy-i0NSML/view?usp=sharing)

- **`RetinaNet_ResNet50_FPN_DuckNet.pth`** - Trained model weights

### License
- **`LICENSE.pdf`** - CC BY-NC license agreement with Wiley (applies to all folder contents--including images, annotations, model weights, and code)

## Quick Start

1. Create the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate bohb_pt
   ```

2. Download the images and model weights from the Google Drive links above

3. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook RetinaNet_ResNet50_PyTorch_CustomDataset.ipynb
   ```

## Citation

If you use this research implementation, please cite:

```bibtex
@article{loken2025ducknet,
  title={DuckNet: an open-source deep learning tool for waterfowl species identification in UAV imagery},
  author={Loken, Z. J. and Ringelman, K. M. and Mini, A. and James, D. and Mitchell, M.},
  journal={Remote Sensing in Ecology and Conservation},
  year={2025},
  doi={10.1002/rse2.70028}
}
```

---

**Questions?** Contact Zack Loken (zack@savannainstitute.org)