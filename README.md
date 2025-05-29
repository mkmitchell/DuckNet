# 🦆 DuckNet

**DuckNet** is an open-source, deep learning-based tool for detecting, localizing, and classifying waterbird species in drone imagery. It adapts the excellent work of [BatNet](https://github.com/GabiK-bat/BatNet).

DuckNet allows users to process custom UAV image datasets using a baseline model developed by Loken et al. (2025, in review). The baseline model—**RetinaNet** with a **ResNet-50** backbone—is trained to identify:

- **Seven drake (male) waterfowl species**:  
  - Gadwall (*Mareca strepera*)  
  - Green-winged teal (*Anas carolinensis*)  
  - Northern pintail (*Anas acuta*)  
  - Northern shoveler (*Spatula clypeata*)  
  - Mallard (*Anas platyrhynchos*)  
  - Redhead (*Aythya americana*)  
  - Ring-necked duck (*Aythya collaris*)
- **A hen (female) waterfowl class** (species-agnostic)
- **American coot** (*Fulica americana*)

Users can fine-tune the baseline model with custom, annotated drone imagery to add additional species, distinguish between life stages (e.g., juvenile vs. adult), or identify different plumage types (e.g., breeding vs. non-breeding). Any visual distinction that can be captured in a UAV image and represented by a labeled bounding box is supported.

DuckNet includes:

- An integrated **annotation interface** for labeling data (outputs LabelMe-format JSON)
- Support for importing existing LabelMe-format annotation files
- Tools to save fine-tuned models for future reuse

---

## 📖 Citation

Loken, Z. J., Ringelman, K. M., Mini, A., James, D., & Mitchell, M. (2025). *DuckNet: an open-source deep learning tool for waterfowl species identification in UAV imagery.* Remote Sensing in Ecology and Conservation (In review).

- **Corresponding author**: Zack Loken (zack@savannainstitute.org)  
- **Lead developer**: Mike Mitchell (mmitchell@ducks.org)

---

## 🛡️ License

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]  

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/  
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png  
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
