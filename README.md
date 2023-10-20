# Btech Project

Implementation of IAANet with LESPS framework.


## Dependencies

* Torch, Torchvision (using CPU, Cude not setup yet)
  ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

  ```
* Matplotlib
  ```bash
    pip install matplotlib
  ```
* Skimage, Tqdm
  ```bash
    pip install scikit-image tqdm
  ```


## Running the code

* Training
  ```bash
    python train.py --dataset_names SIRST3 --label_type centroid

  ```
    
