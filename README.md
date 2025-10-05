# A Progressive Risk Formulation for Enhanced Deep Learning-Based Total Knee Replacement Prediction in Knee Osteoarthritis

This repository contains the official PyTorch implementation for the paper: **A Progressive risk formulation for enhanced deep learning based total knee replacement prediction in knee osteoarthritis** ([Rajamohan et al., 2025](https://link.springer.com/article/10.1007/s11760-025-04404-0)).

## Description

This project provides a deep learning framework to predict the likelihood of a patient requiring Total Knee Replacement (TKR) surgery. The models are designed to work with both **knee radiographs** and **MRI scans** and can predict the risk of TKR over multiple time horizons: **1, 2, and 4 years**.

The core contribution of this work is a novel **"progressive risk formulation"**. This approach incorporates the clinical understanding that osteoarthritis is a degenerative disease, meaning the risk of TKR for a patient should either increase or remain stable over time. By enforcing this constraint during training, the models learn a more robust representation of disease progression, leading to improved prediction accuracy compared to standard baseline models.

The code is organized to facilitate the training and evaluation of various experimental setups described in the paper, including:

  * `Baseline`: A standard deep learning model without the progressive risk constraint.
  * `RiskFORM1`, `RiskFORM2`, `ConReg`, `RiskReg`: Different implementations of the progressive risk formulation and related techniques.

The models in this study were trained and evaluated on data from the Osteoarthritis Initiative (OAI) and the Multicenter Osteoarthritis Study (MOST) datasets.

## Data Acquisition and Setup

### Data Availability

The datasets used in this research are publicly available but require applications for access.

  * **Osteoarthritis Initiative (OAI):** OAI data (imaging and clinical) are distributed via the NIMH Data Archive (NDA). Access requires: (i) creating or linking an NDA account (eRA Commons / Login.gov / PIV/CAC); (ii) agreeing to the OAI data access terms within the NDA; and (iii) requesting the desired OAI collections through the NDA portal. Detailed instructions are provided on the NDA OAI access pages: [https://nda.nih.gov/oai](https://nda.nih.gov/oai)
  * **Multicenter Osteoarthritis Study (MOST):** MOST data are available through the NIA Aging Research Biobank. Investigators create a Biobank account and submit a data request in accordance with the Biobank’s terms and conditions. Current distribution is through the Aging Research Biobank: [https://agingresearchbiobank.nia.nih.gov/](https://agingresearchbiobank.nia.nih.gov/)

### Code and Data Setup

**IMPORTANT:** Before running any scripts, you must update the hardcoded paths for the data and CSV files.

        ```
**Image Data Path Setup:** The path to the directory containing the HDF5 image files needs to be specified in the dataloader scripts.

      * **Files to Modify:** `dataloader.py` and `XrayDataLoader.py` in **every** experiment sub-directory.
      * **Action:** In the `__getitem__` method of the `Radiographloader` or `MRIloader` class, you must define the base path to your image data and prepend it to the filename that is read from the CSV.
        ```python
        # Example in a dataloader's __getitem__ method:
        data_path = "/path/to/your/downloaded/hdf5_images/"
        file_name = self.df.iloc[idx]['h5Name']
        image_path = data_path + file_name

        with h5py.File(image_path, 'r') as hf:
            image = hf['image'][:]
        ```

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/denizlab/TKRRisk-Formulation.git
    cd TKRRisk-Formulation
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment. The required packages can be installed using the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

## Training and Evaluation Workflow

The workflow is a two-step process:

1.  **Train the model** using `train.py`.
2.  **Evaluate the trained model** on the test set (`OAI` or `MOST`) using `evaluate.py`.

### Configurations

All experiments are managed through YAML configuration files located in the `configs` directory within each experiment's folder (e.g., `Radiograph/Baseline/configs/`).

### 1\. Training

Navigate to the desired experiment directory and run the `train.py` script with the appropriate config file. The script will save the best-performing model based on validation performance during the training run.

**Sample Training Command:**

```bash
cd Radiograph/Baseline/
python3 train.py --config ./configs/config_1yr_1.yaml
```

### 2\. Final Evaluation

Use the `evaluate.py` script to get the final performance of the best model saved from training on the unseen test data.

#### Evaluating on the OAI Test Set

```bash
cd Radiograph/Baseline/
python3 evaluate.py --config ./configs/config_4yr_1.yaml --dataset OAI --metric auc --mode test --cv 6
```

#### Evaluating on the MOST Test Set (External Validation)

```bash
cd Radiograph/Baseline/
python3 evaluate.py --config ./configs/config_4yr_1.yaml --dataset MOST --metric auc --mode test --cv 6
```

## Citation

If you use this code in your research, please cite the following paper:

```
@article{Rajamohan2025,
  title={A Progressive risk formulation for enhanced deep learning based total knee replacement prediction in knee osteoarthritis},
  author={Rajamohan, Haresh Rengaraj and et al.},
  journal={Signal, Image and Video Processing},
  year={2025},
  publisher={Springer}
}
```

## License

This project is licensed under the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public license agreement. See `LICENSE` for more details.