# A Progressive Risk Formulation for Enhanced Deep Learning-Based Total Knee Replacement Prediction in Knee Osteoarthritis

This repository contains the official PyTorch implementation for the paper: **A Progressive risk formulation for enhanced deep learning based total knee replacement prediction in knee osteoarthritis** ([Rajamohan et al., 2025](https://link.springer.com/article/10.1007/s11760-025-04404-0)).

## Description

This project provides a deep learning framework to predict the likelihood of a patient requiring Total Knee Replacement (TKR) surgery. The models are designed to work with both **knee radiographs** and **MRI scans** and can predict the risk of TKR over multiple time horizons: **1, 2, and 4 years**.

The core contribution of this work is a novel **"progressive risk formulation"**. This approach incorporates the clinical understanding that osteoarthritis is a degenerative disease, meaning the risk of TKR for a patient should either increase or remain stable over time. By enforcing this constraint during training, the models learn a more robust representation of disease progression, leading to improved prediction accuracy compared to standard baseline models.

The code is organized to facilitate the training and evaluation of various experimental setups described in the paper, including:

  * `Baseline`: A standard deep learning model without the progressive risk constraint.
  * `RiskFORM1`, `RiskFORM2`, `ConReg`, `RiskReg`: Different implementations of the progressive risk formulation and related techniques.

The models in this study were trained and evaluated on data from the Osteoarthritis Initiative (OAI) and the Multicenter Osteoarthritis Study (MOST) datasets.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd TKRRisk-Formulation
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment. The required packages can be installed using the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

## Data (`csv_files`)

The `csv_files` directory contains the necessary data splits for running the experiments. It is organized by imaging modality (`Radiographs` and `MRI`) and then by cross-validation folds.

  * **Fold Structure**: The data is split into multiple folds for cross-validation (e.g., `Fold_1`, `Fold_2`, etc.). Each fold directory contains:

      * `CV_1_train.csv`, `CV_1_val.csv`, etc.: Training and validation splits for each cross-validation run within a fold.
      * `Fold_1_test.csv`: The test set for that specific fold.

  * **CSV File Content**: Each CSV file contains metadata for the image dataset, including:

      * `ID`: A unique identifier for the patient.
      * `Label`: The ground truth label for TKR.
      * `h5Name`: The filename of the corresponding image data, which is expected to be in HDF5 format.
      * Time-based labels (e.g., `1yrLabel1`, `2yrLabel1`, `4yrLabel1`): These columns provide the specific labels for the different prediction time horizons.

## Training and Evaluation

### Configurations

All experiments are managed through YAML configuration files located in the `configs` directory within each experiment's folder (e.g., `Radiograph/Baseline/configs/`). These files allow for easy modification of hyperparameters and settings for different experimental runs.

Key parameters in the `config.py` and `*.yaml` files that you can modify include:

  * **`year`**: Sets the prediction time horizon (e.g., `1yr`, `2yr`, `4yr`).
  * **`tl_model`**: Specifies the backbone model for feature extraction (e.g., `Resnet34`).
  * **`learning_rate`, `batch_size`, `num_epoch`**: Standard training hyperparameters.
  * **`gamma`, `margin`**: Hyperparameters specific to the progressive risk formulation loss functions.
  * **`fold`**: Specifies which cross-validation fold to use for training and evaluation.

### Training

To train a model, navigate to the desired experiment directory and run the `train.py` script with the appropriate config file.

**Sample Training Command:**

```bash
# Navigate to the directory of the desired model and modality
cd Radiograph/Baseline/

# Run the training script, specifying the config file
python3 train.py --config ./configs/config_1yr_1.yaml
```

### Evaluation

To evaluate a trained model, use the `evaluate.py` script. You will need to specify the config file used for training, the dataset, the metric, the data split, and the cross-validation fold number.

**Sample Evaluation Command:**

```bash
# Navigate to the directory of the model to evaluate
cd Radiograph/Baseline/

# Run the evaluation script with the desired parameters
python3 evaluate.py --config ./configs/config_4yr_1.yaml --dataset MOST --metric auc --mode test --cv 6
```

This command will evaluate the model trained with `config_4yr_1.yaml` on the 6th cross-validation fold of the MOST test set, reporting the AUC score.

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

This project is licensed under the terms of the license agreement. See `LICENSE` for more details.