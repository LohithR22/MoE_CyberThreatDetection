# Mixture of Experts (MoE) for Cybersecurity Detection

This project implements a Mixture of Experts (MoE) model using a Switch Transformer architecture for detecting cybersecurity threats. The model is designed to be fine-tuned on datasets located in the `DatasetsForCyberThreat` folder.

## Project Structure

```
moe-cybersecurity-detection
├── src
│   ├── main.py          # Entry point for the application
│   ├── model.py         # Defines the MoE model architecture
│   ├── train.py         # Contains the training loop and evaluation
│   ├── utils.py         # Utility functions for data processing and logging
│   └── datasets
│       └── __init__.py  # Initializes the datasets package
├── DatasetsForCyberThreat
│   └── (your dataset files here)  # Dataset files for training
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/LohithR22/MoE_CyberThreatDetection
   cd moe-cybersecurity-detection
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Place your dataset files in the `DatasetsForCyberThreat` folder.

## Usage

To train the MoE model, run the following command:
```
python src/main.py
```

## Model Architecture

The MoE model is built using a Switch Transformer architecture, which allows for efficient handling of large models by activating only a subset of experts during training and inference.

## Datasets

The datasets used for training and evaluation can be found in the `DatasetsForCyberThreat` folder. Ensure that the data is preprocessed according to the specifications in `src/utils.py`.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
