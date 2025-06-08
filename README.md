# AEDFC with FREI Criterion

## Overview

This project implements the Adaptive Entropy-Driven Feature Clustering (AEDFC) algorithm with the Feature Reduction Entropy Index (FREI) stopping criterion. The algorithm is designed for feature selection in the context of DDoS detection, leveraging conditional entropy to assess feature redundancy.

## Features

- Computes pairwise conditional entropy matrix for feature selection.
- Implements the FREI criterion to determine feature redundancy.
- Includes classes for DDoS detection using Random Forest classifiers.
- Provides adversarial robustness testing against various baseline methods.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd AEDFC_FREI
pip install -r requirements.txt
```

## Usage

To use the AEDFC algorithm, you can initialize the `DDoSDetector` class and load your dataset. Here’s a simple example:

```python
from src.AEDFC_FREI import DDoSDetector

detector = DDoSDetector()
X_train, X_test, y_train, y_test = detector.load_data('path_to_your_dataset.csv')
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements

This project builds upon the foundational work in feature selection and entropy-based methods in machine learning. The FREI criterion is inspired by the need for efficient feature selection in high-dimensional datasets, particularly in cybersecurity applications like DDoS detection.
This project is inspired by the work on feature selection for DDoS detection and the use of entropy-based methods in machine learning. Special thanks to the contributors and researchers in this field.

## Contact
