# Agent Smith for Pdm 

A comprehensive desktop application for training machine learning models and monitoring their predictions in real-time, built with PyQt5 and PyTorch.

## Features

- **Data Management**
  - CSV file loading with preview
  - Column operations (drop columns)
  - Feature engineering using custom Python expressions
  - Data preprocessing (normalization, label encoding)

- **Model Training**
  - Transformer-based neural network architecture
  - Configurable hyperparameters (layers, heads, learning rate)
  - Real-time training progress visualization
  - Loss curves and accuracy metrics
  - Validation confusion matrix

- **Model Analysis**
  - Multiple visualization types:
    - Confusion Matrix
    - Feature Importance
    - T-SNE/PCA dimensionality reduction
    - ROC Curves
  - Interactive visualization controls

- **Real-time Monitoring**
  - Synthetic data generation for testing
  - Live prediction tracking
  - Confidence level monitoring
  - Feature value visualization
  - Prediction distribution tracking
  - Failure probability analysis
  - Adjustable monitoring interval

## Installation

1. **Prerequisites**
   - Python 3.7+
   - pip package manager

2. **Install dependencies**
```bash
pip install pyqt5 pandas numpy matplotlib seaborn scikit-learn torch
```

3. **Clone repository**
```bash
git clone https://github.com/yourusername/ml-monitoring-dashboard.git
cd ml-monitoring-dashboard
```

## Usage

1. **Launch application**
```bash
python ml_monitoring_app.py
```

2. **Workflow**
   1. *Data Tab*
      - Load your CSV dataset
      - Preprocess data (handle categorical features)
      - Select target variable
      - Create custom features if needed
   
   2. *Training Tab*
      - Configure model parameters
      - Start/stop training
      - Monitor real-time metrics
   
   3. *Visualization Tab*
      - Select visualization type
      - Analyze model performance
   
   4. *Monitoring Tab*
      - Start real-time simulation
      - Observe model behavior
      - Analyze prediction patterns

## Technical Details

- **Core Technologies**
  - PyQt5 for GUI
  - PyTorch for deep learning
  - scikit-learn for metrics/visualization
  - Matplotlib/Seaborn for plotting

- **Key Components**
  - Transformer-based architecture
  - Threaded training execution
  - Real-time data simulator
  - Dynamic visualization updates
  - Queue-based monitoring system

- **Architecture**
  - Model-View-Controller pattern
  - Asynchronous worker threads
  - Custom pandas table models
  - Interactive visualization widgets

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Author: Marwan Ashref 
Email: marwanashref861@gmail.com
GitHub: [@marwan149](https://github.com/marwan149)

---

**Note:** The current implementation uses synthetic data generation for monitoring. To use with real-time data streams, modify the `DataSimulator` class to connect to your actual data source.
