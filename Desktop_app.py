import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import seaborn as sns
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QTabWidget, QPushButton, QLabel, QComboBox, QFileDialog, 
                            QSplitter, QTableView, QProgressBar, QGroupBox, QGridLayout,
                            QCheckBox, QSpinBox, QDoubleSpinBox, QSlider, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QAbstractTableModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import time
import threading
import queue
import random
from datetime import datetime

# Define custom model for table data
class PandasModel(QAbstractTableModel):
    def __init__(self, data):
        super(PandasModel, self).__init__()
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Vertical:
                return str(self._data.index[section])
        return None

# Define Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=4, num_layers=2, hidden_dim=128):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x).squeeze(1)
        return self.fc(x)

# Define worker thread for model training to prevent UI freezing
class TrainingWorker(QThread):
    update_progress = pyqtSignal(int)
    update_loss = pyqtSignal(dict)
    update_metrics = pyqtSignal(dict)
    finished_signal = pyqtSignal(dict)

    def __init__(self, model, train_loader, val_loader, optimizer, criterion, epochs):
        super(TrainingWorker, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def run(self):
        train_losses = []
        val_losses = []
        batch_losses = []
        accuracies = []
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            batch_idx = 0
            total_batches = len(self.train_loader)
            
            for batch in self.train_loader:
                X_batch, y_batch = batch
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                
                # Update batch progress and loss
                progress = int(100 * (epoch * total_batches + batch_idx) / (self.epochs * total_batches))
                self.update_progress.emit(progress)
                
                batch_losses.append(loss.item())
                self.update_loss.emit({"epoch": epoch, "batch": batch_idx, "loss": loss.item()})
                
                batch_idx += 1
                
                # Simulate real-time processing delay
                time.sleep(0.01)  # Small delay to simulate actual processing time

            epoch_train_loss = train_loss/len(self.train_loader)
            train_losses.append(epoch_train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0
            all_outputs = []
            all_labels = []
            
            with torch.no_grad():
                for batch in self.val_loader:
                    X_batch, y_batch = batch
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    val_loss += loss.item()
                    
                    all_outputs.append(outputs.cpu().numpy())
                    all_labels.append(y_batch.cpu().numpy())

            epoch_val_loss = val_loss/len(self.val_loader)
            val_losses.append(epoch_val_loss)
            
            # Calculate metrics
            all_outputs = np.vstack(all_outputs)
            all_labels = np.hstack(all_labels)
            y_pred_probs = np.exp(all_outputs - np.max(all_outputs, axis=1, keepdims=True))  # Numerical stability
            y_pred_probs /= np.sum(y_pred_probs, axis=1, keepdims=True)
            y_pred = np.argmax(y_pred_probs, axis=1)
            
            accuracy = accuracy_score(all_labels, y_pred)
            accuracies.append(accuracy)
            
            metrics = {
                "epoch": epoch,
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
                "accuracy": accuracy,
                "y_pred": y_pred,
                "y_pred_probs": y_pred_probs,
                "all_labels": all_labels
            }
            
            self.update_metrics.emit(metrics)
        
        final_results = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "batch_losses": batch_losses,
            "accuracies": accuracies,
            "y_pred": y_pred,
            "y_pred_probs": y_pred_probs,
            "all_labels": all_labels
        }
        
        self.finished_signal.emit(final_results)

# Real-time data simulator for monitoring
class DataSimulator(QThread):
    new_data = pyqtSignal(dict)
    
    def __init__(self, model, feature_names, class_names, interval=1000):
        super(DataSimulator, self).__init__()
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.interval = interval  # milliseconds
        self.running = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def run(self):
        while self.running:
            # Generate synthetic data point
            num_features = len(self.feature_names)
            synthetic_data = torch.randn(1, num_features)
            
            # Get prediction
            self.model.eval()
            with torch.no_grad():
                synthetic_data = synthetic_data.to(self.device)
                output = self.model(synthetic_data)
                probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
                prediction = np.argmax(probs)
                
            # Add some randomness to prediction confidence
            confidence_noise = np.random.normal(0, 0.05)
            confidence = probs[prediction] + confidence_noise
            confidence = max(0, min(1, confidence))
            
            # Create data record
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            data_record = {
                "timestamp": timestamp,
                "features": synthetic_data.cpu().numpy()[0],
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": probs,
                "predicted_class": self.class_names[prediction]
            }
            
            self.new_data.emit(data_record)
            
            # Wait for next interval
            time.sleep(self.interval / 1000)
    
    def stop(self):
        self.running = False

# Main application window
class MLMonitoringApp(QMainWindow):
    def __init__(self):
        super(MLMonitoringApp, self).__init__()
        
        self.setWindowTitle("ML Model Real-time Monitoring Dashboard")
        self.setGeometry(100, 100, 1800, 900)
        
        # Initialize variables
        self.df = None
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.feature_names = []
        self.class_names = []
        self.label_encoders = {}
        self.training_worker = None
        self.data_simulator = None
        self.monitoring_data = []
        self.monitoring_queue = queue.Queue(maxsize=100)  # Store latest points
        
        # Set up the main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Create tabs
        self.setup_data_tab()
        self.setup_training_tab()
        self.setup_visualization_tab()
        self.setup_monitoring_tab()
        
        # Create status bar
        self.statusBar().showMessage("Ready")
        
        # Setup timer for refreshing monitoring data
        self.monitoring_timer = QTimer()
        self.monitoring_timer.timeout.connect(self.update_monitoring_charts)
        self.monitoring_timer.start(1000)  # Update every second
        
    def setup_data_tab(self):
        data_tab = QWidget()
        layout = QVBoxLayout(data_tab)
        
        # Data loading section
        data_group = QGroupBox("Data Loading and Preprocessing")
        data_layout = QVBoxLayout()
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_path_label = QLabel("No file selected")
        load_button = QPushButton("Load CSV Data")
        load_button.clicked.connect(self.load_data)
        file_layout.addWidget(load_button)
        file_layout.addWidget(self.file_path_label)
        data_layout.addLayout(file_layout)
        
        # Data preview
        self.data_preview = QTableView()
        data_layout.addWidget(QLabel("Data Preview:"))
        data_layout.addWidget(self.data_preview)
        
        # Column dropping section
        drop_columns_layout = QHBoxLayout()
        self.drop_columns_combo = QComboBox()
        drop_columns_button = QPushButton("Drop Selected Column")
        drop_columns_button.clicked.connect(self.drop_selected_column)
        drop_columns_layout.addWidget(QLabel("Drop Column:"))
        drop_columns_layout.addWidget(self.drop_columns_combo)
        drop_columns_layout.addWidget(drop_columns_button)
        data_layout.addLayout(drop_columns_layout)

        # Feature creation section
        feature_creation_layout = QVBoxLayout()
        self.feature_code_input = QComboBox()
        self.feature_code_input.setEditable(True)
        self.feature_code_input.setPlaceholderText("Enter Python code to create a new feature (e.g., df['new_feature'] = df['col1'] + df['col2'])")
        create_feature_button = QPushButton("Create Feature")
        create_feature_button.clicked.connect(self.create_feature)
        feature_creation_layout.addWidget(QLabel("Create New Feature:"))
        feature_creation_layout.addWidget(self.feature_code_input)
        feature_creation_layout.addWidget(create_feature_button)
        data_layout.addLayout(feature_creation_layout)

        # Preprocessing options
        preprocess_layout = QHBoxLayout()
        preprocess_button = QPushButton("Preprocess Data")
        preprocess_button.clicked.connect(self.preprocess_data)
        self.target_combo = QComboBox()
        preprocess_layout.addWidget(QLabel("Target Column:"))
        preprocess_layout.addWidget(self.target_combo)
        preprocess_layout.addWidget(preprocess_button)
        data_layout.addLayout(preprocess_layout)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        self.tabs.addTab(data_tab, "Data")
        
    def setup_training_tab(self):
        training_tab = QWidget()
        layout = QVBoxLayout(training_tab)
        
        # Model configuration
        model_group = QGroupBox("Model Configuration")
        model_layout = QGridLayout()
        
        # Hyperparameters
        model_layout.addWidget(QLabel("Hidden Dimension:"), 0, 0)
        self.hidden_dim_spin = QSpinBox()
        self.hidden_dim_spin.setRange(32, 512)
        self.hidden_dim_spin.setSingleStep(32)
        self.hidden_dim_spin.setValue(128)
        model_layout.addWidget(self.hidden_dim_spin, 0, 1)
        
        model_layout.addWidget(QLabel("Number of Heads:"), 0, 2)
        self.num_heads_spin = QSpinBox()
        self.num_heads_spin.setRange(1, 16)
        self.num_heads_spin.setValue(4)
        model_layout.addWidget(self.num_heads_spin, 0, 3)
        
        model_layout.addWidget(QLabel("Number of Layers:"), 1, 0)
        self.num_layers_spin = QSpinBox()
        self.num_layers_spin.setRange(1, 8)
        self.num_layers_spin.setValue(2)
        model_layout.addWidget(self.num_layers_spin, 1, 1)
        
        model_layout.addWidget(QLabel("Learning Rate:"), 1, 2)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(0.001)
        model_layout.addWidget(self.lr_spin, 1, 3)
        
        model_layout.addWidget(QLabel("Batch Size:"), 2, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(16, 1024)
        self.batch_size_spin.setSingleStep(16)
        self.batch_size_spin.setValue(256)
        model_layout.addWidget(self.batch_size_spin, 2, 1)
        
        model_layout.addWidget(QLabel("Epochs:"), 2, 2)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(5)
        model_layout.addWidget(self.epochs_spin, 2, 3)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Training control
        train_group = QGroupBox("Training")
        train_layout = QVBoxLayout()
        
        # Progress bar
        self.train_progress = QProgressBar()
        train_layout.addWidget(self.train_progress)
        
        # Training controls
        control_layout = QHBoxLayout()
        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.start_training)
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.train_button)
        control_layout.addWidget(self.stop_button)
        train_layout.addLayout(control_layout)
        
        # Real-time training charts
        self.train_fig = Figure(figsize=(12, 8))
        self.train_canvas = FigureCanvas(self.train_fig)
        train_layout.addWidget(self.train_canvas)
        
        train_group.setLayout(train_layout)
        layout.addWidget(train_group)
        
        self.tabs.addTab(training_tab, "Training")
        
    def setup_visualization_tab(self):
        viz_tab = QWidget()
        layout = QVBoxLayout(viz_tab)
        
        # Visualization selector
        viz_selector = QHBoxLayout()
        viz_selector.addWidget(QLabel("Visualization Type:"))
        self.viz_combo = QComboBox()
        self.viz_combo.addItems([
            "Confusion Matrix", 
            "Feature Importance", 
            "T-SNE Visualization",
            "ROC Curves",
            "PCA Visualization"
        ])
        self.viz_combo.currentIndexChanged.connect(self.update_visualization)
        viz_selector.addWidget(self.viz_combo)
        
        self.viz_button = QPushButton("Generate Visualization")
        self.viz_button.clicked.connect(self.update_visualization)
        viz_selector.addWidget(self.viz_button)
        layout.addLayout(viz_selector)
        
        # Visualization figure
        self.viz_fig = Figure(figsize=(10, 8))
        self.viz_canvas = FigureCanvas(self.viz_fig)
        layout.addWidget(self.viz_canvas)
        
        self.tabs.addTab(viz_tab, "Visualization")
        
    def setup_monitoring_tab(self):
        monitoring_tab = QWidget()
        layout = QVBoxLayout(monitoring_tab)
        
        # Monitoring control panel
        control_panel = QHBoxLayout()
        
        # Start/Stop Monitoring
        self.start_monitoring_btn = QPushButton("Start Real-time Monitoring")
        self.start_monitoring_btn.clicked.connect(self.start_monitoring)
        self.stop_monitoring_btn = QPushButton("Stop Monitoring")
        self.stop_monitoring_btn.clicked.connect(self.stop_monitoring)
        self.stop_monitoring_btn.setEnabled(False)
        
        control_panel.addWidget(self.start_monitoring_btn)
        control_panel.addWidget(self.stop_monitoring_btn)
        
        # Interval slider
        control_panel.addWidget(QLabel("Interval (ms):"))
        self.interval_slider = QSlider(Qt.Horizontal)
        self.interval_slider.setRange(100, 5000)
        self.interval_slider.setValue(1000)
        self.interval_slider.setTickPosition(QSlider.TicksBelow)
        self.interval_slider.setTickInterval(500)
        self.interval_value_label = QLabel("1000")
        self.interval_slider.valueChanged.connect(
            lambda v: self.interval_value_label.setText(str(v))
        )
        control_panel.addWidget(self.interval_slider)
        control_panel.addWidget(self.interval_value_label)
        
        layout.addLayout(control_panel)
        
        # Monitoring charts layout in a grid
        grid_layout = QGridLayout()
        
        # Create figures for monitoring charts
        self.prediction_fig = Figure(figsize=(6, 4))
        self.prediction_canvas = FigureCanvas(self.prediction_fig)
        grid_layout.addWidget(self.prediction_canvas, 0, 0)
        
        self.confidence_fig = Figure(figsize=(6, 4))
        self.confidence_canvas = FigureCanvas(self.confidence_fig)
        grid_layout.addWidget(self.confidence_canvas, 0, 1)
        
        self.feature_fig = Figure(figsize=(6, 4))
        self.feature_canvas = FigureCanvas(self.feature_fig)
        grid_layout.addWidget(self.feature_canvas, 1, 0)
        
        self.distribution_fig = Figure(figsize=(6, 4))
        self.distribution_canvas = FigureCanvas(self.distribution_fig)
        grid_layout.addWidget(self.distribution_canvas, 1, 1)
        
        self.failure_prob_fig = Figure(figsize=(6, 4))  # New figure for failure probabilities
        self.failure_prob_canvas = FigureCanvas(self.failure_prob_fig)
        grid_layout.addWidget(self.failure_prob_canvas, 2, 0, 1, 2)  # Span across two columns
        
        layout.addLayout(grid_layout)
        
        # Recent predictions table
        self.predictions_table = QTableView()
        layout.addWidget(QLabel("Recent Predictions:"))
        layout.addWidget(self.predictions_table)
        
        self.tabs.addTab(monitoring_tab, "Real-time Monitoring")
        
    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.file_path_label.setText(os.path.basename(file_path))
                
                # Display data preview
                model = PandasModel(self.df.head(10))
                self.data_preview.setModel(model)
                
                # Update target column and drop column dropdowns
                self.target_combo.clear()
                self.drop_columns_combo.clear()
                self.target_combo.addItems(self.df.columns)
                self.drop_columns_combo.addItems(self.df.columns)
                
                self.statusBar().showMessage(f"Loaded data with {self.df.shape[0]} rows and {self.df.shape[1]} columns")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading data: {str(e)}")
                
    def preprocess_data(self):
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please load data first")
            return
            
        try:
            target_col = self.target_combo.currentText()
            
            # Encode categorical columns
            def encode_categorical(df):
                label_encoders = {}
                for col in df.select_dtypes(include=["object"]).columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    label_encoders[col] = le
                return df, label_encoders

            self.df, self.label_encoders = encode_categorical(self.df.copy())
            
            # Feature Engineering (example)
            if "mis" in self.df.columns and "cpm" in self.df.columns:
                self.df["mis_cpm_ratio"] = self.df["mis"] / (self.df["cpm"] + 1e-6)
            
            if "omega" in self.df.columns and "sup" in self.df.columns:
                self.df["omega_sup_product"] = self.df["omega"] * self.df["sup"]
                
            # Define features and target
            features = self.df.drop(columns=[target_col, "instance", "number"], errors='ignore')
            target = self.df[target_col]
            
            self.feature_names = features.columns.tolist()
            
            # Get class names
            if target_col in self.label_encoders:
                self.class_names = self.label_encoders[target_col].classes_
            else:
                num_classes = len(target.unique())
                self.class_names = [f"Class {i}" for i in range(num_classes)]
                
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Convert to tensors
            X = torch.tensor(features_scaled, dtype=torch.float32)
            y = torch.tensor(target.values, dtype=torch.long)
            
            # Split dataset
            train_size = int(0.8 * len(self.df))
            val_size = len(self.df) - train_size
            train_dataset, val_dataset = random_split(TensorDataset(X, y), [train_size, val_size])
            
            # Data loaders
            batch_size = self.batch_size_spin.value()
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize model
            input_dim = X.shape[1]
            num_classes = len(target.unique())
            hidden_dim = self.hidden_dim_spin.value()
            num_heads = self.num_heads_spin.value()
            num_layers = self.num_layers_spin.value()
            
            self.model = TransformerModel(
                input_dim, 
                num_classes, 
                num_heads=num_heads, 
                num_layers=num_layers,
                hidden_dim=hidden_dim
            )
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            
            # Update UI to show success
            QMessageBox.information(self, "Success", "Data preprocessing completed successfully")
            self.statusBar().showMessage("Data preprocessed and model initialized")
            
            # Update data preview to show processed data
            model = PandasModel(self.df.head(10))
            self.data_preview.setModel(model)
            
            # Enable training tab
            self.tabs.setTabEnabled(1, True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error preprocessing data: {str(e)}")
            
    def drop_selected_column(self):
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please load data first")
            return
        
        selected_column = self.drop_columns_combo.currentText()
        if selected_column:
            try:
                self.df.drop(columns=[selected_column], inplace=True)
                
                # Update data preview
                model = PandasModel(self.df.head(10))
                self.data_preview.setModel(model)
                
                # Update dropdowns
                self.target_combo.clear()
                self.drop_columns_combo.clear()
                self.target_combo.addItems(self.df.columns)
                self.drop_columns_combo.addItems(self.df.columns)
                
                self.statusBar().showMessage(f"Column '{selected_column}' dropped successfully")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error dropping column: {str(e)}")
                
    def create_feature(self):
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please load data first")
            return

        try:
            code = self.feature_code_input.currentText()
            exec(code, {"df": self.df})
            self.feature_code_input.addItem(code)  # Save the code for reuse

            # Update data preview
            model = PandasModel(self.df.head(10))
            self.data_preview.setModel(model)

            # Update dropdowns
            self.target_combo.clear()
            self.drop_columns_combo.clear()
            self.target_combo.addItems(self.df.columns)
            self.drop_columns_combo.addItems(self.df.columns)

            self.statusBar().showMessage("Feature created successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error creating feature: {str(e)}")

    def start_training(self):
        if self.model is None or self.train_loader is None:
            QMessageBox.warning(self, "Warning", "Please load and preprocess data first")
            return
            
        # Update batch size if changed
        batch_size = self.batch_size_spin.value()
        if self.train_loader.batch_size != batch_size:
            # Recreate data loaders with new batch size
            dataset = self.train_loader.dataset
            self.train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            dataset = self.val_loader.dataset
            self.val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Set up optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr_spin.value())
        criterion = nn.CrossEntropyLoss()
        
        # Clear previous training plots
        self.train_fig.clear()
        
        # Create subplots for training visualization
        gs = self.train_fig.add_gridspec(2, 2)
        self.loss_ax = self.train_fig.add_subplot(gs[0, 0])
        self.batch_loss_ax = self.train_fig.add_subplot(gs[0, 1])
        self.acc_ax = self.train_fig.add_subplot(gs[1, 0])
        self.conf_mat_ax = self.train_fig.add_subplot(gs[1, 1])
        
        self.train_canvas.draw()
        
        # Create and start worker thread
        self.training_worker = TrainingWorker(
            self.model, 
            self.train_loader, 
            self.val_loader, 
            optimizer, 
            criterion, 
            self.epochs_spin.value()
        )
        
        # Connect signals
        self.training_worker.update_progress.connect(self.update_progress)
        self.training_worker.update_loss.connect(self.update_loss_plot)
        self.training_worker.update_metrics.connect(self.update_metrics)
        self.training_worker.finished_signal.connect(self.training_finished)
        
        # Start training
        self.training_worker.start()
        
        # Update UI
        self.train_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.statusBar().showMessage("Training in progress...")
        
    def stop_training(self):
        if self.training_worker and self.training_worker.isRunning():
            self.training_worker.requestInterruption()  # Gracefully request thread interruption
            self.training_worker.wait()  # Wait for the thread to finish
            
            # Update UI
            self.train_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.statusBar().showMessage("Training stopped")
        
    def update_progress(self, value):
        self.train_progress.setValue(value)
        
    def update_loss_plot(self, loss_data):
        epoch = loss_data["epoch"]
        batch = loss_data["batch"]
        loss = loss_data["loss"]
        
        # Store for batch loss plot
        if not hasattr(self, 'batch_losses'):
            self.batch_losses = []
        self.batch_losses.append(loss)
        
        # Update batch loss plot
        self.batch_loss_ax.clear()
        self.batch_loss_ax.plot(self.batch_losses[-100:], 'r-')
        self.batch_loss_ax.set_title("Recent Batch Loss")
        self.batch_loss_ax.set_xlabel("Batch")
        self.batch_loss_ax.set_ylabel("Loss")
        self.train_canvas.draw()
        
    def update_metrics(self, metrics):
        epoch = metrics["epoch"]
        train_loss = metrics["train_loss"]
        val_loss = metrics["val_loss"]
        accuracy = metrics["accuracy"]
        y_pred = metrics["y_pred"]
        all_labels = metrics["all_labels"]
        
        # Store metrics for epoch plots
        if not hasattr(self, 'train_losses'):
            self.train_losses = []
            self.val_losses = []
            self.accuracies = []
            
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.accuracies.append(accuracy)
        
        # Update loss plot
        epochs = range(1, len(self.train_losses) + 1)
        self.loss_ax.clear()
        self.loss_ax.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        self.loss_ax.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        self.loss_ax.set_title("Training and Validation Loss")
        self.loss_ax.set_xlabel("Epochs")
        self.loss_ax.set_ylabel("Loss")
        self.loss_ax.legend()
        
        # Update accuracy plot
        self.acc_ax.clear()
        self.acc_ax.plot(epochs, self.accuracies, 'g-')
        self.acc_ax.set_title("Validation Accuracy")
        self.acc_ax.set_xlabel("Epochs")
        self.acc_ax.set_ylabel("Accuracy")
        
        # Update confusion matrix plot
        self.conf_mat_ax.clear()
        cm = confusion_matrix(all_labels, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=self.conf_mat_ax)
        self.conf_mat_ax.set_title("Confusion Matrix")
        self.conf_mat_ax.set_xlabel("Predicted Label")
        self.conf_mat_ax.set_ylabel("True Label")
        
        self.train_canvas.draw()
        
    def training_finished(self, results):
        # Update UI
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.statusBar().showMessage("Training completed")
        
        # Store final results
        self.training_results = results
        
        # Enable visualization and monitoring tabs
        self.tabs.setTabEnabled(2, True)
        self.tabs.setTabEnabled(3, True)
        
        # Show completion message
        QMessageBox.information(self, "Training Complete", 
                               f"Training completed with final accuracy: {results['accuracies'][-1]:.4f}")
    
    def update_visualization(self):
        if not hasattr(self, 'training_results'):
            QMessageBox.warning(self, "Warning", "Please complete training first")
            return
            
        viz_type = self.viz_combo.currentText()
        
        # Clear previous visualization
        self.viz_fig.clear()
        ax = self.viz_fig.add_subplot(111)
        
        if viz_type == "Confusion Matrix":
            # Get confusion matrix
            cm = confusion_matrix(self.training_results['all_labels'], self.training_results['y_pred'])
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            
            # Add class names if available
            if len(self.class_names) > 0:
                ax.set_xticklabels(self.class_names)
                ax.set_yticklabels(self.class_names)
                
        elif viz_type == "Feature Importance":
            # For transformer models, we use gradient-based feature importance
            # This is a placeholder - would need actual gradients for real implementation
            importance = np.random.rand(len(self.feature_names))
            
            # Create sorted importance plot
            sorted_idx = np.argsort(importance)
            sorted_features = [self.feature_names[i] for i in sorted_idx]
            
            ax.barh(range(len(sorted_features)), importance[sorted_idx])
            ax.set_yticks(range(len(sorted_features)))
            ax.set_yticklabels(sorted_features)
            ax.set_title("Feature Importance")
            ax.set_xlabel("Importance")
            
        elif viz_type == "T-SNE Visualization":
            # Get validation data
            all_data = []
            all_labels = []
            
            for batch in self.val_loader:
                X_batch, y_batch = batch
                all_data.append(X_batch.numpy())
                all_labels.append(y_batch.numpy())
                
            X_val = np.vstack(all_data)
            y_val = np.hstack(all_labels)
            
            # Apply T-SNE
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(X_val)
            
            # Plot T-SNE results
            num_classes = len(np.unique(y_val))
            colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
            
            for i in range(num_classes):
                ax.scatter(X_tsne[y_val == i, 0], X_tsne[y_val == i, 1], 
                          c=[colors[i]], label=self.class_names[i] if i < len(self.class_names) else f"Class {i}")
            
            ax.set_title("T-SNE Visualization")
            ax.legend()
            
        elif viz_type == "ROC Curves":
            # Get prediction probabilities
            y_pred_probs = self.training_results['y_pred_probs']
            y_true = self.training_results['all_labels']
            
            # Plot ROC curve for each class
            from sklearn.metrics import roc_curve, auc
            
            num_classes = y_pred_probs.shape[1]
            
            for i in range(num_classes):
                fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_pred_probs[:, i])
                roc_auc = auc(fpr, tpr)
                
                class_name = self.class_names[i] if i < len(self.class_names) else f"Class {i}"
                ax.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
                
            # Add diagonal line
            ax.plot([0, 1], [0, 1], 'k--', lw=2)
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves')
            ax.legend(loc="lower right")
            
        elif viz_type == "PCA Visualization":
            # Get validation data
            all_data = []
            all_labels = []
            
            for batch in self.val_loader:
                X_batch, y_batch = batch
                all_data.append(X_batch.numpy())
                all_labels.append(y_batch.numpy())
                
            X_val = np.vstack(all_data)
            y_val = np.hstack(all_labels)
            
            # Apply PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_val)
            
            # Plot PCA results
            num_classes = len(np.unique(y_val))
            colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
            
            for i in range(num_classes):
                ax.scatter(X_pca[y_val == i, 0], X_pca[y_val == i, 1], 
                          c=[colors[i]], label=self.class_names[i] if i < len(self.class_names) else f"Class {i}")
            
            # Add explained variance
            var_explained = pca.explained_variance_ratio_
            ax.set_xlabel(f'PC1 ({var_explained[0]:.2%} variance)')
            ax.set_ylabel(f'PC2 ({var_explained[1]:.2%} variance)')
            ax.set_title("PCA Visualization")
            ax.legend()
            
        # Update the canvas
        self.viz_canvas.draw()
        
    def start_monitoring(self):
        if not hasattr(self, 'model') or self.model is None:
            QMessageBox.warning(self, "Warning", "Please train a model first")
            return
            
        # Initialize monitoring data storage
        self.monitoring_data = []
        
        # Set up data simulator
        interval = self.interval_slider.value()
        self.data_simulator = DataSimulator(
            self.model, 
            self.feature_names, 
            self.class_names,
            interval=interval
        )
        
        # Connect new data signal
        self.data_simulator.new_data.connect(self.process_new_data)
        
        # Start simulator
        self.data_simulator.start()
        
        # Set up monitoring charts
        self.setup_monitoring_charts()
        
        # Update UI
        self.start_monitoring_btn.setEnabled(False)
        self.stop_monitoring_btn.setEnabled(True)
        self.statusBar().showMessage("Real-time monitoring started")
        
    def stop_monitoring(self):
        if self.data_simulator is not None:
            self.data_simulator.stop()
            self.data_simulator.wait()
            
        # Update UI
        self.start_monitoring_btn.setEnabled(True)
        self.stop_monitoring_btn.setEnabled(False)
        self.statusBar().showMessage("Monitoring stopped")
        
    def setup_monitoring_charts(self):
        # Clear previous charts
        self.prediction_fig.clear()
        self.confidence_fig.clear()
        self.feature_fig.clear()
        self.distribution_fig.clear()
        
        # Create new axes
        self.prediction_ax = self.prediction_fig.add_subplot(111)
        self.confidence_ax = self.confidence_fig.add_subplot(111)
        self.feature_ax = self.feature_fig.add_subplot(111)
        self.distribution_ax = self.distribution_fig.add_subplot(111)
        
        # Set up initial plots
        # Prediction timeline
        self.prediction_line, = self.prediction_ax.plot([], [], 'b-')
        self.prediction_ax.set_title("Prediction Timeline")
        self.prediction_ax.set_xlabel("Time")
        self.prediction_ax.set_ylabel("Predicted Class")
        self.prediction_ax.set_ylim(-0.5, len(self.class_names) - 0.5)
        self.prediction_ax.set_yticks(range(len(self.class_names)))
        self.prediction_ax.set_yticklabels(self.class_names)
        
        # Confidence timeline
        self.confidence_line, = self.confidence_ax.plot([], [], 'r-')
        self.confidence_ax.set_title("Prediction Confidence")
        self.confidence_ax.set_xlabel("Time")
        self.confidence_ax.set_ylabel("Confidence")
        self.confidence_ax.set_ylim(0, 1)
        
        # Feature values
        self.feature_ax.set_title("Latest Feature Values")
        self.feature_ax.set_xlabel("Value")
        self.feature_ax.set_ylabel("Feature")
        
        # Class distribution
        self.distribution_ax.set_title("Prediction Distribution")
        self.distribution_ax.set_xlabel("Class")
        self.distribution_ax.set_ylabel("Count")
        
        # Draw initial canvases
        self.prediction_canvas.draw()
        self.confidence_canvas.draw()
        self.feature_canvas.draw()
        self.distribution_canvas.draw()
        
    def process_new_data(self, data):
        # Store in monitoring data
        self.monitoring_data.append(data)
        
        # Keep only the last 100 points
        if len(self.monitoring_data) > 100:
            self.monitoring_data = self.monitoring_data[-100:]
            
        # Update predictions table
        predictions_df = pd.DataFrame([
            {"Time": d["timestamp"], 
             "Prediction": d["predicted_class"], 
             "Confidence": f"{d['confidence']:.4f}"}
            for d in self.monitoring_data[-10:]  # Show only last 10
        ])
        
        table_model = PandasModel(predictions_df)
        self.predictions_table.setModel(table_model)
        
    def update_monitoring_charts(self):
        if not hasattr(self, 'monitoring_data') or len(self.monitoring_data) == 0:
            return
            
        # Timestamps for x-axis
        timestamps = [d["timestamp"] for d in self.monitoring_data]
        
        # Update prediction timeline
        predictions = [d["prediction"] for d in self.monitoring_data]
        self.prediction_ax.clear()
        self.prediction_ax.plot(range(len(timestamps)), predictions, 'bo-')
        self.prediction_ax.set_title("Prediction Timeline")
        self.prediction_ax.set_xlabel("Sample")
        self.prediction_ax.set_ylabel("Predicted Class")
        
        if len(self.class_names) > 0:
            self.prediction_ax.set_yticks(range(len(self.class_names)))
            self.prediction_ax.set_yticklabels(self.class_names)
        
        # Update confidence timeline
        confidences = [d["confidence"] for d in self.monitoring_data]
        self.confidence_ax.clear()
        self.confidence_ax.plot(range(len(timestamps)), confidences, 'ro-')
        self.confidence_ax.set_title("Prediction Confidence")
        self.confidence_ax.set_xlabel("Sample")
        self.confidence_ax.set_ylabel("Confidence")
        self.confidence_ax.set_ylim(0, 1)
        
        # Update feature values (for latest data point)
        latest_features = self.monitoring_data[-1]["features"]
        self.feature_ax.clear()
        
        # Select top features to show
        num_features_to_show = min(10, len(self.feature_names))
        indices = range(num_features_to_show)
        
        y_pos = np.arange(num_features_to_show)
        self.feature_ax.barh(y_pos, latest_features[:num_features_to_show])
        self.feature_ax.set_title("Latest Feature Values (Top 10)")
        self.feature_ax.set_xlabel("Value")
        self.feature_ax.set_yticks(y_pos)
        self.feature_ax.set_yticklabels([self.feature_names[i] for i in indices])
        
        # Update class distribution
        all_predictions = [d["prediction"] for d in self.monitoring_data]
        unique_classes, counts = np.unique(all_predictions, return_counts=True)
        
        self.distribution_ax.clear()
        bars = self.distribution_ax.bar(unique_classes, counts)
        
        # Add class labels if available
        if len(self.class_names) > 0:
            self.distribution_ax.set_xticks(unique_classes)
            labels = [self.class_names[i] if i < len(self.class_names) else f"Class {i}" 
                    for i in unique_classes]
            self.distribution_ax.set_xticklabels(labels, rotation=45, ha="right")
        
        self.distribution_ax.set_title("Prediction Distribution")
        self.distribution_ax.set_xlabel("Class")
        self.distribution_ax.set_ylabel("Count")
        
        # Update failure probabilities
        self.failure_prob_ax = self.failure_prob_fig.add_subplot(111)
        self.failure_prob_ax.clear()
        
        # Calculate average failure probabilities for each class
        probabilities = np.array([d["probabilities"] for d in self.monitoring_data])
        avg_failure_probs = 1 - probabilities.mean(axis=0)
        
        self.failure_prob_ax.bar(range(len(avg_failure_probs)), avg_failure_probs, color='orange')
        self.failure_prob_ax.set_title("Average Failure Probability by Class")
        self.failure_prob_ax.set_xlabel("Class")
        self.failure_prob_ax.set_ylabel("Failure Probability")
        
        if len(self.class_names) > 0:
            self.failure_prob_ax.set_xticks(range(len(self.class_names)))
            self.failure_prob_ax.set_xticklabels(self.class_names, rotation=45, ha="right")
        
        # Draw updated canvases
        self.prediction_canvas.draw()
        self.confidence_canvas.draw()
        self.feature_canvas.draw()
        self.distribution_canvas.draw()
        self.failure_prob_canvas.draw()
        
        
# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MLMonitoringApp()
    window.show()
    sys.exit(app.exec_())