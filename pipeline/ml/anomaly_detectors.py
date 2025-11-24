"""
Anomaly Detection Ensemble
=========================

Ensemble of unsupervised anomaly detection methods:
- Isolation Forest
- HDBSCAN clustering
- Variational Autoencoder reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    warnings.warn("HDBSCAN not available. Install with: pip install hdbscan")


class IsolationForestDetector:
    """
    Isolation Forest anomaly detection on latent space.
    """

    def __init__(self, contamination: float = 0.1,
                 n_estimators: int = 100,
                 random_state: int = 42):
        """
        Initialize Isolation Forest detector.

        Parameters:
            contamination: Expected proportion of anomalies
            n_estimators: Number of trees
            random_state: Random seed
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray):
        """
        Fit the isolation forest.

        Parameters:
            X: Training data (n_samples, n_features)
        """
        # Scale data
        X_scaled = self.scaler.fit_transform(X)

        # Fit model
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        self.model.fit(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores.

        Parameters:
            X: Input data

        Returns:
            np.ndarray: Anomaly scores (higher = more anomalous)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)

        # Isolation Forest returns -1 for anomalies, 1 for normal
        # Convert to anomaly scores (0-1 range)
        predictions = self.model.predict(X_scaled)

        # Convert to anomaly scores: -1 -> 1 (anomaly), 1 -> 0 (normal)
        anomaly_scores = (predictions == -1).astype(float)

        # Also use decision function for more nuanced scores
        decision_scores = self.model.decision_function(X_scaled)
        # decision_function returns negative values for anomalies
        # Convert to 0-1 scale
        anomaly_scores = 1 / (1 + np.exp(decision_scores))  # Sigmoid

        return anomaly_scores

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available."""
        return None  # Isolation Forest doesn't provide feature importance


class HDBSCANDetector:
    """
    HDBSCAN-based anomaly detection using density clustering.
    """

    def __init__(self, min_cluster_size: int = 5,
                 min_samples: Optional[int] = None,
                 cluster_selection_epsilon: float = 0.0):
        """
        Initialize HDBSCAN detector.

        Parameters:
            min_cluster_size: Minimum cluster size
            min_samples: Number of samples in neighborhood for core point
            cluster_selection_epsilon: Distance threshold for cluster selection
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN not available. Install with: pip install hdbscan")

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray):
        """
        Fit HDBSCAN clustering.

        Parameters:
            X: Training data
        """
        X_scaled = self.scaler.fit_transform(X)

        self.model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            prediction_data=True  # Enable prediction on new data
        )
        self.model.fit(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores based on cluster membership.

        Parameters:
            X: Input data

        Returns:
            np.ndarray: Anomaly scores (higher = more anomalous)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)

        # Get cluster labels and probabilities
        labels, probabilities = hdbscan.approximate_predict(self.model, X_scaled)

        # Convert to anomaly scores
        # Points with label -1 are noise/anomalies
        # Use 1 - probability for anomalies in clusters
        anomaly_scores = np.zeros(len(X))

        # Noise points get high anomaly score
        noise_mask = labels == -1
        anomaly_scores[noise_mask] = 1.0

        # Points in clusters get anomaly score based on low probability
        cluster_mask = labels != -1
        anomaly_scores[cluster_mask] = 1.0 - probabilities[cluster_mask]

        return anomaly_scores


class VAEDetector(nn.Module):
    """
    Variational Autoencoder for reconstruction-based anomaly detection.
    """

    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: List[int] = [128, 64]):
        """
        Initialize VAE detector.

        Parameters:
            input_dim: Input feature dimension
            latent_dim: Latent space dimension
            hidden_dims: Hidden layer dimensions for encoder/decoder
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Build encoder
        encoder_layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU()
            ])
            current_dim = hidden_dim

        # Latent space (mean and log variance)
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Build decoder
        decoder_layers = []
        current_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU()
            ])
            current_dim = hidden_dim

        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.

        Parameters:
            x: Input tensor

        Returns:
            tuple: (mu, logvar) for latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            tuple: (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def loss_function(self, reconstruction: torch.Tensor, x: torch.Tensor,
                     mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute VAE loss (reconstruction + KL divergence).

        Parameters:
            reconstruction: Reconstructed input
            x: Original input
            mu: Latent mean
            logvar: Latent log variance

        Returns:
            torch.Tensor: Total loss
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, x, reduction='sum')

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + kl_loss

    def _select_device(self, device: str) -> str:
        """Select the appropriate device."""
        if device == 'auto':
            if torch.backends.mps.is_available():
                return 'mps'
            elif torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return device

    def fit(self, X: np.ndarray, epochs: int = 100, batch_size: int = 32,
            learning_rate: float = 1e-3, device: str = 'auto'):
        """
        Fit VAE on training data.

        Parameters:
            X: Training data
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            device: Device to train on
        """
        device = self._select_device(device)
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        X_tensor = torch.FloatTensor(X).to(device)
        dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()

                reconstruction, mu, logvar = self.forward(batch_x)
                loss = self.loss_function(reconstruction, batch_x, mu, logvar)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    def predict(self, X: np.ndarray, device: str = 'auto') -> np.ndarray:
        """
        Predict anomaly scores based on reconstruction error.

        Parameters:
            X: Input data

        Returns:
            np.ndarray: Anomaly scores (higher = more anomalous)
        """
        device = self._select_device(device)
        self.eval()
        self.to(device)

        X_tensor = torch.FloatTensor(X).to(device)

        with torch.no_grad():
            reconstruction, _, _ = self.forward(X_tensor)
            reconstruction_error = torch.mean((reconstruction - X_tensor)**2, dim=1)

        # Convert to anomaly scores (normalize to 0-1)
        anomaly_scores = reconstruction_error.cpu().numpy()
        anomaly_scores = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores) + 1e-10)

        return anomaly_scores


class EnsembleDetector:
    """
    Ensemble of anomaly detection methods with weighted aggregation.
    """

    def __init__(self, input_dim: int,
                 methods: List[str] = ['isolation_forest', 'hdbscan', 'vae'],
                 weights: Optional[List[float]] = None):
        """
        Initialize ensemble detector.

        Parameters:
            input_dim: Input feature dimension
            methods: List of detection methods to use
            weights: Weights for each method (default: equal)
        """
        self.input_dim = input_dim
        self.methods = methods
        self.weights = weights or [1.0 / len(methods)] * len(methods)

        # Initialize detectors
        self.detectors = {}
        self._initialize_detectors()

    def _initialize_detectors(self):
        """Initialize individual detectors."""
        for method in self.methods:
            if method == 'isolation_forest':
                self.detectors[method] = IsolationForestDetector()
            elif method == 'hdbscan':
                if HDBSCAN_AVAILABLE:
                    self.detectors[method] = HDBSCANDetector()
                else:
                    warnings.warn("HDBSCAN not available, skipping")
            elif method == 'vae':
                self.detectors[method] = VAEDetector(self.input_dim)

    def fit(self, X: np.ndarray, **kwargs):
        """
        Fit all detectors in ensemble.

        Parameters:
            X: Training data
            **kwargs: Additional arguments for specific detectors
        """
        for method, detector in self.detectors.items():
            if method == 'vae':
                # VAE needs special fitting
                detector.fit(X, **kwargs.get('vae_kwargs', {}))
            else:
                detector.fit(X)

    def predict(self, X: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Predict anomalies using ensemble.

        Parameters:
            X: Input data

        Returns:
            dict: Ensemble predictions and individual scores
        """
        individual_scores = {}
        ensemble_score = np.zeros(len(X))

        for i, (method, detector) in enumerate(self.detectors.items()):
            scores = detector.predict(X)
            individual_scores[method] = scores
            ensemble_score += self.weights[i] * scores

        # Normalize ensemble score to 0-1 range
        ensemble_score = np.clip(ensemble_score, 0, 1)

        # Consensus detection (majority vote)
        consensus_threshold = 0.5
        individual_predictions = np.array([scores > consensus_threshold for scores in individual_scores.values()])
        consensus = np.mean(individual_predictions, axis=0) > 0.5

        return {
            'ensemble_scores': ensemble_score,
            'individual_scores': individual_scores,
            'consensus_prediction': consensus,
            'n_methods_agree': np.sum(individual_predictions, axis=0)
        }

    def get_feature_importance(self) -> Dict[str, Any]:
        """
        Get feature importance from individual detectors.

        Returns:
            dict: Feature importance information
        """
        importance_info = {}
        for method, detector in self.detectors.items():
            importance = detector.get_feature_importance()
            if importance is not None:
                importance_info[method] = importance

        return importance_info
