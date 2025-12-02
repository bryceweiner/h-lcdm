"""
Anomaly Detection Ensemble
=========================

Ensemble of unsupervised anomaly detection methods:
- Isolation Forest (with robust scaling)
- HDBSCAN clustering (with robust scaling)
- Variational Autoencoder reconstruction (with non-Gaussian loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import warnings

# Suppress sklearn deprecation warnings for internal API changes
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn.utils.deprecation')
# Suppress HDBSCAN warnings about no clusters
warnings.filterwarnings('ignore', message='Clusterer does not have any defined clusters')

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    warnings.warn("HDBSCAN not available. Install with: pip install hdbscan")


class IsolationForestDetector:
    """
    Isolation Forest anomaly detection on latent space.
    Uses RobustScaler to handle cosmological outliers properly.
    """

    def __init__(self, contamination: float = 0.1,
                 n_estimators: int = 100,
                 random_state: int = 42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        # Use RobustScaler (median/IQR) instead of StandardScaler (mean/std)
        # Mean/std are easily corrupted by the very anomalies we want to find.
        self.scaler = RobustScaler()

    def fit(self, X: np.ndarray):
        """Fit the isolation forest."""
        X_scaled = self.scaler.fit_transform(X)
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        self.model.fit(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        # decision_function returns negative for anomalies
        decision_scores = self.model.decision_function(X_scaled)
        # Convert to 0-1 probability-like score
        anomaly_scores = 1 / (1 + np.exp(decision_scores))
        return anomaly_scores

    def get_feature_importance(self) -> Optional[np.ndarray]:
        return None


class HDBSCANDetector:
    """
    HDBSCAN-based anomaly detection.
    Uses RobustScaler for input normalization.
    """

    def __init__(self, min_cluster_size: int = 5,
                 min_samples: Optional[int] = None,
                 cluster_selection_epsilon: float = 0.0):
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN not available")

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.model = None
        self.scaler = RobustScaler()

    def fit(self, X: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            prediction_data=True
        )
        self.model.fit(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted")

        X_scaled = self.scaler.transform(X)
        labels, probabilities = hdbscan.approximate_predict(self.model, X_scaled)

        anomaly_scores = np.zeros(len(X))
        noise_mask = labels == -1
        anomaly_scores[noise_mask] = 1.0
        
        cluster_mask = labels != -1
        # Score is inverse of cluster membership probability
        anomaly_scores[cluster_mask] = 1.0 - probabilities[cluster_mask]

        return anomaly_scores


class VAEDetector(nn.Module):
    """
    Variational Autoencoder for reconstruction-based anomaly detection.
    Implements robust loss functions for non-Gaussian data.
    """

    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: List[int] = [128, 64]):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU()
            ])
            current_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
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
        
        # Use RobustScaler for preprocessing internally if fit with numpy array
        self.scaler = RobustScaler()

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def loss_function(self, reconstruction: torch.Tensor, x: torch.Tensor,
                     mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute VAE loss using Huber loss for robustness against outliers.
        MSE is too sensitive to extreme values common in cosmology.
        """
        # Robust reconstruction loss (Huber/SmoothL1)
        recon_loss = F.smooth_l1_loss(reconstruction, x, reduction='sum')

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + kl_loss

    def _select_device(self, device: str) -> str:
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
        """Fit VAE on training data."""
        # Scale data first
        X_scaled = self.scaler.fit_transform(X)
        
        device = self._select_device(device)
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        X_tensor = torch.FloatTensor(X_scaled).to(device)
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

    def predict(self, X: np.ndarray, device: str = 'auto') -> np.ndarray:
        """Predict anomaly scores."""
        device = self._select_device(device)
        self.eval()
        self.to(device)

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(device)

        with torch.no_grad():
            reconstruction, _, _ = self.forward(X_tensor)
            # Use Smooth L1 error for scoring too
            error = F.smooth_l1_loss(reconstruction, X_tensor, reduction='none')
            reconstruction_error = torch.mean(error, dim=1)

        anomaly_scores = reconstruction_error.cpu().numpy()
        # Min-max normalization
        min_val = np.min(anomaly_scores)
        max_val = np.max(anomaly_scores)
        if max_val > min_val:
            anomaly_scores = (anomaly_scores - min_val) / (max_val - min_val)
        else:
            anomaly_scores = np.zeros_like(anomaly_scores)

        return anomaly_scores


class EnsembleDetector:
    """
    Ensemble of anomaly detection methods with weighted aggregation.
    """

    def __init__(self, input_dim: int,
                 methods: List[str] = ['isolation_forest', 'hdbscan', 'vae'],
                 weights: Optional[List[float]] = None):
        self.input_dim = input_dim
        self.methods = methods
        self.weights = weights or [1.0 / len(methods)] * len(methods)
        self.detectors = {}
        self._initialize_detectors()

    def _initialize_detectors(self):
        """Initialize individual detectors with robust parameters."""
        for method in self.methods:
            if method == 'isolation_forest':
                self.detectors[method] = IsolationForestDetector(
                    contamination=0.05,
                    n_estimators=200,
                    random_state=42
                )
            elif method == 'hdbscan':
                if HDBSCAN_AVAILABLE:
                    self.detectors[method] = HDBSCANDetector(
                        min_cluster_size=10,
                        min_samples=None,
                        cluster_selection_epsilon=0.0
                    )
            elif method == 'vae':
                self.detectors[method] = VAEDetector(
                    input_dim=self.input_dim,
                    latent_dim=64,
                    hidden_dims=[256, 128]
                )

    def fit(self, X: np.ndarray, **kwargs):
        for method, detector in self.detectors.items():
            if method == 'vae':
                detector.fit(X, **kwargs.get('vae_kwargs', {}))
            else:
                detector.fit(X)

    def predict(self, X: np.ndarray, **kwargs) -> Dict[str, Any]:
        individual_scores = {}
        ensemble_score = np.zeros(len(X))

        for i, (method, detector) in enumerate(self.detectors.items()):
            scores = detector.predict(X)
            individual_scores[method] = scores
            ensemble_score += self.weights[i] * scores

        ensemble_score = np.clip(ensemble_score, 0, 1)
        
        # Robust consensus (median vote)
        predictions = np.array([s > 0.5 for s in individual_scores.values()])
        consensus = np.median(predictions, axis=0) > 0.5

        return {
            'ensemble_scores': ensemble_score,
            'individual_scores': individual_scores,
            'consensus_prediction': consensus,
            'n_methods_agree': np.sum(predictions, axis=0)
        }

    def get_feature_importance(self) -> Dict[str, Any]:
        importance_info = {}
        for method, detector in self.detectors.items():
            importance = detector.get_feature_importance()
            if importance is not None:
                importance_info[method] = importance
        return importance_info
