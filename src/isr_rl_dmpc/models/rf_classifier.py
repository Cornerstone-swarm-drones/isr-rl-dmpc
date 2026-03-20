"""
RF Classifier - Neural-Network-Based Drone Classification from RF Fingerprints

Classifies drone targets as FRIENDLY, UNKNOWN, or HOSTILE using RF signal
features including mel-frequency coefficients, transient parameters, frequency
offset, harmonic distortion, and phase noise.

Complements the BayesianClassifier in modules/classification_engine.py by
providing a learned, neural-net-based classifier operating on RF features.

Classification Labels:
  FRIENDLY = +1  (cooperative drone)
  UNKNOWN  =  0  (unidentified)
  HOSTILE  = -1  (adversarial drone)
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Tuple, Optional
from enum import IntEnum
import logging

logger = logging.getLogger(__name__)


class DroneClass(IntEnum):
    """Drone classification labels."""
    HOSTILE = -1
    UNKNOWN = 0
    FRIENDLY = 1


# Mapping from class index (0, 1, 2) to DroneClass label
_INDEX_TO_LABEL = [DroneClass.HOSTILE, DroneClass.UNKNOWN, DroneClass.FRIENDLY]


@dataclass
class RFFingerprint:
    """
    RF fingerprint features extracted from a drone's radio emissions.

    Attributes:
        mel_coefficients: Mel-frequency cepstral coefficients of the RF signal
        transient_params: Parameters characterizing signal turn-on transient
        freq_offset: Carrier frequency offset in Hz
        harmonic_distortion: Harmonic distortion ratios (dBc)
        phase_noise: Phase noise power spectral density in dBc/Hz
    """

    mel_coefficients: np.ndarray = field(default_factory=lambda: np.zeros(13))
    transient_params: np.ndarray = field(default_factory=lambda: np.zeros(8))
    freq_offset: float = 0.0
    harmonic_distortion: np.ndarray = field(default_factory=lambda: np.zeros(5))
    phase_noise: float = 0.0

    def __post_init__(self):
        """Validate array shapes."""
        self.mel_coefficients = np.asarray(self.mel_coefficients, dtype=np.float64)
        self.transient_params = np.asarray(self.transient_params, dtype=np.float64)
        self.harmonic_distortion = np.asarray(self.harmonic_distortion, dtype=np.float64)

    def to_feature_vector(self) -> np.ndarray:
        """
        Flatten all fingerprint fields into a single 1-D feature vector.

        Returns:
            1-D numpy array of concatenated features.
        """
        return np.concatenate([
            self.mel_coefficients,
            self.transient_params,
            np.array([self.freq_offset]),
            self.harmonic_distortion,
            np.array([self.phase_noise]),
        ])


class RFClassifier(nn.Module):
    """
    Neural-network classifier for RF-fingerprint-based drone identification.

    Architecture:
      Input (feature_dim) → Linear(hidden_dim) → BatchNorm → ReLU → Dropout →
      Linear(hidden_dim) → BatchNorm → ReLU → Dropout → Linear(num_classes)

    The three output classes correspond to FRIENDLY (+1), UNKNOWN (0), and
    HOSTILE (-1).
    """

    def __init__(self, feature_dim: int = 28, num_classes: int = 3,
                 hidden_dim: int = 128):
        """
        Initialize RF classifier.

        Args:
            feature_dim: Dimension of RF fingerprint feature vector
            num_classes: Number of classification classes (default 3)
            hidden_dim: Size of hidden layers
        """
        super(RFClassifier, self).__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Layer 1
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)

        # Layer 2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(0.1)

        # Output
        self.fc3 = nn.Linear(hidden_dim, num_classes)

        self.relu = nn.ReLU()

        logger.info(f"RFClassifier initialized (feature_dim={feature_dim}, "
                    f"num_classes={num_classes}, hidden_dim={hidden_dim})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing class logits.

        Args:
            x: Input tensor of shape (batch_size, feature_dim) or (feature_dim,)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        logits = self.fc3(x)
        return logits

    @torch.no_grad()
    def classify(self, fingerprint: RFFingerprint) -> Tuple[int, float]:
        """
        Classify a drone from its RF fingerprint.

        Args:
            fingerprint: RF fingerprint dataclass instance

        Returns:
            Tuple of (class_label, confidence) where class_label is one of
            {-1, 0, +1} and confidence is in [0, 1].
        """
        self.eval()
        features = fingerprint.to_feature_vector()
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        confidence, class_idx = torch.max(probs, dim=1)

        label = int(_INDEX_TO_LABEL[class_idx.item()])
        return label, float(confidence.item())

    @staticmethod
    def extract_features(raw_signal: np.ndarray,
                         sample_rate: float) -> RFFingerprint:
        """
        Extract RF fingerprint features from a raw IQ signal.

        Computes mel-frequency coefficients, transient parameters, frequency
        offset, harmonic distortion, and phase noise from the raw samples.

        Args:
            raw_signal: Raw IQ samples as a 1-D complex or real numpy array
            sample_rate: Sampling rate in Hz

        Returns:
            RFFingerprint with extracted features
        """
        signal = np.asarray(raw_signal, dtype=np.complex128)
        n_samples = len(signal)

        # Mel-frequency coefficients via FFT magnitude binning
        spectrum = np.abs(np.fft.fft(signal))[:n_samples // 2]
        n_mels = 13
        bin_edges = np.linspace(0, len(spectrum), n_mels + 1, dtype=int)
        mel_coefficients = np.array([
            np.mean(spectrum[bin_edges[i]:bin_edges[i + 1]])
            if bin_edges[i] < bin_edges[i + 1] else 0.0
            for i in range(n_mels)
        ])

        # Transient parameters from signal envelope
        envelope = np.abs(signal)
        transient_params = np.zeros(8)
        if n_samples >= 8:
            # Rise time, overshoot, settling statistics
            transient_params[0] = float(np.max(envelope[:n_samples // 4]))
            transient_params[1] = float(np.mean(envelope[:n_samples // 4]))
            transient_params[2] = float(np.std(envelope[:n_samples // 4]))
            transient_params[3] = float(np.max(envelope) - np.mean(envelope))
            transient_params[4:8] = np.polyfit(
                np.arange(min(4, n_samples)), envelope[:min(4, n_samples)], 3
            ) if n_samples >= 4 else np.zeros(4)

        # Frequency offset from peak of power spectrum
        freqs = np.fft.fftfreq(n_samples, d=1.0 / sample_rate)
        power = np.abs(np.fft.fft(signal)) ** 2
        peak_idx = np.argmax(power[:n_samples // 2])
        freq_offset = float(freqs[peak_idx])

        # Harmonic distortion (first 5 harmonics relative to fundamental)
        fundamental_freq = freqs[peak_idx] if peak_idx > 0 else sample_rate / n_samples
        harmonic_distortion = np.zeros(5)
        for h in range(5):
            h_freq = fundamental_freq * (h + 2)
            h_idx = int(abs(h_freq) * n_samples / sample_rate) % (n_samples // 2)
            if power[peak_idx] > 0:
                ratio = power[h_idx] / power[peak_idx]
                harmonic_distortion[h] = float(
                    10.0 * np.log10(ratio) if ratio > 0 else -120.0
                )
            else:
                harmonic_distortion[h] = -120.0

        # Phase noise estimate
        phase = np.unwrap(np.angle(signal))
        phase_noise = float(np.std(np.diff(phase))) if n_samples > 1 else 0.0

        logger.debug(f"Extracted RF fingerprint: freq_offset={freq_offset:.1f} Hz, "
                     f"phase_noise={phase_noise:.4f}")

        return RFFingerprint(
            mel_coefficients=mel_coefficients,
            transient_params=transient_params,
            freq_offset=freq_offset,
            harmonic_distortion=harmonic_distortion,
            phase_noise=phase_noise,
        )
