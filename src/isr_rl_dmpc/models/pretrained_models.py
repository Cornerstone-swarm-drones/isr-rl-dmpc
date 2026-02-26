"""
Pretrained Models - Checkpoint Management and Model Registry

Provides utilities for saving, loading, and cataloguing trained model
checkpoints used across the ISR-RL-DMPC system (value networks, policy
networks, RF classifiers, etc.).
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Type
import logging
from dataclasses import dataclass, field, asdict
import json

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """
    Metadata associated with a saved model checkpoint.

    Attributes:
        name: Human-readable model name
        version: Semantic version string
        state_dim: Dimension of the state space the model was trained on
        action_dim: Dimension of the action space (0 for state-value models)
        training_steps: Total training steps completed
        metrics: Dictionary of evaluation metrics (e.g. loss, reward)
    """

    name: str = ""
    version: str = "0.1.0"
    state_dim: int = 0
    action_dim: int = 0
    training_steps: int = 0
    metrics: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelMetadata":
        """Reconstruct from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ModelRegistry:
    """
    Registry for managing saved model checkpoints on disk.

    Stores each model as a directory containing a checkpoint file and a
    JSON metadata sidecar under the configured registry root.

    Directory layout::

        registry_dir/
        ├── model_a/
        │   ├── checkpoint.pt
        │   └── metadata.json
        └── model_b/
            ├── checkpoint.pt
            └── metadata.json
    """

    _CHECKPOINT_FILE = "checkpoint.pt"
    _METADATA_FILE = "metadata.json"

    def __init__(self, registry_dir: Path):
        """
        Initialize model registry.

        Args:
            registry_dir: Root directory for storing model checkpoints
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ModelRegistry initialized at {self.registry_dir}")

    def register(self, name: str, model: nn.Module,
                 metadata: Optional[ModelMetadata] = None) -> Path:
        """
        Save a model and its metadata to the registry.

        Args:
            name: Unique model name (used as directory name)
            model: PyTorch model to save
            metadata: Optional metadata; a default is created if not provided

        Returns:
            Path to the saved checkpoint directory
        """
        model_dir = self.registry_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save checkpoint
        ckpt_path = model_dir / self._CHECKPOINT_FILE
        torch.save(model.state_dict(), ckpt_path)

        # Save metadata
        if metadata is None:
            metadata = ModelMetadata(name=name)
        meta_path = model_dir / self._METADATA_FILE
        with open(meta_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        logger.info(f"Registered model '{name}' at {model_dir}")
        return model_dir

    def load(self, name: str,
             model_class: Optional[Type[nn.Module]] = None,
             **model_kwargs) -> Tuple[Optional[nn.Module], ModelMetadata]:
        """
        Load a model and its metadata from the registry.

        If *model_class* and *model_kwargs* are provided, an instance is
        created and the saved state dict is loaded into it. Otherwise only
        the raw state dict is returned via the metadata's metrics field.

        Args:
            name: Model name (directory name in registry)
            model_class: Optional class to instantiate
            **model_kwargs: Keyword arguments forwarded to model_class()

        Returns:
            Tuple of (model_instance_or_None, metadata)
        """
        model_dir = self.registry_dir / name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model '{name}' not found in registry")

        metadata = self.get_metadata(name)

        model = None
        ckpt_path = model_dir / self._CHECKPOINT_FILE
        if model_class is not None:
            model = model_class(**model_kwargs)
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            logger.info(f"Loaded model '{name}' ({model_class.__name__})")
        else:
            logger.info(f"Loaded metadata for '{name}' (no model_class provided)")

        return model, metadata

    def list_models(self) -> List[str]:
        """
        List all registered model names.

        Returns:
            Sorted list of model names present in the registry
        """
        return sorted(
            d.name for d in self.registry_dir.iterdir()
            if d.is_dir() and (d / self._CHECKPOINT_FILE).exists()
        )

    def get_metadata(self, name: str) -> ModelMetadata:
        """
        Retrieve metadata for a registered model.

        Args:
            name: Model name

        Returns:
            ModelMetadata instance

        Raises:
            FileNotFoundError: If model or metadata file does not exist
        """
        meta_path = self.registry_dir / name / self._METADATA_FILE
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Metadata for model '{name}' not found at {meta_path}"
            )
        with open(meta_path, "r") as f:
            data = json.load(f)
        return ModelMetadata.from_dict(data)


# ---------------------------------------------------------------------------
# Standalone helper functions
# ---------------------------------------------------------------------------

def save_checkpoint(path: str, model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    epoch: int = 0,
                    metadata: Optional[Dict] = None) -> None:
    """
    Save a training checkpoint to disk.

    Args:
        path: File path for the checkpoint (e.g. ``checkpoints/step_1000.pt``)
        model: PyTorch model whose state dict to save
        optimizer: Optional optimizer whose state dict to save
        epoch: Current training epoch / step counter
        metadata: Optional dictionary of extra information
    """
    ckpt: Dict = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if metadata is not None:
        ckpt["metadata"] = metadata

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    logger.info(f"Checkpoint saved to {out_path} (epoch={epoch})")


def load_checkpoint(path: str,
                    model: Optional[nn.Module] = None,
                    optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
    """
    Load a training checkpoint from disk.

    If *model* or *optimizer* are provided their state dicts are restored
    in-place from the checkpoint.

    Args:
        path: File path of the checkpoint
        model: Optional model to restore weights into
        optimizer: Optional optimizer to restore state into

    Returns:
        Full checkpoint dictionary (contains at least ``model_state_dict``
        and ``epoch``).
    """
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if model is not None and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Restored model weights from {ckpt_path}")

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info(f"Restored optimizer state from {ckpt_path}")

    logger.info(f"Checkpoint loaded from {ckpt_path} (epoch={ckpt.get('epoch', 'N/A')})")
    return ckpt
