"""
Checkpoint Utilities — Generic file-based persistence helpers.

Provides a lightweight :class:`ModelRegistry` for organising named
configuration or parameter archives on disk, plus standalone
:func:`save_checkpoint` / :func:`load_checkpoint` helpers used by the
DMPC controller.

All neural-network-specific logic (PyTorch state-dict handling, etc.)
has been removed.  Archives are now plain NumPy ``.npz`` or JSON files.
"""

from __future__ import annotations

import json
import logging
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """
    Metadata associated with a saved configuration checkpoint.

    Attributes:
        name:           Human-readable model/config name.
        version:        Semantic version string.
        state_dim:      Dimension of the state space.
        control_dim:    Dimension of the control/action space.
        training_steps: Reserved field (always 0 for pure DMPC configs).
        metrics:        Dictionary of evaluation metrics.
    """

    name: str = ""
    version: str = "0.1.0"
    state_dim: int = 0
    control_dim: int = 0
    training_steps: int = 0
    metrics: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to JSON-serialisable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelMetadata":
        """Reconstruct from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ModelRegistry:
    """
    Registry for managing saved configuration checkpoints on disk.

    Directory layout::

        registry_dir/
        ├── config_a/
        │   ├── checkpoint.npz
        │   └── metadata.json
        └── config_b/
            ├── checkpoint.npz
            └── metadata.json
    """

    _CHECKPOINT_FILE = "checkpoint.npz"
    _METADATA_FILE = "metadata.json"

    def __init__(self, registry_dir: Path) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ModelRegistry initialised at %s", self.registry_dir)

    def register(
        self,
        name: str,
        data: Dict[str, Any],
        metadata: Optional[ModelMetadata] = None,
    ) -> Path:
        """
        Save a NumPy data dictionary and optional metadata to the registry.

        Args:
            name:     Unique configuration name (used as directory name).
            data:     Dictionary mapping string keys to NumPy-serialisable
                      values (arrays, scalars, strings).
            metadata: Optional :class:`ModelMetadata` instance.

        Returns:
            Path to the saved checkpoint directory.
        """
        model_dir = self.registry_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = model_dir / self._CHECKPOINT_FILE
        np.savez(ckpt_path, **{k: np.asarray(v) for k, v in data.items()})

        if metadata is None:
            metadata = ModelMetadata(name=name)
        meta_path = model_dir / self._METADATA_FILE
        with open(meta_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        logger.info("Registered config '%s' at %s", name, model_dir)
        return model_dir

    def load(self, name: str) -> tuple[Dict[str, Any], ModelMetadata]:
        """
        Load a configuration and its metadata from the registry.

        Args:
            name: Configuration name (directory name in registry).

        Returns:
            Tuple of ``(data_dict, metadata)``.
        """
        model_dir = self.registry_dir / name
        if not model_dir.exists():
            raise FileNotFoundError(f"Config '{name}' not found in registry")

        metadata = self.get_metadata(name)
        ckpt_path = model_dir / self._CHECKPOINT_FILE
        raw = np.load(ckpt_path)
        data = {k: raw[k] for k in raw.files}
        logger.info("Loaded config '%s'", name)
        return data, metadata

    def list_models(self) -> List[str]:
        """Return sorted list of registered configuration names."""
        return sorted(
            d.name
            for d in self.registry_dir.iterdir()
            if d.is_dir() and (d / self._CHECKPOINT_FILE).exists()
        )

    def get_metadata(self, name: str) -> ModelMetadata:
        """Retrieve metadata for a registered configuration."""
        meta_path = self.registry_dir / name / self._METADATA_FILE
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Metadata for '{name}' not found at {meta_path}"
            )
        with open(meta_path) as f:
            data = json.load(f)
        return ModelMetadata.from_dict(data)


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path: str, data: Dict[str, Any], metadata: Optional[Dict] = None) -> None:
    """
    Save a NumPy data dictionary to a ``.npz`` checkpoint file.

    Args:
        path:     Output file path (the ``.npz`` extension is added if absent).
        data:     Dictionary of NumPy-serialisable values to persist.
        metadata: Optional dictionary of scalar metadata values.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: np.asarray(v) for k, v in data.items()}
    if metadata:
        # Store metadata as a JSON string in a dedicated array entry
        payload["_metadata_json"] = np.array(json.dumps(metadata))
    np.savez(out_path, **payload)
    logger.info("Checkpoint saved to %s", out_path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    """
    Load a NumPy ``.npz`` checkpoint file.

    Args:
        path: File path of the checkpoint.

    Returns:
        Dictionary of arrays/values loaded from the archive.
    """
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    raw = np.load(ckpt_path, allow_pickle=False)
    result: Dict[str, Any] = {k: raw[k] for k in raw.files}
    if "_metadata_json" in result:
        result["_metadata"] = json.loads(str(result.pop("_metadata_json")))
    logger.info("Checkpoint loaded from %s", ckpt_path)
    return result
