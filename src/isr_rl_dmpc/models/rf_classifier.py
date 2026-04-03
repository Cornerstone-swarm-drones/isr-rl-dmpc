"""
rf_classifier.py — removed.

The neural-network-based RFClassifier has been removed.  RF-based drone
identification is now handled by the Bayesian classifier in
``isr_rl_dmpc.modules.classification_engine``.  This stub is kept for
import backward-compatibility.
"""

__all__: list = []


def __getattr__(name: str):
    raise ImportError(
        f"'{name}' is not available: the neural-network RF classifier has "
        "been removed.  Use "
        "'isr_rl_dmpc.modules.classification_engine.ClassificationEngine' "
        "for target identification."
    )
