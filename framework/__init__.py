"""Framework layer: adapter protocol, manifest interpreter, manager, workers."""
from .adapter import (
    AudioResult,
    ModelAdapter,
    ModelCapabilities,
    ModelNotInitializedError,
    ModelNotReadyError,
    ModelSettings,
    NoModelForCapabilityError,
    SynthesisError,
    create_adapter,
    load_manifest,
)
from .model_manager import ModelManager

__all__ = [
    "AudioResult",
    "ModelAdapter",
    "ModelCapabilities",
    "ModelNotInitializedError",
    "ModelNotReadyError",
    "ModelSettings",
    "NoModelForCapabilityError",
    "SynthesisError",
    "create_adapter",
    "load_manifest",
    "ModelManager",
]