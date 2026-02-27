from typing import List
import os
import importlib

from core.model.model_instance import ModelInstance, ModelInstanceType

_MODEL_PROVIDER = {}

def register_model_provider(name: str = None):
    def decorator(cls):
        key = name or cls.__name__
        _MODEL_PROVIDER[key] = cls
        return cls
    return decorator

def auto_import_model_provider():
    current_dir = os.path.dirname(__file__)
    package = __package__

    for filename in os.listdir(current_dir):
        if filename.startswith("model_instance_") and filename.endswith(".py"):
            module_name = filename[:-3]
            full_module_name = f"{package}.{module_name}"
            importlib.import_module(full_module_name)

auto_import_model_provider()

class ModelManager:

    @staticmethod
    def get_supported_provider() -> List[str]:
        return list(_MODEL_PROVIDER.keys())

    @staticmethod
    def get_model_instance(provider: str, model_type: ModelInstanceType, **kwargs) -> ModelInstance:
        if provider not in _MODEL_PROVIDER.keys():
            raise RuntimeError(f"'{provider}' is not supported.")
        return _MODEL_PROVIDER.get(provider)(model_type=model_type, **kwargs)
