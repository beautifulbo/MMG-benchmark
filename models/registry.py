# coding: utf-8

"""
Model registry for plugin-style model registration.
Usage:
    from models.registry import register_model

    @register_model('MMCL')
    class MMCL(GeneralRecommender):
        ...
"""
import importlib


class ModelRegistry:
    """
    Registry for models. Allows plugin-style registration of new models.
    """
    _models = {}

    @classmethod
    def register(cls, name):
        """
        Decorator to register a model.

        Args:
            name: Model name string

        Usage:
            @ModelRegistry.register('DGMRec')
            class DGMRec(GeneralRecommender):
                ...
        """
        def decorator(Model):
            cls._models[name] = Model
            return Model
        return decorator

    @classmethod
    def get(cls, name):
        """
        Get model class by name.

        Args:
            name: Model name string

        Returns:
            Model class or None if not found
        """
        if name in cls._models:
            return cls._models[name]

        # Try dynamic import from models module
        model_file_name = name.lower()
        module_path = '.'.join(['models', model_file_name])
        try:
            model_module = importlib.import_module(module_path, __name__)
            model_class = getattr(model_module, name)
            cls._models[name] = model_class
            return model_class
        except (ImportError, AttributeError):
            return None

    @classmethod
    def list_models(cls):
        """List all registered model names."""
        return list(cls._models.keys())

    @classmethod
    def clear(cls):
        """Clear all registered models. Useful for testing."""
        cls._models = {}


def register_model(name):
    """Convenience decorator wrapper."""
    return ModelRegistry.register(name)