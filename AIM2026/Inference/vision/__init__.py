"""VE perception (VLDrive vision encoder) for the rule-based Expert."""
__all__ = ['VisionRuntime', 'Perception', 'DEFAULT_LIDAR_TO_BASE', 'lidar_to_base']


def __getattr__(name):
    # Config/weight identity checks must not initialize the model dependencies.
    if name not in __all__:
        raise AttributeError(name)
    from importlib import import_module
    module = import_module('.preprocess' if name == 'lidar_to_base' else '.runtime', __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
