DATAMODULE_REGISTRY = {}


def register_datamodule(name):
    def decorator(cls):
        DATAMODULE_REGISTRY[name] = cls
        return cls
    return decorator
