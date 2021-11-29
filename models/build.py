from .vlmo.vlmo_module import VlmoModule


def build_model(config):
    model_type = config.model.type
    if model_type == 'VLMO':
        # TODO: Update Model Arch params
        model = VlmoModule(config=config)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
