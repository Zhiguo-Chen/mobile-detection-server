from .fasterrcnn.fasterrcnn import FasterRCNN

MODELS = {
    'fasterrcnn': FasterRCNN
}


def get_model(model_name):
    if model_name not in MODELS:
        raise ValueError('"{}" is not in models'.format(model_name))
    return MODELS[model_name]
