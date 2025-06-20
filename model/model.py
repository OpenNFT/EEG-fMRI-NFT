import joblib


class Inference:
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        if x.shape[1] != 48:
            return None
        pred = float(self.model.predict(x.reshape(-1, 1).T)) / 10
        return (pred + .5) * 2


def get_model(save_model_path):
    model = joblib.load(save_model_path)
    return Inference(model)
