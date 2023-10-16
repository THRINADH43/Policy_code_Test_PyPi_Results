from pureml import BasePredictor, Input, Output
import pureml


class Predictor(BasePredictor):
    label = "credit_example_test_pypi1:main:v1"
    input = Input(type="numpy ndarray")
    output = Output(type="numpy ndarray")

    def load_models(self):
        self.model = pureml.model.fetch(self.label)

    def predict(self, data):
        predictions = self.model.predict(data)

        return predictions
