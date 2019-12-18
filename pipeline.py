import sys
import train.predict


class LIDARPipeline:

    def __init__(self, weightsFile):
        self.model = train.predict.load_model(weightsFile)

    def predict_position(self, point_cloud):
        return train.predict.predict_point_cloud(self.model, point_cloud)
