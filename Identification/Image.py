# Code written by Simon Schauppenlehner
# Last change: 22.06.2019

class Image:

    # private class variables

    _unique_id = 0
    # every Image has the same model
    _model = None

    #  public methods

    def __init__(self, imgage_path, rider_id):
        self.id = Image._get_unique_id()
        self.imgage_path = imgage_path
        self.features = Image._model.calc_features(self.imgage_path)
        self.rider_id = rider_id

    def get_features(self):
        return self.features

    def get_rider_id(self):
        return self.rider_id

    def get_path(self):
        return self.imgage_path

    @classmethod
    def set_model(cls, model):
        Image._model = model

    @classmethod
    def get_model_details(cls):
        return cls._model.get_model_details()

    # private methods

    @classmethod
    def _get_unique_id(cls):
        cls._unique_id += 1
        return cls._unique_id - 1
