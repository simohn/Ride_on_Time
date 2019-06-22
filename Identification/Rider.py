# Code written by Simon Schauppenlehner
# Last change: 22.06.2019

import numpy as np


class Rider:

    # Private Class variable

    _unique_id = 0

    #  public methods

    def __init__(self, id=-1):
        if id == -1:
            self.id = self._get_unique_id()
        else:
            self.id = id
        self.images = []
        self.features_average = []
        self.features_variance = []

    def add_image(self, image):
        self.images.append(image)

    def get_total_distance(self, image):
        total_dist = np.linalg.norm(self.get_features_average()-image.get_features())

        return total_dist

    def get_id(self):
        return self.id

    def calc_features_average(self):
        images_cnt = len(self.images)
        average = np.zeros(2048)

        for image in self.images:
            average += image.get_features()

        average = np.true_divide(average, images_cnt)

        self.features_average = average

    def calc_features_variance(self):
        images_cnt = len(self.images)
        variance = np.zeros(2048)

        for image in self.images:
            variance += np.power((image.get_features()-self.features_average), 2)

        variance = np.true_divide(variance, images_cnt)

        self.features_variance = variance

    def get_features_average(self):
        return self.features_average

    def get_features_variance(self):
        return self.features_variance

    # private methods

    @classmethod
    def _get_unique_id(cls):
        cls._unique_id += 1
        return cls._unique_id - 1
