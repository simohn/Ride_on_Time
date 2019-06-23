# Code written by Simon Schauppenlehner
# Last change: 22.06.2019

import xlsxwriter
import numpy as np
from scipy.spatial.distance import cdist


class Evaluator:

    # Public methods

    def __init__(self):
        self.riders = []

    def add_rider(self, rider):
        self.riders.append(rider)

    def get_predicted_rider(self, image):

        ranking = []

        for index, rider in enumerate(self.riders):
            dist = rider.get_total_distance(image)
            ranking.append((rider, dist))

        def take_second(elem):
            return elem[1]

        ranking.sort(key=take_second)

        return ranking[0][0]

    def get_val_acc(self, test_images):
        count_correct = 0
        count_false = 0

        for test_index, test_image in enumerate(test_images):

            if self.get_predicted_rider(test_image).get_id() == test_image.get_rider_id():
                count_correct = count_correct + 1
            else:
                count_false = count_false + 1

        return count_correct / (count_false+count_correct)

    def export_stat(self, test_images, path_and_name):

        workbook = xlsxwriter.Workbook(path_and_name)

        cf_green = workbook.add_format()
        cf_green.set_pattern(1)  # This is optional when using a solid fill.
        cf_green.set_bg_color("#008000")  # green

        cf_red = workbook.add_format()
        cf_red.set_pattern(1)  # This is optional when using a solid fill.
        cf_red.set_bg_color("#FF0000")  # green

        cf_grey = workbook.add_format()
        cf_grey.set_pattern(1)  # This is optional when using a solid fill.
        cf_grey.set_bg_color("#A0A0A0")  # grey

        # ------ test images predictions ------

        worksheet1 = workbook.add_worksheet()
        worksheet2 = workbook.add_worksheet()

        # Header
        worksheet1.write(0, 0, "IMG_RiderID", cf_grey)
        worksheet1.write(0, 1, "1_RiderID", cf_grey)
        worksheet1.write(0, 2, "1_Dist", cf_grey)
        worksheet1.write(0, 3, "2_RiderID", cf_grey)
        worksheet1.write(0, 4, "2_Dist", cf_grey)
        worksheet1.write(0, 5, "3_RiderID", cf_grey)
        worksheet1.write(0, 6, "3_Dist", cf_grey)
        worksheet1.write(0, 7, "DistTrueRider", cf_grey)

        worksheet2.write(0, 0, "IMG_RiderID", cf_grey)
        worksheet2.write(0, 1, "1_RiderID", cf_grey)
        worksheet2.write(0, 2, "1_RelDist", cf_grey)
        worksheet2.write(0, 3, "2_RiderID", cf_grey)
        worksheet2.write(0, 4, "2_RelDist", cf_grey)
        worksheet2.write(0, 5, "3_RiderID", cf_grey)
        worksheet2.write(0, 6, "3_RelDist", cf_grey)
        worksheet2.write(0, 7, "RelDistTrueRider", cf_grey)

        ranking = []

        for test_index, test_image in enumerate(test_images):
            for index, rider in enumerate(self.riders):
                dist = rider.get_total_distance(test_image)
                ranking.append((rider, dist))

            def take_second(elem):
                return elem[1]

            ranking.sort(key=take_second)

            test_image_rider_id = test_image.get_rider_id()
            dist_to_correct_rider = 0

            for x in ranking:
                if x[0].get_id() == test_image_rider_id:
                    dist_to_correct_rider = x[1]
                    break

            worksheet1.write(test_index+1, 0, test_image_rider_id)
            worksheet2.write(test_index+1, 0, test_image_rider_id)

            worksheet1.write(test_index + 1, 1, ranking[0][0].get_id())
            worksheet1.write(test_index + 1, 2, ranking[0][1])

            dist_rel = ranking[0][1] / ranking[0][1] - 1
            worksheet2.write(test_index + 1, 1, ranking[0][0].get_id())
            worksheet2.write(test_index + 1, 2, dist_rel)

            pred_rider_is_correct = ranking[0][0].get_id() == test_image_rider_id

            for i in range(1, 3):

                dist_rel = ranking[i][1]/ranking[0][1]-1

                cf = workbook.add_format()
                cf.set_pattern(1)  # This is optional when using a solid fill.
                cf.set_bg_color(self._get_color_code(dist_rel, 0.5, 0.1))

                if pred_rider_is_correct:
                    worksheet1.write(test_index + 1, 2 * i + 1, ranking[i][0].get_id())
                    worksheet1.write(test_index + 1, 2 * i + 2, ranking[i][1], cf)

                    worksheet2.write(test_index + 1, 2 * i + 1, ranking[i][0].get_id())
                    worksheet2.write(test_index + 1, 2 * i + 2, dist_rel, cf)
                else:
                    worksheet1.write(test_index + 1, 2 * i + 1, ranking[i][0].get_id())
                    worksheet1.write(test_index + 1, 2 * i + 2, ranking[i][1])

                    worksheet2.write(test_index + 1, 2 * i + 1, ranking[i][0].get_id())
                    worksheet2.write(test_index + 1, 2 * i + 2, dist_rel)

            cf = workbook.add_format()
            cf.set_pattern(1)  # This is optional when using a solid fill.
            cf.set_bg_color(self._get_color_code(dist_to_correct_rider, 0.5, 0.1))

            if pred_rider_is_correct:
                worksheet1.write(test_index + 1, 7, dist_to_correct_rider)
                worksheet2.write(test_index + 1, 7, dist_to_correct_rider/ranking[0][1]-1)
            else:
                worksheet1.write(test_index + 1, 7, dist_to_correct_rider, cf_red)
                worksheet2.write(test_index + 1, 7, dist_to_correct_rider / ranking[0][1] - 1, cf_red)

            ranking.clear()

        worksheet1.freeze_panes(1, 1)
        worksheet2.freeze_panes(1, 1)

        # ------ total distances riders -------

        worksheet3 = workbook.add_worksheet()

        riders_count = len(self.riders)
        features_average_riders = np.zeros((riders_count, 2048))

        for index, rider in enumerate(self.riders):
            features_average_riders[index] = rider.get_features_average()

            # Header
            worksheet3.write(0, index+1, "RID_" + str(self.riders[index].get_id()), cf_grey)
            worksheet3.write(index+1, 0, "RID_" + str(self.riders[index].get_id()), cf_grey)

        dist_matrix = cdist(features_average_riders, features_average_riders, metric='euclidean')

        for row in range(1, riders_count+1):
            for col in range(1, riders_count + 1):
                dist = dist_matrix[row-1][col-1]

                cf = workbook.add_format()
                cf.set_pattern(1)  # This is optional when using a solid fill.
                cf.set_bg_color(self._get_color_code(dist, 20, 5))

                worksheet3.write(row, col, dist, cf)

        worksheet3.freeze_panes(1, 1)

        workbook.close()

    def is_imgage_rider(self, imgage, rider):
        pred_rider = self.get_predicted_rider(imgage)

        if pred_rider == rider:
            return True
        else:
            return False

    def get_rider_by_id(self, id):

        for index, rider in enumerate(self.riders):
            if str(rider.get_id()) == str(id):
                return rider

        return None

    # Private methods

    def _get_color_code(self, dist, dist_max, dist_min):

        k = 1/(dist_max-dist_min)
        d = -k*dist_min

        # " R G B "
        # "#008000"

        if dist < dist_min:
            dist_rel = 0
        elif dist > dist_max:
            dist_rel = 1
        else:
            dist_rel = k*dist + d

            if dist_rel < 0:
                dist_rel = 0
            elif dist_rel > 1:
                dist_rel = 1

        color_code_hex = '#%02x%02x%02x' % (int((1-dist_rel)*255), int(dist_rel*255), 0)

        return color_code_hex
