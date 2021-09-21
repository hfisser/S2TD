import numpy as np
import geopandas as gpd
from shapely.geometry import box, Polygon
from S2TruckDetect.src.S2TD.utils.ProgressBar import ProgressBar

SECONDS_OFFSET_B02_B04 = 1.01  # Sentinel-2 temporal sensing offset between B02 and B04


class ObjectExtractor:
    def __init__(self, rf_truck_detector):
        self.detector = rf_truck_detector

    def extract(self, predictions_arr):
        """
        Iterates over all positions predicted as "blue" in the prediction array and tries to extract target objects
        around this blue pixel. This extraction is done by recursive neighborhood search. If no target object is found
        at a specific "blue" position, it is dropped. Otherwise, a bounding box is created to delineate the object.
        An extracted object finally gets characterizing attributes including approximate speed and heading, and is
        added to an output GeoDataFrame holding all extracted objects as bounding boxes with four real-world
        coordinates.
        :param predictions_arr: np.int8, the full two-dimensional random forest prediction.
        :return: gpd.GeoDataFrame
        """
        detector = self.detector
        predictions_copy, probabilities_copy = predictions_arr.copy(), detector.probabilities.copy()
        predictions_copy[predictions_copy == 1] = 0
        blue_ys, blue_xs = np.where(predictions_copy == 2)
        out_gpd = gpd.GeoDataFrame({"geometry": []}, crs=detector.meta["crs"])  # output GeoDataFrame
        detection_boxes, directions, direction_descriptions, speeds, mean_probs, sub_size = [], [], [], [], [], 9
        pb = ProgressBar(len(blue_ys), int(len(blue_ys) * 0.025))
        for i, y_blue, x_blue in zip(range(len(blue_ys)), blue_ys, blue_xs):
            pb.update(i)
            if predictions_copy[y_blue, x_blue] == 0:
                continue
            subset_9 = detector.pick_arr_subset(predictions_copy, y_blue, x_blue, sub_size).copy()
            subset_3 = detector.pick_arr_subset(predictions_copy, y_blue, x_blue, 3).copy()
            subset_9_probs = detector.pick_arr_subset(probabilities_copy, y_blue, x_blue, sub_size).copy()
            half_idx_y = y_blue if subset_9.shape[0] < sub_size else int(subset_9.shape[0] * 0.5)
            half_idx_x = x_blue if subset_9.shape[1] < sub_size else int(subset_9.shape[1] * 0.5)
            try:
                current_value = subset_9[half_idx_y, half_idx_x]
            except IndexError:  # upper array edge
                half_idx_y, half_idx_x = int(sub_size / 2), int(sub_size / 2)  # index from lower edge is ok
                current_value = subset_9[half_idx_y, half_idx_x]
            new_value = 100
            if not all([value in subset_9 for value in [2, 3, 4]]):
                continue
            result_tuple = self._cluster_array(
                arr=subset_9,
                probs=subset_9_probs,
                point=[half_idx_y, half_idx_x],
                new_value=new_value,
                current_value=current_value,
                yet_seen_indices=[],
                yet_seen_values=[],
                skipped_one=False)
            cluster = result_tuple[0]
            if np.count_nonzero(cluster == new_value) < 3:
                continue
            else:
                out_gpd, predictions_copy = self._postprocess_cluster(
                    cluster, predictions_copy,
                    subset_3, y_blue, x_blue,
                    half_idx_y, half_idx_x,
                    new_value,
                    out_gpd)
        print("")
        return out_gpd

    def _cluster_array(self,
                       arr,
                       probs,
                       point,
                       new_value,
                       current_value,
                       yet_seen_indices,
                       yet_seen_values,
                       skipped_one):
        """
        Looks for non-zeros in a 3x3 window around a point in an array and assigns a new value to these non-zeros.
        :param arr: numpy array, the prediction subset.
        :param point: array-like of int y, x indices, the current position.
        :param new_value: int value to assign to array positions included in the cluster.
        :param current_value: int the current prediction value.
        :param yet_seen_indices: list of lists, each list is a point with int y, x indices that has been seen before.
        :param yet_seen_values: list of values, each value is a prediction value at the yet_seen_indices.
        :return: tuple holding a numpy array and a list
        """
        detector = self.detector
        if len(yet_seen_indices) == 0:
            yet_seen_indices.append(point)
            yet_seen_values.append(current_value)
        arr_modified = arr.copy()
        arr_modified[point[0], point[1]] = 0
        window_3x3 = detector.pick_arr_subset(arr_modified.copy(), point[0], point[1], 3)
        if window_3x3[1, 1] == 2:
            window_3x3[window_3x3 == 4] = 1  # eliminate reds in 3x3 neighborhood of blue
        y, x, ys, xs, window_idx, offset_y, offset_x = point[0], point[1], [], [], 0, 0, 0
        window_3x3_probs = detector.pick_arr_subset(probs, y, x, 3)
        # first look for values on horizontal and vertical, if none given try corners
        windows, windows_probs = [window_3x3], [window_3x3_probs]
        windows = windows[0:1] if current_value == 4 or skipped_one else windows
        while len(ys) == 0 and window_idx < len(windows):
            window = windows[window_idx]
            window_probs = windows_probs[window_idx]
            offset_y, offset_x = int(window.shape[0] / 2), int(window.shape[1] / 2)  # offset for window ymin and xmin
            go_next = current_value + 1 in window or current_value == 2
            target_value = current_value + 1 if go_next else current_value
            match = window == target_value
            target_value = current_value if np.count_nonzero(match) == 0 else target_value
            match = window == target_value
            ys, xs = np.where(match)
            if len(ys) > 1:  # look for match with highest probability
                window_probs_target = window_probs[target_value - 1] * match
                max_prob = (window_probs_target == np.max(window_probs_target))
                ys, xs = np.where(max_prob)
            window_idx += 1
        ymin, xmin = int(np.clip(point[0] - offset_y, 0, np.inf)), int(np.clip(point[1] - offset_x, 0, np.inf))
        for y_local, x_local in zip(ys, xs):
            y, x = ymin + y_local, xmin + x_local
            if [y, x] not in yet_seen_indices or len(yet_seen_indices) == 0:
                try:
                    current_value = arr[y, x]
                except IndexError:
                    continue
                if 4 in yet_seen_values and current_value <= 3:  # red yet seen but this is green or blue
                    continue
                arr_modified[y, x] = new_value
                yet_seen_indices.append([y, x])
                yet_seen_values.append(current_value)
                # avoid picking many more reds than blues and greens
                n_picks = [np.count_nonzero(np.array(yet_seen_values) == value) for value in [2, 3, 4]]
                if n_picks[2] > n_picks[0] and n_picks[2] > n_picks[1]:
                    break  # finish clustering in order to avoid picking many reds at the edge of object
                arr_modified, yet_seen_indices, yet_seen_values, skipped_one = self._cluster_array(
                    arr_modified, probs, [y, x], new_value, current_value, yet_seen_indices, yet_seen_values,
                    skipped_one)
        arr_modified[point[0], point[1]] = new_value
        return arr_modified, yet_seen_indices, yet_seen_values, skipped_one

    def _postprocess_cluster(self,
                             cluster,
                             preds_copy,
                             prediction_subset_3,
                             y_blue,
                             x_blue,
                             half_idx_y,
                             half_idx_x,
                             new_value,
                             out_gpd):
        """
        Post-processes a cluster and creates a prediction box.
        :param cluster: numpy array, the cluster.
        :param preds_copy: numpy array, copy of the random forest prediction.
        :param prediction_subset_3: numpy array, the prediction in a 3x3 window around the blue origin pixel.
        :param y_blue: int, the array y index of the blue origin pixel.
        :param x_blue: int, the array x index of the blue origin pixel.
        :param half_idx_y:
        :param half_idx_x:
        :param new_value:
        :param out_gpd:
        :return:
        """
        # add neighboring blue in 3x3 window around blue
        ys_blue_additional, xs_blue_additional = np.where(prediction_subset_3 == 2)
        ys_blue_additional += half_idx_y - 1  # get index in subset
        xs_blue_additional += half_idx_x - 1
        for y_blue_add, x_blue_add in zip(ys_blue_additional, xs_blue_additional):
            cluster[int(np.clip(y_blue_add, 0, np.inf)), int(np.clip(x_blue_add, 0, np.inf))] = new_value
        cluster[cluster != new_value] = 0
        cluster_ys, cluster_xs = np.where(cluster == new_value)
        # corner of 9x9 subset
        ymin_subset, xmin_subset = np.clip(y_blue - half_idx_y, 0, np.inf), np.clip(x_blue - half_idx_x, 0, np.inf)
        cluster_ys += ymin_subset.astype(cluster_ys.dtype)
        cluster_xs += xmin_subset.astype(cluster_xs.dtype)
        ymin, xmin = np.min(cluster_ys), np.min(cluster_xs)
        # +1 on index because Polygon has to extent up to upper bound of pixel (array coords at upper left corner)
        ymax, xmax = np.max(cluster_ys) + 1, np.max(cluster_xs) + 1
        # check if blue, green and red are given in box and box is large enough, otherwise drop
        box_preds = preds_copy[ymin:ymax, xmin:xmax].copy()
        box_probs = self.detector.probabilities[1:, ymin:ymax, xmin:xmax].copy()
        max_probs = [np.nanmax(box_probs[value - 2] * (box_preds == value)) for value in (2, 3, 4)]
        mean_max_spectral_probability = np.nanmean(max_probs)
        mean_spectral_probability = np.nanmean(np.nanmax(box_probs, 0))
        all_given = all([value in box_preds for value in [2, 3, 4]])
        large_enough = box_preds.shape[0] > 2 or box_preds.shape[1] > 2
        too_large = box_preds.shape[0] > 5 or box_preds.shape[1] > 5
        if too_large > 0 or not all_given or not large_enough:
            return out_gpd, preds_copy
        # calculate direction
        by, bx = np.where(box_preds == 2)
        ry, rx = np.where(box_preds == 4)
        # simply use first index
        blue_indices = np.int8([by[0], bx[0]])
        red_indices = np.int8([ry[0], rx[0]])
        vector = (blue_indices - red_indices) * np.int8([1, -1])  # multiply x by -1 because axis reverse
        direction = self.calc_vector_direction_in_degree(vector)
        speed = self.calc_speed(box_preds.shape)
        # create output box
        lat, lon = self.detector.lat, self.detector.lon
        lon_min, lat_min = lon[xmin], lat[ymin]
        try:
            lon_max = lon[xmax]
        except IndexError:  # may happen at edge of array
            # calculate next coordinate beyond array bound -> this is just the upper boundary of the box
            lon_max = lon[-1] + (lon[-1] - lon[-2])
        try:
            lat_max = lat[ymax]
        except IndexError:
            lat_max = lat[-1] + (lat[-1] - lat[-2])
        cluster_box = Polygon(box(lon_min, lat_min, lon_max, lat_max))
        score = mean_max_spectral_probability + mean_spectral_probability
        if score > 1.2:
            # set box cells to zero value in predictions array
            preds_copy[ymin:ymax, xmin:xmax] *= np.zeros_like(box_preds)
            blue_indices = np.where(box_preds == 2)
            for yb, xb in zip(blue_indices[0], blue_indices[1]):  # 3x3 around cell blues to 0
                ymin, ymax = np.clip(yb - 1, 0, preds_copy.shape[0]), np.clip(yb + 2, 0, preds_copy.shape[0])
                xmin, xmax = np.clip(xb - 1, 0, preds_copy.shape[1]), np.clip(xb + 2, 0, preds_copy.shape[1])
                preds_copy[ymin:ymax, xmin:xmax] *= np.int8(preds_copy[ymin:ymax, xmin:xmax] != 2)
            box_idx = len(out_gpd)
            for key, value in zip(["geometry", "id", "detection_score", "mean_spectral_probability",
                                   "mean_max_spectral_probability", "max_blue_probability",
                                   "max_green_probability", "max_red_probability", "direction_degree",
                                   "direction_description", "speed", "red", "green", "blue"],
                                  [cluster_box, box_idx, score,
                                   mean_spectral_probability, mean_max_spectral_probability,
                                   max_probs[0], max_probs[1], max_probs[2], direction,
                                   self.direction_degree_to_description(direction), speed]):
                out_gpd.loc[box_idx, key] = value
        return out_gpd, preds_copy

    @staticmethod
    def calc_speed(box_shape):
        """
        Calculates a speed [km/h] from the dimensions of the prediction box.
        :param box_shape: array-like the shape of the prediction box.
        :return: float
        """
        diameter = np.max(box_shape) * 10 - 10  # 10 m resolution; -10 for center of pixel
        return np.sqrt(diameter * 20)/SECONDS_OFFSET_B02_B04 * 3.6

    @staticmethod
    def calc_vector_direction_in_degree(vector):
        """
        Calculates a vector orientation in degrees (0-360).
        :param vector: array-like y, x indices.
        :return: float
        """
        # [1,1] -> 45°; [-1,1] -> 135°; [-1,-1] -> 225°; [1,-1] -> 315°
        return np.degrees(np.arctan2(vector[1], vector[0])) % 360

    @staticmethod
    def direction_degree_to_description(direction_degree):
        """
        Translates a direction from degrees to a compass direction description.
        :param direction_degree: float direction in degrees.
        :return: str
        """
        step = 45
        bins = np.arange(0, 359, step, dtype=np.float32)
        descriptions = np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
        diff = np.abs(bins - direction_degree)
        return descriptions[np.where(diff == np.min(diff))[0][0]]
