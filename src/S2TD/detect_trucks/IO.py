import pickle
import numpy as np
import rasterio as rio


class IO:
    def __init__(self, rf_truck_detector):
        """
        This is a class for all IO-related operations of the truck detector.
        :param rf_truck_detector: RFTruckDetector instance.
        """
        self.detector = rf_truck_detector

    def read_bands(self, file_path):
        """
        Reads Sentinel-2 bands from a file. It is necessary to read the data through this method to provide the
        metadata to the RFTruckDetector.
        :param file_path: str file path from where to read the Sentinel-2 band data. The file must contain four
        Sentinel-2 bands in the following order: B04, B03, B02, B08 (red, green, blue, near-infrared). Optimally,
        the data is in a projected CRS.
        :return: numpy array
        """
        try:
            with rio.open(file_path, "r") as src:
                if src.count < 4:
                    raise ValueError("Not enough bands. Got %s but need 4 Sentinel-2 bands in the order: "
                                     "B04, B03, B02, B08" % src.count)
                self.detector.meta = src.meta
                band_stack = np.zeros((src.count, src.height, src.width), dtype=src.meta["dtype"])
                for band_idx in range(4):  # hard-coded to avoid reading more than the four bands
                    band_stack[band_idx] = src.read(band_idx + 1)
        except rio.errors.RasterioIOError as e:
            print("Failed reading from %s" % file_path)
            raise e
        return band_stack

    @staticmethod
    def read_model(path):
        """
        Reads a trained random forest model from pickle.
        :param path: str file path to model.
        :return: sklearn.ensemble.RandomForestClassifier the model
        """
        try:
            model = pickle.load(open(path, "rb"))
        except FileNotFoundError as e:
            raise FileNotFoundError("Model file not found at: %s" % path)
        return model

    @staticmethod
    def write_model(model, path):
        """
        Writes a random forest model to pickle.
        :param model: sklearn.ensemble.RandomForestClassifier.
        :param path: str file path to model.
        :return: None
        """
        pickle.dump(model, open(path, "wb"))

    @staticmethod
    def _write_boxes(file_path, prediction_boxes, suffix):
        """
        Wrapper for writing predicted boxes.
        :param file_path: str file path to which boxes are written.
        :param prediction_boxes: gpd.GeoDataFrame prediction boxes.
        :param suffix: str that specifies the file format. One of: [".geojson", ".gpkg"].
        :return: None
        """
        if not file_path.endswith(suffix):
            file_path += suffix
        drivers = {".geojson": "GeoJSON", ".gpkg": "GPKG"}
        try:
            prediction_boxes.to_file(file_path, driver=drivers[suffix])
        except TypeError as e:
            raise e
        else:
            print("Wrote to: %s" % file_path)
