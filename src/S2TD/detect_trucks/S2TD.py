import os
import numpy as np
import xarray as xr
import geopandas as gpd
from fiona import errors
from datetime import datetime
from S2TruckDetect.src.S2TD.detect_trucks.ObjectExtractor import ObjectExtractor
from S2TruckDetect.src.S2TD.detect_trucks.IO import IO
from S2TruckDetect.src.S2TD.osm_utils.osm_utils import fetch_roads
from S2TruckDetect.src.S2TD.osm_utils.osm_utils import rasterize_roads
from S2TruckDetect.src.S2TD.array_utils.math import rescale_s2
from S2TruckDetect.src.S2TD.array_utils.math import normalized_ratio
from S2TruckDetect.src.S2TD.array_utils.geocoding import lat_from_meta
from S2TruckDetect.src.S2TD.array_utils.geocoding import lon_from_meta
from S2TruckDetect.src.S2TD.array_utils.geocoding import metadata_to_bbox_epsg4326
from S2TruckDetect.src.S2TD.utils.console import print_start
from S2TruckDetect.src.S2TD.utils.console import print_status
from S2TruckDetect.src.S2TD.utils.console import print_end

MODEL_PATH = os.path.join(os.path.dirname(__file__), "rf_model.pickle")  # RF model


class S2TD:
    def __init__(self, dir_out):
        """
        This is the interface to the Sentinel-2 TruckDetect (S2TD) method.
        :param dir_out: str directory where to save OSM road polygons.
        """
        self.dir_out = dir_out  # str directory for saving OSM road polygons
        self.io = IO(self)  # class for IO operations
        self.rf = self.io.read_model(MODEL_PATH)  # random forest model
        self.lat, self.lon = None, None  # np array lat pixel coordinates, np array lon pixel coordinates
        self.meta = None  # rasterio meta
        self.feature_stack = None  # 3d np array, the feature stack for the random forest prediction
        self.probabilities = None  # 3d np array, the random forest probabilities
        self.osm_buffer = None  # float buffer in meters for buffering OSM road vectors -> polygons
        self.verbose = None

    def detect(self, s2_file, osm_buffer=20.0, verbose=True):
        """
        Detects moving trucks and other large moving vehicles on roads using the Sentinel-2 TruckDetect (S2TD) method.
        The method relies on a temporal sensing offset of the Sentinel-2 Multispectral Instrument (MSI),
        causing spatially and spectrally disassembled reflectance patterns of large moving vehicles. It uses a random
        forest prediction and a recursive neighborhood object extraction. The output are bounding boxes,
        each delineating a detected large moving vehicle (usually a truck).
        :param s2_file: str file path from where to read the Sentinel-2 band data. The file must contain four
        Sentinel-2 bands in the following order: B04, B03, B02, B08 (red, green, blue, near-infrared). They should be
        cloud-masked, so, cloudy pixels should be np.nan in the data. Optimally, the data is in a projected CRS.
        :param osm_buffer: float the buffer distance [m] for the highest OpenStreetMaps (OSM) road category
        ("motorway"), defaults to 20.0 m. This determines how large the polygon buffer is, starting at the road line
        vector obtained from OSM. Lower road classes ("trunk", "primary") are buffered with a buffer 5 m smaller than
        the next-higher class. Only OSM road classes of the keys "motorway", "trunk", "primary" are processed.
        :param verbose: bool
        :return: dict: "detection_boxes" is a GeoDataFrame with bounding boxes delineating the detections,
        "prediction_raster" is a two-dimensional numpy array of the random forest prediction based upon which moving
        trucks were detected.
        """
        print("verbose=%s" % str(verbose))
        t0 = datetime.now()
        self.verbose = verbose
        self.osm_buffer = osm_buffer
        print_start("S2TD - Start", self.verbose)
        print_status("Preparing Sentinel-2 data", self.verbose)
        bands = self.io.read_bands(s2_file)
        print_status("Building feature stack", self.verbose)
        self._build_feature_stack(bands)
        print_status("Predicting target pixels", self.verbose)
        prediction_raster = self._predict()
        print_status("Extracting target objects", self.verbose)
        detection_boxes = self._extract_objects(prediction_raster)
        print_end("S2TD - End", datetime.now()-t0, self.verbose)
        return {"detection_boxes": detection_boxes, "prediction_raster": prediction_raster}

    def _build_feature_stack(self, band_stack):
        """
        Builds the feature stack.
        :param band_stack: numpy array of 3-dimensional Sentinel-2 band data in order: B04, B03, B02, B08
        :return: None
        """
        bs_preprocessed = self._preprocess_bands(band_stack)
        shape = bs_preprocessed.shape
        fs = np.zeros((7, shape[1], shape[2]), dtype=np.float16)  # feature stack
        fs[0] = np.nanvar(bs_preprocessed[0:3], 0, dtype=np.float16, ddof=-1)
        fs[1] = normalized_ratio(bs_preprocessed[0], bs_preprocessed[2]).astype(np.float16)  # red/blue
        fs[2] = normalized_ratio(bs_preprocessed[1], bs_preprocessed[2]).astype(np.float16)  # green/blue
        for band_idx in range(shape[0]):
            bs_preprocessed[band_idx] -= np.nanmean(bs_preprocessed[band_idx])  # center band to its mean
        fs[3] = bs_preprocessed[0]  # B04 minus its mean
        fs[4] = bs_preprocessed[1]  # B03 minus its mean
        fs[5] = bs_preprocessed[2]  # B02 minus its mean
        fs[6] = bs_preprocessed[3]  # B08 minus its mean
        fs[:, np.isnan(fs[3])] = np.nan  # ensure nans
        self.feature_stack = fs.astype(np.float16)  # 3-dimensional, holds seven features

    def _predict(self):
        """
        Executes the random forest prediction including all data wrangling.
        :return: np.int8
        """
        if self.rf is None:
            self.rf = self.io.read_model(MODEL_PATH)
        vars_reshaped = []
        for band_idx in range(self.feature_stack.shape[0]):
            vars_reshaped.append(self.feature_stack[band_idx].flatten())
        vars_reshaped = np.array(vars_reshaped).swapaxes(0, 1)  # (n observations, n variables)
        nan_mask = np.zeros_like(vars_reshaped)
        for var_idx in range(vars_reshaped.shape[1]):
            nan_mask[:, var_idx] = ~np.isnan(vars_reshaped[:, var_idx])  # exclude nans
        not_nan = np.nanmin(nan_mask, 1).astype(np.bool) * np.min(np.isfinite(vars_reshaped), 1).astype(np.bool)
        predictions_flat = self.rf.predict_proba(vars_reshaped[not_nan])
        probabilities_shaped = vars_reshaped[:, 0:4].copy()
        for idx in range(predictions_flat.shape[1]):
            probabilities_shaped[not_nan, idx] = predictions_flat[:, idx]
        probabilities_shaped = np.swapaxes(probabilities_shaped, 0, 1)
        probabilities_shaped = probabilities_shaped.reshape((probabilities_shaped.shape[0], self.feature_stack.shape[1],
                                                             self.feature_stack.shape[2]))
        self.probabilities = probabilities_shaped
        self.probabilities[:, np.isnan(self.feature_stack[0])] = 0
        prediction = self._postprocess_prediction()
        return prediction

    def _extract_objects(self, predictions_arr):
        """
        Wrapper for the object extraction.
        :param predictions_arr: np.int8 the random forest prediction.
        :return: gpd.GeoDataFrame
        """
        extractor = ObjectExtractor(self)
        out_gpd = extractor.extract(predictions_arr)
        return out_gpd

    def _preprocess_bands(self, band_stack):
        """
        Pre-processes the Sentinel-2 band data.
        :param band_stack: numpy array with 4 two-dimensional arrays. The four Sentinel-2 bands must be in the
        following order: B04, B03, B02, B08 (red, green, blue, near-infrared).
        :return: None
        """
        bands_rescaled = band_stack[0:4].copy()
        bands_rescaled[np.isnan(bands_rescaled)] = 0
        bands_rescaled = rescale_s2(bands_rescaled)
        bands_rescaled[bands_rescaled == 0] = np.nan
        bands_rescaled = bands_rescaled.astype(np.float16)
        band_stack = None
        self.lat, self.lon = lat_from_meta(self.meta), lon_from_meta(self.meta)
        bbox_epsg4326 = list(np.flip(metadata_to_bbox_epsg4326(self.meta)))
        print_status("Fetching OSM road data", self.verbose)
        osm_mask = self._create_osm_mask(bbox_epsg4326, bands_rescaled[0], {"lat": self.lat, "lon": self.lon})
        if np.count_nonzero(osm_mask) == 0:
            raise ValueError("No OSM roads of requested road types in aoi")
        bands_rescaled *= osm_mask
        bands_rescaled[bands_rescaled == 0] = np.nan
        osm_mask = None
        return bands_rescaled

    def _postprocess_prediction(self):
        self.probabilities[1][self.probabilities[1] < 0.75] = 0
        classification = np.nanargmax(self.probabilities, 0) + 1
        classification[np.max(self.probabilities, 0) == 0] = 0
        return classification.astype(np.int8)

    @staticmethod
    def pick_arr_subset(arr, y, x, size):
        """
        Picks a subset from a 2-dimensional array. The subset is a window with the provided position in its center.
        :param arr: numpy array.
        :param y: int y index.
        :param x: int x index.
        :param size: int size of subset in pixels.
        :return: numpy array
        """
        pseudo_max = np.inf
        size_low = int(size / 2)
        size_up = int(size / 2)
        size_up = size_up + 1 if (size_low + size_up) < size else size_up
        ymin, ymax = int(np.clip(y - size_low, 0, pseudo_max)), int(np.clip(y + size_up, 0, pseudo_max))
        xmin, xmax = int(np.clip(x - size_low, 0, pseudo_max)), int(np.clip(x + size_up, 0, pseudo_max))
        n = len(arr.shape)
        if n == 2:
            subset = arr[ymin:ymax, xmin:xmax]
        elif n == 3:
            subset = arr[:, int(np.clip(y - size_low, 0, pseudo_max)):int(np.clip(y + size_up, 0, pseudo_max)),
                         int(np.clip(x - size_low, 0, pseudo_max)):int(np.clip(x + size_up, 0, pseudo_max))]
        else:
            subset = arr
        return subset

    def _create_osm_mask(self, bbox, reference_arr, lat_lon_dict):
        """
        Wrapper for creating a rasterized OSM road mask.
        :param bbox: array-like of four coordinates: miny, minx, maxy, maxx.
        :param reference_arr: numpy array, the grid to be used for the road mask.
        :param lat_lon_dict: dict of "lat" and "lon" coordinates numpy arrays.
        :return: np.float16
        """
        osm_file = fetch_roads(bbox,
                               ["motorway", "trunk", "primary"],
                               self.osm_buffer,
                               self.dir_out,
                               str(bbox).replace(", ", "_").replace("-", "minus")[1:-1] + "_osm_roads",
                               self.meta["crs"])
        try:
            osm_vec = gpd.read_file(osm_file)
        except errors.DriverError:
            return np.zeros_like(reference_arr, dtype=np.float16)
        else:
            print_status("Rasterizing OSM road data", self.verbose)
            ref_xr = xr.DataArray(data=reference_arr, coords=lat_lon_dict, dims=["lat", "lon"])
            osm_raster = rasterize_roads(osm_vec, ref_xr).astype(np.float16)
            osm_raster[osm_raster != 0] = 1
            osm_raster[osm_raster == 0] = np.nan
            return osm_raster


if __name__ == "__main__":
    dirs = dict(main="F:\\Masterarbeit\\DLR\\project\\1_truck_detection")
    f = "F:\\Masterarbeit\\DLR\\project\\1_truck_detection\\validation\\data\\s2\\archive\\s2_bands_AS_Dierdorf_VQ_Nord_2018-07-22_2018-07-22_merged.tiff"
    rf_td = S2TD(dirs["main"])
    boxes = rf_td.detect(f, 20.0)
    boxes["detection_boxes"].to_file(os.path.join(dirs["main"], "test.gpkg"), driver="GPKG")
