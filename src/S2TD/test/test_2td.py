import os
import shutil
import unittest
import numpy as np
from glob import glob
import geopandas as gpd
from shapely.geometry import Polygon
from S2TruckDetect.src.S2TD.detect_trucks.S2TD import S2TD

s2_file = os.path.join("resources", "S2_test_data.tif")
dir_out = "resources"


class TestS2TD(unittest.TestCase):
    def test_input_dir_out(self):
        with self.assertRaises(ValueError):
            s2td = S2TD(99)  # must be str
        with self.assertRaises(FileExistsError):
            s2td = S2TD("wrong_dir_out")  # must exist
        with self.assertRaises(ValueError):
            s2td = S2TD(s2_file)  # must be directory

    def test_input_s2_file(self):
        s2td = S2TD(dir_out)
        with self.assertRaises(ValueError):
            s2td.detect(99, 20., True)  # must be str
        with self.assertRaises(FileExistsError):
            s2td.detect("no_file", 20., True)  # must exist
        with self.assertRaises(ValueError):
            s2td.detect(dir_out, 20., True)  # must be file

    def test_input_osm_buffer(self):
        with self.assertRaises(ValueError):
            s2td = S2TD(dir_out)
            s2td.detect(s2_file, "no_float", True)  # must be float

    def test_input_verbose(self):
        with self.assertRaises(ValueError):
            s2td = S2TD(dir_out)
            s2td.detect(s2_file, 20., "True")  # must be bool

    def test_output(self):
        s2td = S2TD(dir_out)
        result = s2td.detect(s2_file, 20., True)
        self.assertIsInstance(result, dict)
        keys = list(result.keys())
        self.assertEqual(keys[0], "detection_boxes")
        self.assertEqual(keys[1], "prediction_raster")
        self.assertIsInstance(result[keys[0]], gpd.GeoDataFrame)
        self.assertIsInstance(result[keys[0]].iloc[0].geometry, Polygon)
        self.assertIsInstance(result[keys[1]], np.ndarray)
        self.assertIsInstance(result[keys[1]][0, 0], np.int8)
        shutil.rmtree("cache")  # created by OSM OverpassAPI package
        os.remove(glob(os.path.join("resources", "*osm_roads.gpkg"))[0])


if __name__ == "__main__":
    unittest.main()
