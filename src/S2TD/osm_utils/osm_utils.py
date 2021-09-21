import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from S2TruckDetect.src.S2TD.array_utils.points import rasterize
from OSMPythonTools.overpass import Overpass
from OSMPythonTools.overpass import overpassQueryBuilder


def buffer_bbox(bbox_osm):
    """
    Buffers a EPSG:4326 bounding box slightly to ensure covering the whole area of interest.
    :param bbox_osm: array-like of four coordinates: miny, minx, maxy, maxx.
    :return: array-like of four coordinates: miny, minx, maxy, maxx
    """
    offset_lat, offset_lon = 0.02, 0.02  # degrees
    bbox_osm[0] -= offset_lat  # min lat
    bbox_osm[1] -= offset_lon  # min lon
    bbox_osm[2] += offset_lat  # max lat
    bbox_osm[3] += offset_lon  # max lon
    return bbox_osm


def fetch_osm(bbox, osm_value="motorway", osm_key="highway"):
    """
    Fetches OSM road data from the OverpassAPI.
    :param bbox: array-like of four coordinates: miny, minx, maxy, maxx.
    :param osm_value: str specifies the OSM value to be retrieved.
    :param osm_key: str specifies the OSM key to be retrieved.
    :return: gpd.GeoDataFrame
    """
    element_type = ["way", "relation"]
    bbox_osm = buffer_bbox(bbox)
    quot = '"'
    select = quot + osm_key + quot + "=" + quot + osm_value + quot
    select_link = select.replace(osm_value, osm_value + "_link")  # also get road links
    select_junction = select.replace(osm_value, osm_value + "_junction")
    geoms = []
    for selector in [select, select_link, select_junction]:
        query = overpassQueryBuilder(bbox=bbox_osm,
                                     elementType=element_type,
                                     selector=selector,
                                     out="body",
                                     includeGeometry=True)
        try:
            elements = Overpass().query(query, timeout=120).elements()
        except Exception:  # type?
            elements = []
            Warning("Could not download OSM data")
        # create multiline of all elements
        if len(elements) > 0:
            for i in range(len(elements)):
                elem = elements[i]
                try:
                    geoms.append(elem.geometry())
                except Exception:
                    continue
        else:
            Warning("Could not retrieve " + select)
    if len(geoms) > 0:
        lines = gpd.GeoDataFrame(crs="EPSG:4326", geometry=geoms)
        n = len(geoms)
        lines["osm_value"] = [osm_value] * n  # add road type
        return lines


def fetch_roads(bbox, osm_values, buffer_meters, dir_out, filename, crs):
    """
    Iterates over all OSM road values of interest, fetches the road data, creates buffered road polygons.
    :param bbox: array-like of four coordinates: miny, minx, maxy, maxx.
    :param osm_values: array-like of str OSM road values.
    :param buffer_meters: float buffer in meters for creating road polygons.
    :param dir_out: str directory where to write the road data.
    :param filename: str file name prefix to use when writing the road data.
    :param crs: str output crs.
    :return: str file path to road polygons
    """
    if crs.is_geographic:
        # estimate UTM EPSG for buffer in meters. Output will be CRS specified by crs argument
        p = Point(bbox[:2][::-1])
        crs_projected = gpd.GeoDataFrame({"geometry": [p]}, crs="EPSG:4326").estimate_utm_crs().to_string()
    else:
        crs_projected = crs.to_string()
    fwrite = os.path.join(dir_out, filename + ".gpkg")
    file_tmp = os.path.join(dir_out, "tmp.gpkg")
    buffer_dist = "buffer_distance"
    if os.path.exists(file_tmp):
        os.remove(file_tmp)
    if os.path.exists(fwrite):
        pass
    else:
        roads = []
        offset = 5  # meters
        # buffer according to road type
        m, t, p, s, ter = "motorway", "trunk", "primary", "secondary", "tertiary"
        buffers = {m: buffer_meters, t: buffer_meters - offset, p: buffer_meters - offset,
                   s: buffer_meters - (3 * offset), ter: buffer_meters - (4 * offset)}
        osm_values_int = {m: 1, t: 2, p: 3, s: 4, ter: 5}
        for osm_value in osm_values:
            roads_osm = fetch_osm(bbox=bbox, osm_value=osm_value)
            if roads_osm is None:
                pass
            else:
                roads_osm.to_file(file_tmp, driver="GPKG")
                roads_osm = gpd.read_file(file_tmp)
                roads_osm = roads_osm.to_crs(crs_projected)
                roads_osm[buffer_dist] = [buffers[osm_value]] * len(roads_osm)
                roads_osm["osm_value_int"] = osm_values_int[osm_value]
                roads.append(roads_osm)
        try:
            roads_merge = gpd.GeoDataFrame(pd.concat(roads, ignore_index=True), crs=roads[0].crs)  # merge all roads
        except ValueError:
            Warning("No road vectors")
        else:
            buffered = roads_merge.buffer(distance=roads_merge[buffer_dist])  # buffer the road vectors -> polygons
            roads_merge.geometry = buffered
            roads_merge.to_crs(crs).to_file(fwrite, driver="GPKG")
            if os.path.exists(file_tmp):
                os.remove(file_tmp)
    return fwrite


def rasterize_roads(osm, reference_raster):
    """
    Rasterizes road polygons to a reference grid.
    :param osm: gpd.GeoDataFrame contains the road polygons.
    :param reference_raster: numpy array with two dimensions, the reference grid.
    :return: numpy array with two dimensions, the rasterized road polygons.
    """
    osm_values = list(set(osm["osm_value"]))
    nan_placeholder = 100
    road_rasters = []
    for osm_value in osm_values:
        osm_subset = osm[osm["osm_value"] == osm_value]
        raster = rasterize(osm_subset, reference_raster.lat, reference_raster.lon)
        cond = np.isfinite(raster)
        raster_osm = np.where(cond, list(osm_subset.osm_value_int)[0],
                              nan_placeholder)  # use placeholder instead of nan first
        raster_osm = raster_osm.astype(np.float)
        road_rasters.append(raster_osm)
        # merge road types in one layer
    # use the lowest value (highest road level) because some intersect
    road_raster_np = np.int8(road_rasters).min(axis=0)
    road_raster_np[road_raster_np == nan_placeholder] = 0
    return road_raster_np  # 0=no_road 1=motorway, 2=trunk, ...
