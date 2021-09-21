import numpy as np
import xarray as xr
from rasterio import features
from S2TruckDetect.src.S2TD.array_utils.geocoding import transform_lat_lon


def rasterize(polygons, lat, lon, fill=np.nan):
    """
    Rasterizes polygons.
    :param polygons: gpd.GeoDataFrame to be rasterized.
    :param lat: numpy array of latitude coordinates.
    :param lon: numpy array of longitude coordinates.
    :param fill: numeric fill value (no data).
    :return: xr.DataArray
    """
    transform = transform_lat_lon(lat, lon)
    out_shape = (len(lat), len(lon))
    raster = features.rasterize(polygons.geometry, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float)
    return xr.DataArray(raster, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
