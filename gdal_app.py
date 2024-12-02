import os
import sys
import ctypes
# print(ctypes.util.find_library('gdal'))
# Path to the GDAL DLLs
gdal_dll_path = r"C:\ProgramData\Miniconda3\envs\gdal_env_latest\Library\bin"
r"C:\ProgramData\Miniconda3\envs\gdal_env\Library\bin"

if sys.version_info >= (3, 8):
     os.add_dll_directory(gdal_dll_path)

# print(ctypes.util.find_library('gdal'))
import cv2
import numpy as np
from osgeo import gdal


# Globals for mouse interaction
mouse_pressed = False
start_mouse_pos = None
image, transform = None, None
current_zoom = 1000  # Start with 1000x1000 pixels
current_zoom_level = 0  # Base resolution
block_size = 600
dataset = None

current_center = (35.178471, 31.789318)  # Longitude, Latitude
 # Update the block's origin for the new block
current_block_origin_x = None
current_block_origin_y = None


def get_overview_geotransform(dataset, overview_level):
    """
    Calculate the GeoTransform for a specific overview level.

    Parameters:
        dataset (gdal.Dataset): The GDAL dataset.
        overview_level (int): The desired overview level (0-based).

    Returns:
        list: The calculated GeoTransform for the overview level.
    """
    # Base GeoTransform
    base_geo_transform = dataset.GetGeoTransform()
   
    # Base and overview dimensions
    band = dataset.GetRasterBand(1)
    overview = band.GetOverview(overview_level)
    if overview is None:
        raise ValueError(f"Overview level {overview_level} does not exist.")

    base_x_size = band.XSize
    base_y_size = band.YSize
    overview_x_size = overview.XSize
    overview_y_size = overview.YSize

    # Calculate scale factors
    x_scale = base_x_size / overview_x_size
    y_scale = base_y_size / overview_y_size

    # Adjust pixel size in GeoTransform
    overview_geo_transform = list(base_geo_transform)
    overview_geo_transform[1] *= x_scale  # Pixel width
    overview_geo_transform[5] *= y_scale  # Pixel height

    return overview_geo_transform

def read_tiff_block_gdal(dataset, coord_center, block_size, overview_level):
    # Get the GeoTransform for the overview level (either from overviews or base level)
    if overview_level == -1:
        # Use the base level GeoTransform (main dataset)
        overview_geo_transform = dataset.GetGeoTransform()
    else:
        # Use the GeoTransform of the overview level
        overview_geo_transform = get_overview_geotransform(dataset, overview_level)

    # Calculate pixel coordinates from geographical coordinates
    lon, lat = coord_center
    x_pixel = int((lon - overview_geo_transform[0]) / overview_geo_transform[1])
    y_pixel = int((lat - overview_geo_transform[3]) / overview_geo_transform[5])

    # Calculate the block's top-left corner
    x_offset = max(x_pixel - block_size // 2, 0)
    y_offset = max(y_pixel - block_size // 2, 0)

    # Initialize a list to hold band data
    bands_data = []

    # Loop through all the bands in the dataset
    for band_index in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(band_index)

        if overview_level == -1:
            # Read directly from the base level (dataset)
            block_data = band.ReadAsArray(x_offset, y_offset, block_size, block_size)
        else:
            # Get the overview (pyramid) level
            overview = band.GetOverview(overview_level)
            if overview is None:
                raise ValueError(f"Overview level {overview_level} does not exist for band {band_index}.")

            # Ensure block size does not exceed the overview's dimensions
            x_block_size = min(block_size, overview.XSize - x_offset)
            y_block_size = min(block_size, overview.YSize - y_offset)

            # Read the block data for this band from the overview
            block_data = overview.ReadAsArray(x_offset, y_offset, x_block_size, y_block_size)

        if block_data is None:
            raise ValueError(f"Unable to read data from band {band_index}.")
        
        bands_data.append(block_data)

    # Merge the bands into a color image (BGR order expected by OpenCV)
    if len(bands_data) > 1:
        # If bands are in RGB order, swap the channels to BGR
        if len(bands_data) == 3:
            red, green, blue = bands_data
            color_image = cv2.merge([blue, green, red])  # BGR order
        else:
            color_image = cv2.merge(bands_data)  # No change needed if in BGR order
    else:
        color_image = bands_data[0]  # Single-band grayscale image
        
    return color_image, overview_geo_transform


def mouse_callback(event, x, y, flags, param):
    """Handles mouse events for dragging and updating the map."""
    global mouse_pressed, start_mouse_pos, current_center, image, transform , current_zoom ,current_zoom_level , block_size,current_block_origin_x,current_block_origin_y
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Mouse button pressed
        mouse_pressed = True
        start_mouse_pos = (x, y)
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            # Optional: Show a dragging indicator
            pass
    
    elif event == cv2.EVENT_LBUTTONUP:
        # Mouse button released, calculate offset and update view
        mouse_pressed = False
        end_mouse_pos = (x, y)
        
        # Calculate pixel offset
        dx = end_mouse_pos[0] - start_mouse_pos[0]
        dy = end_mouse_pos[1] - start_mouse_pos[1]
        
        # Convert pixel offset to degrees using the transform
        lat_offset = -dy * transform[1]  # Vertical resolution
        lon_offset = dx * transform[5]   # Horizontal resolution
        
        # Update the current center (current_center) based on offset
        current_center = (
            current_center[0] + lon_offset,  # Update longitude
            current_center[1] + lat_offset   # Update latitude
        )
        # Update center coordinates
        current_center = (current_center[0] + lat_offset, current_center[1] + lon_offset)
        
        # Load new portion of the map
        image, transform = read_tiff_block_gdal(dataset, current_center, block_size, current_zoom_level)
        
        # Update the display
        cv2.imshow("Map View", image)
    if event == cv2.EVENT_MOUSEWHEEL:
        # Set the maximum zoom level (number of overviews + base level)
        max_zoom_level = 13

        # Determine zoom direction
        if flags > 0:  # Scroll up: Zoom in (higher resolution)
            current_zoom_level = max(-1, current_zoom_level - 1)
        else:  # Scroll down: Zoom out (lower resolution)
            current_zoom_level = min(max_zoom_level, current_zoom_level + 1)

        # Calculate the dataset-relative pixel position
        dataset_x = current_block_origin_x + x
        dataset_y = current_block_origin_y + y

        # Convert the dataset-relative pixel position to geographic coordinates
        lon = transform[0] + dataset_x * transform[1] + dataset_y * transform[2]
        lat = transform[3] + dataset_x * transform[4] + dataset_y * transform[5]

        # Update the current center to the new geographic coordinates
        current_center = (lon, lat)

        # Read the new image block at the updated zoom level
        image, transform = read_tiff_block_gdal(dataset, current_center, block_size, current_zoom_level)

        # Update the block's origin for the new block
        current_block_origin_x = int((current_center[0] - transform[0]) / transform[1]) - block_size // 2
        current_block_origin_y = int((current_center[1] - transform[3]) / transform[5]) - block_size // 2

        # Display the updated image
        cv2.imshow("Map View", image)
    # if event == cv2.EVENT_MOUSEWHEEL:
    #     # with rasterio.open(TIFF_PATH) as src:
    #     #     max_zoom_level = len(src.overviews(1))  # Number of overviews available
    #     max_zoom_level = 13
    #     if flags > 0:  # Scroll up: Zoom in (higher resolution)
    #         current_zoom_level = max(-1, current_zoom_level - 1)
    #     else:  # Scroll down: Zoom out (lower resolution)
    #         current_zoom_level = min(max_zoom_level, current_zoom_level + 1)
        
    #     image, transform = read_tiff_block_gdal(dataset, current_center, block_size, current_zoom_level)
    #     cv2.imshow("Map View", image)

def main():
    os.environ["OPENCV_GUI_PLUGIN"] = "OFF"  # Disable GUI plugins (like Qt)
    global transform , image , dataset , block_size , current_zoom_level,current_block_origin_x,current_block_origin_y
    tiff_path = r"my_map.tif"
    start_coords = (35.178471, 31.789318)  # Longitude, Latitude
    block_size = 500
    current_zoom_level = 5  # Change this to see the effect of different levels
    dataset = gdal.Open(tiff_path)
    if dataset is None:
        raise FileNotFoundError(f"Unable to open file: {tiff_path}")
    
    image, transform = read_tiff_block_gdal(dataset, start_coords, block_size, current_zoom_level)
     # Update the block's origin for the new block
    current_block_origin_x = int((current_center[0] - transform[0]) / transform[1]) - block_size // 2
    current_block_origin_y = int((current_center[1] - transform[3]) / transform[5]) - block_size // 2

    # Set up the OpenCV window and mouse callback
    cv2.namedWindow("Map View")
    cv2.setMouseCallback("Map View", mouse_callback)
    
    while True:
        # Display the current map view
        cv2.imshow("Map View", image)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    # Read the block

    # # Display the image
    # cv2.imshow('GeoTIFF Block', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()