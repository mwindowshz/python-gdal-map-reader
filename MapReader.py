import os
import sys
import cv2
import numpy as np
from osgeo import gdal


class GdalMapReader:
    def __init__(self):
        self.dataset = None
        self.file_name = None
        self.initialized = False
        self.image = None
        self.current_transform = None
        self.current_zoom_level = 0  # Base resolution
        self.block_size_x = 900
        self.block_size_y = 600
        self.num_overviews = 0
        self.current_block_origin_x  = -1
        self.current_block_origin_y = -1
        self.current_block_X = 900
        self.current_block_Y = 600
    # Globals for mouse interaction
    # start_mouse_pos = None
    # mouse_pressed = False



        self.current_center = (35.178471, 31.789318)  # Longitude, Latitude
    # Update the block's origin for the new block
    # self.current_block_origin_x = None
    # self.current_block_origin_y = None

    def get_image(self):
        return self.image
    
    def open_file(self,file_name):
        self.file_name = file_name
        self.dataset = gdal.Open(file_name)
        if  self.dataset is None:
            self.initialized = False
            print (f"Unable to open file: {file_name}")
            return
        # Get the first raster band (assuming single-band dataset)
        band = self.dataset.GetRasterBand(1)

        # Get the number of overviews
        self.num_overviews = band.GetOverviewCount() - 1 #startig from 0 
        self.initialized = True

    def get_overview_geotransform(self, overview_level):
        """
        Calculate the GeoTransform for a specific overview level.

        Parameters:
            dataset (gdal.Dataset): The GDAL dataset.
            overview_level (int): The desired overview level (0-based).

        Returns:
            list: The calculated GeoTransform for the overview level.
        """
        # Base GeoTransform
        base_geo_transform = self.dataset.GetGeoTransform()

        # Base and overview dimensions
        band = self.dataset.GetRasterBand(1)
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
  
    def read_tiff_block_gdal(self, coord_center, block_size_x,block_size_y, overview_level):
        """
        Read a block of data from the map

        This function reads a block from the map, from a specific overview, were coord_center is in the middle

        Args:
            coord_center (tuple):  values (lon,lat)
            block_size (int): size of block, funciton will return square box
            overview_level (int) : zoom level on map (-1 base level, 0-max overviews in map ) 

        Returns:
             cv2 image , transform
        """
        # Get the GeoTransform for the overview level (either from overviews or base level)
        if overview_level == -1:
            # Use the base level GeoTransform (main dataset)
            overview_geo_transform = self.dataset.GetGeoTransform()
        else:
            # Use the GeoTransform of the overview level
            overview_geo_transform = self.get_overview_geotransform(overview_level)

        self.curent_center = coord_center
        self.current_zoom_level = overview_level
        # Calculate pixel coordinates from geographical coordinates
        lon, lat = coord_center
        x_pixel = int((lon - overview_geo_transform[0]) / overview_geo_transform[1])
        y_pixel = int((lat - overview_geo_transform[3]) / overview_geo_transform[5])

        # Calculate the block's top-left corner
        x_offset = max(x_pixel - block_size_x // 2, 0)
        y_offset = max(y_pixel - block_size_y // 2, 0)

        # Initialize a list to hold band data
        bands_data = []

        # Loop through all the bands in the dataset
        for band_index in range(1, self.dataset.RasterCount + 1):
            band = self.dataset.GetRasterBand(band_index)

            if overview_level == -1:
                # Read directly from the base level (dataset)
                block_data = band.ReadAsArray(x_offset, y_offset, block_size_x, block_size_y)
            else:
                # Get the overview (pyramid) level
                overview = band.GetOverview(overview_level)
                if overview is None:
                    raise ValueError(f"Overview level {overview_level} does not exist for band {band_index}.")

                # Ensure block size does not exceed the overview's dimensions
                x_block_size = min(block_size_x, overview.XSize - x_offset)
                y_block_size = min(block_size_y, overview.YSize - y_offset)
                self.current_block_X = x_block_size
                self.current_block_Y = y_block_size
                if x_block_size != block_size_x or y_block_size != block_size_y:
                    print("block size mismtach")
                    # continue
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

        # Update the transform and block origin for the new zoom level
        self.current_transform = overview_geo_transform    
        
        self.current_block_origin_x = x_offset# int((self.current_center[0] - overview_geo_transform[0]) / overview_geo_transform[1]) - self.block_size_x // 2
        self.current_block_origin_y = y_offset#int((self.current_center[1] - overview_geo_transform[3]) / overview_geo_transform[5]) - self.block_size_y // 2
        self.image = color_image
        self.block_size_x  = block_size_x
        self.block_size_y  = block_size_y
        return color_image, overview_geo_transform

    def drag(self, start_mouse_pos,end_mouse_pos):
        """
        Handle the drag event to update the displayed map.

        This function calculates the map movement based on the start 
        and end positions of the mouse drag and updates the map view.

        Args:
            start_mouse_pos (tuple): The starting position of the mouse drag in (x, y) coordinates.
            end_mouse_pos (tuple): The ending position of the mouse drag in (x, y) coordinates.

        Returns:
            cv2 image , transform
        """
        # Calculate pixel offset
        dx = end_mouse_pos[0] - start_mouse_pos[0]
        dy = end_mouse_pos[1] - start_mouse_pos[1]
        
        # Convert pixel offset to degrees using the transform
        lat_offset = dy * self.current_transform[1]  # Vertical resolution
        lon_offset = dx * self.current_transform[5]   # Horizontal resolution
        
        # Update the current center (current_center) based on offset
        self.current_center = (
            self.current_center[0] + lon_offset,  # Update longitude
            self.current_center[1] + lat_offset   # Update latitude
        )
        # Update center coordinates
        # self.curent_center = (current_center[0] + lat_offset, current_center[1] + lon_offset)
        
        
        # Load new portion of the map
        self.image, self.current_transform = self.read_tiff_block_gdal(self.current_center, self.block_size_x,self.block_size_y, self.current_zoom_level)
        return self.image, self.current_transform

    def coord_to_pixel_double(self,geo_transform, coord_lon_east, coord_lat_north):
        """
        Converts geographic coordinates to pixel coordinates.
        
        Parameters:
            geo_transform (tuple): The 6-element affine GeoTransform.
            coord_lon_east (float): The longitude (x) in geographic space.
            coord_lat_north (float): The latitude (y) in geographic space.

        Returns:
            tuple: The pixel (x, y) coordinates as a pair of floats.
        """
        # Invert the geotransform
        inv_geo_transform = gdal.InvGeoTransform(geo_transform)
        # if not inv_geo_transform[0]:
        #     raise RuntimeError("Could not invert GeoTransform")
        # inv_geo_transform = inv_geo_transform[1]

        # Apply the inverted geotransform
        pxx, pyy = gdal.ApplyGeoTransform(inv_geo_transform, coord_lon_east, coord_lat_north)
        return pxx, pyy 
    
    def latlon_to_pixel(self,lat, lon, transform=None):
        """
        Convert geographic coordinates (latitude, longitude) to pixel coordinates.

        Args:
            lat (float): Latitude.
            lon (float): Longitude.
            transform (tuple): Affine transform of the dataset.

        Returns:
            tuple: (pixel_x, pixel_y) corresponding to the geographic coordinates.
        """
        if transform == None:
            transform = self.current_transform
        return self.coord_to_pixel_double(transform,lon,lat)
    
        det = transform[1] * transform[5] - transform[2] * transform[4]
        if det == 0:
            raise ValueError("Transform matrix is singular, cannot compute pixel coordinates.")
        
        # Calculate the inverse transform
        inv_transform = (
            transform[5] / det,  # a'
            -transform[2] / det, # b'
            -transform[4] / det, # c'
            transform[1] / det,  # d'
        )
        
        # Apply inverse transform
        pixel_x = inv_transform[0] * (lon - transform[0]) + inv_transform[1] * (lat - transform[3])
        pixel_y = inv_transform[2] * (lon - transform[0]) + inv_transform[3] * (lat - transform[3])
        return pixel_x, pixel_y

    def pixel_to_latlon(self,pixel_x, pixel_y, transform,relative = False):
        """
        Convert pixel coordinates to geographic coordinates (latitude, longitude).

        Args:
            pixel_x (float): The x-coordinate (column) in pixel space.
            pixel_y (float): The y-coordinate (row) in pixel space.
            transform (tuple): Affine transform of the dataset.

        Returns:
            tuple: (longitude, latitude) corresponding to the pixel coordinates.
        """
        if relative:
            pixel_x = self.current_block_origin_x + pixel_x
            pixel_y = self.current_block_origin_y + pixel_y
        if transform == None:
            transform = self.current_transform

        lon = transform[0] + pixel_x * transform[1] + pixel_y * transform[2]
        lat = transform[3] + pixel_x * transform[4] + pixel_y * transform[5]
        return lon, lat
    
    def block_pixel_pos_lon_lat(self,offset_x,offset_y):
        x= self.current_block_origin_x + offset_x
        y= self.current_block_origin_y + offset_y
        return  self.pixel_to_latlon(x,y,self.current_transform)
        
    def zoom(self, wheel_flag, mouse_pos):
        prev_zoom_level = self.current_zoom_level
        max_zoom_level = self.num_overviews
        previous_center = self.current_center

        if wheel_flag > 0:  # Scroll up: Zoom in (higher resolution)
            self.current_zoom_level = max(-1, self.current_zoom_level - 1)
        else:  # Scroll down: Zoom out (lower resolution)
            self.current_zoom_level = min(max_zoom_level, self.current_zoom_level + 1)
        if prev_zoom_level == self.current_zoom_level:
            return
        # Convert the current center from lat/lon to pixels
        current_pixel = self.latlon_to_pixel(lat = self.current_center[1],lon=self.current_center[0],transform=self.current_transform)
        mouse_pixel = mouse_pos  # Assuming mouse_pos is already in pixel coordinates

        # Calculate the scale factor based on the zoom level change
        scale_factor = 2 ** (self.current_zoom_level - (self.current_zoom_level + (1 if wheel_flag > 0 else -1)))

        # Update the pixel coordinates while keeping the mouse position fixed
        new_pixel_x = current_pixel[0] + (mouse_pixel[0] - (self.block_size_x / 2)) * (1 - scale_factor)
        new_pixel_y = current_pixel[1] + (mouse_pixel[1] - (self.block_size_y / 2)) * (1 - scale_factor)

        # Convert the new pixel coordinates back to lat/lon
        new_center = self.pixel_to_latlon(new_pixel_x, new_pixel_y,self.current_transform)
        self.current_center = new_center

        # Read the new block
        self.image, self.current_transform =  self.read_tiff_block_gdal(self.current_center, self.block_size_x,self.block_size_y ,self.current_zoom_level)

      
   

    def get_dataset_bounds(self):
        """
        Get the geographic bounding box (top-left and bottom-right coordinates) of the entire dataset.

        Args:
            dataset (gdal.Dataset): The GDAL dataset.

        Returns:
            dict: A dictionary with the top-left (tl) and bottom-right (br) coordinates in (longitude, latitude).
        """
        # Get the affine transform of the dataset
        transform = self.dataset.GetGeoTransform()

        # Get the size of the dataset in pixels
        width = self.dataset.RasterXSize
        height = self.dataset.RasterYSize

        # Top-left corner (0, 0)
        tl_lon, tl_lat = self.pixel_to_latlon(0, 0, transform)

        # Bottom-right corner (width, height)
        br_lon, br_lat = self.pixel_to_latlon(width, height, transform)

        return {"tl": (tl_lon, tl_lat), "br": (br_lon, br_lat)}
    
    def is_coord_visiable(self,lon,lat):
        pixel_x, pixel_y = self.latlon_to_pixel(lat=lat,lon=lon)
        return self.is_on_map((pixel_x,pixel_y),False)
    
    def is_on_map(self,mouse_pos,relative = True):
        current_x_end = self.current_block_origin_x + self.block_size_x
        current_y_end = self.current_block_origin_y + self.block_size_y
        pixel_x = mouse_pos[0] if relative == False else self.current_block_origin_x + mouse_pos[0]
        pixel_y = mouse_pos[1] if relative == False else self.current_block_origin_y + mouse_pos[1]
        if self.current_block_origin_x <= pixel_x <= current_x_end and \
            self.current_block_origin_y <= pixel_y <= current_y_end:
            return True
        else:
            return False
    def get_relative_pixel_positon(self,lon,lat):
        pixel_x, pixel_y = self.latlon_to_pixel(lat=lat,lon=lon,transform=self.current_transform)
        x = pixel_x - self.current_block_origin_x 
        y = pixel_y - self.current_block_origin_y 
        return x,y 
    def is_coord_seen(self,lon,lat):
        return