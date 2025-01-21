from MapReader import GdalMapReader
import cv2


def mouse_callback(event, x, y, flags, param):
    """
    Handles mouse events for dragging and updating the map.

    Args:
        event: The mouse event type (e.g., EVENT_LBUTTONDOWN).
        x, y: The mouse position.
        flags: Additional flags for the event.
        param: Additional parameters passed to the callback (e.g., GdalMapReader instance).
    """
    gdal_reader = param["reader"]  # Access the GdalMapReader instance
    state = param["state"]  # Access shared state like mouse positions

    if event == cv2.EVENT_LBUTTONDOWN:
        # Mouse button pressed
        state["mouse_pressed"] = True
        state["start_mouse_pos"] = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if state["mouse_pressed"]:
            # end_mouse_pos = (x, y)
            # print ('start' ,state["start_mouse_pos"],'end', end_mouse_pos)
            # # # Call the drag method to update the map view
            # gdal_reader.drag(state["start_mouse_pos"], end_mouse_pos)
            # Optional: Show dragging indicator
            pass
        lon,lat = gdal_reader.block_pixel_pos_lon_lat(x,y)
        status_text = f"(Lon:{lon:.6f} Lat:{lat:.6f}"
        print(status_text)
        # Update the status bar
        # cv2.displayStatusBar("Map View", status_text, cv2.WINDOW_NORMAL)

    elif event == cv2.EVENT_LBUTTONUP:
        # Mouse button released
        state["mouse_pressed"] = False
        end_mouse_pos = (x, y)
        # Call the drag method to update the map view
        gdal_reader.drag(state["start_mouse_pos"], end_mouse_pos)

    elif event == cv2.EVENT_MOUSEWHEEL:
        # with rasterio.open(TIFF_PATH) as src:
        #     max_zoom_level = len(src.overviews(1))  # Number of overviews available
       gdal_reader.zoom(flags,(x,y))

def main():
    
    tiff_path =  "my_ortho_file.tif" 
    start_coords = (35.178471, 31.789318)  # Longitude, Latitude
    block_size_x = 900
    block_size_y = 500
    current_zoom_level = 5  # Change this to see the effect of different levels
    gdal_reader = GdalMapReader()
    gdal_reader.open_file(tiff_path)

    image,transform = gdal_reader.read_tiff_block_gdal(start_coords, block_size_x,block_size_y, current_zoom_level)
    
     # Shared state for mouse events
    mouse_state = {"mouse_pressed": False, "start_mouse_pos": (0, 0)}

    # # Set up the OpenCV window and mouse callback
    cv2.namedWindow("Map View")
    cv2.setMouseCallback("Map View", mouse_callback, {"reader": gdal_reader, "state": mouse_state})
    
    while True:
        # Display the current map view
        cv2.imshow("Map View", gdal_reader.get_image())
        
        # Break the loop on 'q' key press
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()