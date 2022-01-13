
# Horizontal Lines Detection Documentation
### Brief
The usage of this class is to detect only the horizontal lines and to return some information for them in order to trigger some of the special functions like the roundabout and the intersection navigation.
### Usage
Create an instance of the class DetectHorizontal and and initialize pass the angument of the default mask file of your choice. This will be the mask that will be used while the normal detection is enabled. \

`det = DetectHorizontal(mask_filename="default_mask_real.json")`

Then, run the function `detection` with the arguments:

   `image` : the frame that you want to detect the horizontal lines
   `stop_signal_at=300` (optional) : in which height of the image you want the closer line to be to return a stop signal
   `min_line_length=100` (optional) : the minimum line length that the function will detect
   `reset=False` : set to True when you want to reset its function to the normal (after the prcision detection)


The function will return a dictionary that will include the info:
`
info_dict = {
            "detection_dist_intensity": -1,  # 1 to <screen_fractions> metric of how close is the detected line
            "avg_y": -1, # the average height value detected
            "min_y": -1, # the minimum height value detected
            "lines_found": 0, # the number of lines found
            "stop_signal": 0  # 0 if not stop_signal, 1 if stop_signal
        }
`
More return information can be added if needed.

