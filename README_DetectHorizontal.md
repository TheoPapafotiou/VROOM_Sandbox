
# Horizontal Lines Detection Documentation
### Brief

One very important aspect of the car's navigation is the detection of the horizontal lines on the road, in order for the vehicle to stop accurately and safely before them. The method used to detect these lines is the same as that used to detect the lanes on the lane detection implementation, the houghlines function of the OpenCV library. At first, all the lines, above a max pixel value, are detected. Then, there are a few methods that eliminate the not horizontal lines. 
        
Some of them are the elimination of some lines using their slope, the use of two masking states and the elimination of the non centred lines. The slope is calculated in order to exclude the non horizontal lines. Then, in order to filter the potential noise, there are two detection states; the normal state and the precision state. When the car is close to a horizontal line, the precision state is activated and the image mask gets smaller in order to filter potential noise. Finally, we calculate the average x-value of the non excluded lines and if its not centred, it means that the horizontal lines that were calculated are not the lines that we are looking for, so we exclude them as well. 

These are some of the methods used to detect the horizontal lines and certainly, the exisiting methods will be improved and more will be added according to the needs that will come into view while testing.  

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

