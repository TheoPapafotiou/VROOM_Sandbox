# VROOM_Sandbox

# Navigation Intersection 


* houghLine.py

this file contains the first method of corner's recognition with the aid of Hough Lines Detection 

* masksForIntersection.py

This file contains the masks for each case/path the car follows

- Straight path
    - it contains the following files:
    straight_i.png, where i is from 1 to 4,
    straight_simulation.mp4, video from the simulation
    intersection_straight.mp4, video from realife

- Small right turn 
    - it contains the following files:
    rsmallj.png, where j is from 0 to 6,
    rsmal_simulation.mp4, video taken from simulation
    intersection_small_right.mp4, video realife

- Big turn 
    - it contains the files:
    ... (to be completed)

* multiRansac.py

this file contains the second method : applying ransac, finding the best two lines that are not parallels, calculating the intersection of these lines

* edge_detection.py

this file contains the straight route navigation solution (Harris Corner Detection, Ransac) 
