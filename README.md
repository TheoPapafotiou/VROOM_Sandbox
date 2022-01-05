## Roundabout

Hi reader,

The script contains **8** functions in total of the class `CenterOfRoundanout`.(This name will propably change in the future)

--------------------

#### preprocessing
- Saves a copy of the orignal image to ***self.image_copy***
- Changes the image from BGR to RGB as opencv reads images in the BGR format.
- Applies a color threshold to detect only white lines

----------------------------

#### findAndDrawContours
- Finding all possible contours 
- Drawing with **Red** colour all the detected contours, on ***self.image***, so they can be distinguishable

---------------------------

#### get_contour_centers
- Finding all the possible centers of the detected countours from `findAndDrawContours`

----------------------------

#### drawPoints
- Draw these centers with **Blue** colour on ***self.image***

----------------------------------

#### findTheCenter
- Finding the neareset center from the detected centers to the **bottom left edge** using euclidean method

------------------------

#### findTheLine
- Finding the first white pixel at the right side of the nearest detected center from `findTheLine` 
- Draw this point with **Green** colour on ***self.image***
- Draw a line to the point detected with **White** colour on ***self.image_copy***

-------------------------------------

#### findTheAngle
- prints the angle of the created line

-----------------------------------

#### centerOfRoundabout
- Runs all the above functions
- In case of using this class only this function needs to be run.

> Regarding any question, you can contact me on Discord:)