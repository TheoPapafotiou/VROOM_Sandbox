class LaneKeeping:
    
    def __init__(self):
        self.angles = [22, 3, 6, -1]
        self.counter = -1

    def laneKeeping(self, img):
        
        self.counter += 1

        return self.angles[self.counter]