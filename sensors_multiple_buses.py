import time
import sys
import signal

import VL53L1X


print("""distance.py
Display the distance read from the sensor.
Uses the "Short Range" timing budget by default.
Press Ctrl+C to exit.
""")


# Open and start the VL53L1X sensor.
# If you've previously used change-address.py then you
# should use the new i2c address here.
# If you're using a software i2c bus (ie: HyperPixel4) then
# you should `ls /dev/i2c-*` and use the relevant bus number.
tof1 = VL53L1X.VL53L1X(i2c_bus=4, i2c_address=0x29)
tof1.open()
tof1.set_timing(66000, 70)

tof2 = VL53L1X.VL53L1X(i2c_bus=5, i2c_address=0x29)
tof2.open()
tof2.set_timing(66000, 70)

tof3 = VL53L1X.VL53L1X(i2c_bus=6, i2c_address=0x29)
tof3.open()
tof3.set_timing(66000, 70)

tof4 = VL53L1X.VL53L1X(i2c_bus=7, i2c_address=0x29)
tof4.open()
tof4.set_timing(66000, 70)

# Optionally set an explicit timing budget
# These values are measurement time in microseconds,
# and inter-measurement time in milliseconds.
# If you uncomment the line below to set a budget you
# should use `tof.start_ranging(0)`

def scan(type="w"):
    if type == "w":
        # Wide scan forward ~30deg angle
        print("Scan: wide")
        return VL53L1X.VL53L1xUserRoi(0, 15, 15, 0)
    elif type == "c":
        # Focused scan forward
        print("Scan: center")
        return VL53L1X.VL53L1xUserRoi(6, 9, 9, 6)
    elif type == "t":
        # Focused scan top
        print("Scan: top")
        return VL53L1X.VL53L1xUserRoi(6, 15, 9, 12)
    elif type == "b":
        # Focused scan bottom
        print("Scan: bottom")
        return VL53L1X.VL53L1xUserRoi(6, 3, 9, 0)
    elif type == "l":
        # Focused scan left
        print("Scan: left")
        return VL53L1X.VL53L1xUserRoi(0, 9, 3, 6)
    elif type == "r":
        # Focused scan right
        print("Scan: right")
        return VL53L1X.VL53L1xUserRoi(12, 9, 15, 6)
    else:
        print("Scan: wide (default)")
        return VL53L1X.VL53L1xUserRoi(0, 15, 15, 0)

roi = scan("default")

tof1.set_user_roi(roi)
tof2.set_user_roi(roi)
tof3.set_user_roi(roi)
tof4.set_user_roi(roi)

tof1.start_ranging(0)  # Start ranging
tof2.start_ranging(0)  # 0 = Unchanged
tof3.start_ranging(0)  # 1 = Short Range
tof4.start_ranging(0)  # 2 = Medium Range
                      # 3 = Long Range

running = True


def exit_handler(signal, frame):
    global running
    running = False
    tof1.stop_ranging()
    tof2.stop_ranging()
    tof3.stop_ranging()
    tof4.stop_ranging()
    print()
    sys.exit(0)


# Attach a signal handler to catch SIGINT (Ctrl+C) and exit gracefully
signal.signal(signal.SIGINT, exit_handler)

while running:
    start1 = time.time()
    distance_in_mm = tof1.get_distance()
    print("Time: {}s".format(time.time()-start1))
    print("Distance: {}mm".format(distance_in_mm))
    
    start2 = time.time()
    distance_in_mm = tof2.get_distance()
    print("Time: {}s".format(time.time()-start2))
    print("Distance: {}mm".format(distance_in_mm))
    
    start3 = time.time()
    distance_in_mm = tof3.get_distance()
    print("Time: {}s".format(time.time()-start3))
    print("Distance: {}mm".format(distance_in_mm))
    
    start4 = time.time()
    distance_in_mm = tof4.get_distance()
    print("Time: {}s".format(time.time()-start4))
    print("Distance: {}mm".format(distance_in_mm))
    
    print('--------------------------')
    time.sleep(0.001)
