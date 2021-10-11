
#!/usr/bin/env python
# https://github.com/pimoroni/vl53l1x-python/tree/master/examples

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
tof = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29)
tof.open()

# Optionally set an explicit timing budget
# These values are measurement time in microseconds,
# and inter-measurement time in milliseconds.
# If you uncomment the line below to set a budget you
# should use `tof.start_ranging(0)`
tof.set_timing(66000, 70)

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

tof.set_user_roi(roi)

tof.start_ranging(0)  # Start ranging
                      # 0 = Unchanged
                      # 1 = Short Range
                      # 2 = Medium Range
                      # 3 = Long Range

running = True


def exit_handler(signal, frame):
    global running
    running = False
    tof.stop_ranging()
    print()
    sys.exit(0)


# Attach a signal handler to catch SIGINT (Ctrl+C) and exit gracefully
signal.signal(signal.SIGINT, exit_handler)

while running:
    start = time.time()
    distance_in_mm = tof.get_distance()
    print("Time: {}s".format(time.time()-start))
    print("Distance: {}mm".format(distance_in_mm))
    time.sleep(0.002)