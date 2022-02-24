#!/usr/bin/env python

import time
import sys
import signal

import VL53L1X


print("""distance.py

Display the distance read from the sensor.

Press Ctrl+C to exit.

""")


"""
Open and start the VL53L1X ranging sensor for each channel of the TCA9548A
"""
tof = VL53L1X.VL53L1X(i2c_bus=8, i2c_address=0x29, tca9548a_num=4, tca9548a_addr=0x70)
time.sleep(1)
tof.open()
tof.start_ranging(1)  # Start ranging, 1 = Short Range, 2 = Medium Range, 3 = Long Range
tof.set_timing(66000, 70)

def exit_handler(signal, frame):
    global running
    running = False
    tof.stop_ranging()  
    print()
    sys.exit(0)


running = True
signal.signal(signal.SIGINT, exit_handler)
k = 1

start = time.time()
while time.time() - start < 2:
    k += 1
    if k % 4 == 1:
        time1 = time.time()
        distance_in_mm = tof.get_distance()
        print("Time 1: ", time.time() - time1)
        print("Sensor 1 distance: {}mm".format(distance_in_mm))
    print("-----------------------------------")
    time.sleep(0.01)
