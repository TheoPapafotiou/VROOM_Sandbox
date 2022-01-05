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
tof1 = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29, tca9548a_num=2, tca9548a_addr=0x70)
tof2 = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29, tca9548a_num=4, tca9548a_addr=0x70)
tof3 = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29, tca9548a_num=6, tca9548a_addr=0x70)
tof1.open()
tof1.start_ranging(1)  # Start ranging, 1 = Short Range, 2 = Medium Range, 3 = Long Range
tof1.set_timing(66000, 70)
tof2.open()
tof2.start_ranging(1)  # Start ranging, 1 = Short Range, 2 = Medium Range, 3 = Long Range
tof2.set_timing(66000, 70)
tof3.open()
tof3.start_ranging(1)  # Start ranging, 1 = Short Range, 2 = Medium Range, 3 = Long Range
tof3.set_timing(66000, 70)

def exit_handler(signal, frame):
    global running
    running = False
    tof1.stop_ranging()
    tof2.stop_ranging()
    tof3.stop_ranging()
    print()
    sys.exit(0)


running = True
signal.signal(signal.SIGINT, exit_handler)
k = 1

while running:
    k += 1
    if k % 3 == 1:
        time1 = time.time()
        distance_in_mm = tof1.get_distance()
        print("Time 1: ", time.time() - time1)
        print("Sensor 1 distance: {}mm".format(distance_in_mm))
    elif k % 3 == 2:
        time2 = time.time()
        distance_in_mm = tof2.get_distance()
        print("Time 2: ", time.time() - time2)
        print("Sensor 2 distance: {}mm".format(distance_in_mm))
    elif k % 3 == 0:
        time3 = time.time()
        distance_in_mm = tof3.get_distance()
        print("Time 3: ", time.time() - time3)
        print("Sensor 3 distance: {}mm".format(distance_in_mm))
    print("-----------------------------------")
    time.sleep(0.01)
