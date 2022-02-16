#!/usr/bin/env python

import VL53L1X
import Jetson.GPIO as GPIO
import time

addr_current = 0x29
addr_desired = [0x33, 0x34, 0x35, 0x36]
gpio_pins = [21, 22, 23, 24]
time_sleep = 0.5


GPIO.setmode(GPIO.BOARD)
GPIO.setup(gpio_pins, GPIO.OUT)
for i in range(4):

    GPIO.output(gpio_pins[i], GPIO.HIGH)
    time.sleep(time_sleep)

    print("""
    Current address: {:02x}
    Desired address: {:02x}
    """.format(addr_current, addr_desired[i]))
    tof = VL53L1X.VL53L1X(i2c_bus=8, i2c_address=addr_current)
    tof.open()
    tof.change_address(addr_desired[i])
    tof.close()
    time.sleep(time_sleep)

GPIO.cleanup()