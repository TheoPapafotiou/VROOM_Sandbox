#Markov decision process (MDP)
import math

from matplotlib.pyplot import xcorr 
# finding the best move with -> bayesian optimization 

# step1: path planning -> input
# step2 : navigation .. knownn: static environment 
#          unknown: the behavopr of other road users (MDP)
# step3: motion planning: speed, steering angle, lane-changning...
# step4: vehicle control execute the path 

# we estimate the vehicle state 

# Convolutional neural network (CNN): 
#  filter matrix: input image-> yi = SUM(wi,j*x) + bi

def execute_manuever_left():
    return angle, speed

def execute_manuever_right():
    return angle, speed

def execute_straight():
    return angle, speed

if gps is not None:
    #method with gps finding the (x,y) spot and navigate 

else:
   # physics and standard method values -> fixed -> kinematics , yaw 
   

    if trajectory == type1:
        execute_manuever_left()
    elif trajectory ==  type2:
        execute_manuever_right()
    else :
        execute_straight() 


def move(u,theta,dt,a,w):
    x += (u * math.cos(theta) - a * math.sin(theta) * w) *dt
    y += (u*math.sin(theta) + a*math.cos(theta)*w) *dt
    theta += w * dt

def follow_path():
    target = path[waypoint]
    delta_x = target[0] - x
    delta_y = target[1] - y
    u = delta_x * math.cos(theta) + delta_y * math.sin(theta) 
    w = (-1/a) * math.sin(theta) + (1/a)*math.cos(theta) 

