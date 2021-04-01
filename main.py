from djitellopy import Tello
from utils import *
import cv2, time, sys, math, socket
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv

# Reference:  https://www.youtube.com/watch?v=wlT_0fhGrGg&t=45s

# Define Tag
#marker_size = 13.6 #cm
#marker_size = 16.4
marker_size = 14.1
# Load Aruco Dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()

# Define the frame size
width = 360
height = 240

def init_drone():
    tello = Tello()
    tello.connect()
    tello.for_back_velocity = 0
    tello.left_right_velocity = 0
    tello.up_down_velocity = 0
    tello.yaw_velocity = 0
    tello.speed = 0
    if tello.get_battery() < 15:
        return "Low Battery: Go recharge"
    tello.streamoff()
    tello.streamon()
    return tello

def get_tello_frame(drone, width = 360, height = 240): # width and height for image resizing
    tello_frame = drone.get_frame_read()
    image = tello_frame.frame
    processing_img = cv2.resize(image, (width, height))
    return processing_img

# Init the drone
drone = init_drone()
font = cv2.FONT_HERSHEY_PLAIN

# From online:
camera_matrix = np.array([[921.170702, 0.000000, 459.904354], [0.000000, 919.018377, 351.238301], [0.000000, 0.000000, 1.000000]])
distortion = np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])

x_values = []
y_values = []
z_values = []
lr_values = []
fb_values = []
ud_values = []
height_values = []
xspeed = []
yspeed = []
zspeed = []
roll_values = []
pitch_values = []
yaw_values = []
xerror_values = []
yerror_values = []
disterror_values = []
log_dict = {'x': x_values, 'y': y_values, 'z': z_values, 'lr': lr_values, 'fb': fb_values, 'ud': ud_values, 'h': height_values,
            'xs': xspeed, 'ys': yspeed, 'zs': zspeed, 'roll': roll_values, 'pitch': pitch_values, 'yaw': yaw_values,
            'xe': xerror_values, 'ye': yerror_values, 'diste': disterror_values}
time.sleep(2)
start_time = time.time()
drone.takeoff()
# Processing the frame
first_contact = None

#===========control loop starts=================================================================================
while (time.time() - start_time < 40):
    frame = get_tello_frame(drone, width, height)
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(image=gray_img, dictionary=aruco_dict, parameters=parameters)

    log_dict['h'].append(drone.get_height())
    log_dict['xs'].append(drone.get_speed_x())
    log_dict['ys'].append(drone.get_speed_y())
    log_dict['zs'].append(drone.get_speed_z())
    log_dict['roll'].append(drone.get_roll())
    log_dict['pitch'].append(drone.get_pitch())
    log_dict['yaw'].append(drone.get_yaw())

    if np.all(ids != None):
        first_contact = len(log_dict['x'])
        ret = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, distortion)
        center = np.mean(corners[0][0], axis=0)
        xerror = width/2 - center[0]
        yerror = height/2 - center[1]
        rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]

        # Collecting Data:
        log_dict['x'].append(tvec[0])
        log_dict['y'].append(tvec[1])
        log_dict['z'].append(tvec[2])
        dist = np.sqrt(tvec[0] ** 2 + tvec[1] ** 2 + (tvec[2]) ** 2)
        disterror = dist - 150
        # INSERT CONTROL HERE
        #error = -(abs(y) - 70)
        #integratedError = integratedError + error
        kpx = 0.2
        kpy = 0.4
        kpz = 0.1
        control_LR = -1 * kpx * xerror
        control_FB = kpz * disterror
        control_UD = kpy * yerror
        control_LR = int(np.clip(control_LR,-100,100))
        control_FB = int(np.clip(control_FB,-100,100))
        control_UD = int(np.clip(control_UD,-100,100))

        # code for task 2: flying through a hoop
        #if disterror <= 250:
            #if drone.send_rc_control:
                #drone.send_rc_control(control_LR, 60, control_UD, 0)
            #log_dict['lr'].append(control_LR)
            #log_dict['fb'].append(60)
            #log_dict['ud'].append(control_UD)
            #break
        if drone.send_rc_control:
            drone.send_rc_control(control_LR, control_FB, control_UD, 0)

        log_dict['lr'].append(control_LR)
        log_dict['fb'].append(control_FB)
        log_dict['ud'].append(control_UD)
        log_dict['xe'].append(xerror)
        log_dict['ye'].append(yerror)
        log_dict['diste'].append(disterror)

        # Draw CV2
        aruco.drawDetectedMarkers(frame, corners)
        aruco.drawAxis(frame, camera_matrix, distortion, rvec, tvec, 10)
        str_position = "MARKER Position x=%4.0f y=%4.0f z=%4.0f"%(tvec[0], tvec[1], tvec[2])
        cv2.putText(frame, str_position, (0,100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    else:
        if first_contact is not None:
            print("no aruco but previously found aruco")
            if drone.send_rc_control:
                drone.send_rc_control(0, 0, -10, 0)
            log_dict['ud'].append(-10)
        else:
            print("no aruco detected")
            if drone.send_rc_control:
                drone.send_rc_control(0, 0, 15, 0)
            log_dict['ud'].append(15)
        log_dict['x'].append(0)
        log_dict['y'].append(0)
        log_dict['z'].append(0)
        log_dict['lr'].append(0)
        log_dict['fb'].append(0)
        log_dict['xe'].append(0)
        log_dict['ye'].append(0)
        log_dict['diste'].append(0)

    cv2.imshow('Frame', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'): # waits for 1 milisecond and can break with q
        cv2.destroyAllWindows()
        break

Tello.land(drone)

#===========plotting stats====================================================================================
time = list(range(1, len(log_dict['x']) + 1))
time_data = [element * 0.009 for element in time]

plt.figure()
plt.plot(time_data, [ele for ele in log_dict['x']])
plt.xlabel('Time (sec)')
plt.ylabel('Translation in the x-axis (cm)')
plt.title('X Translation of the Marker in Camera Frame')
plt.savefig("x_trans_1.png")

plt.figure()
plt.plot(time_data, [ele for ele in log_dict['y']])
plt.xlabel('Time (sec)')
plt.ylabel('Translation in the y-axis (cm)')
plt.title('Y Translation of the Marker in Camera Frame')
plt.savefig("y_trans_1.png")

plt.figure()
plt.plot(time_data, log_dict['z'])
plt.xlabel('Time (sec)')
plt.ylabel('Translation in the z-axis (cm)')
plt.title('Z Translation of the Marker in Camera Frame')
plt.savefig("z_trans_1.png")

plt.figure()
dist_calc = np.sqrt(np.array(log_dict['x'])**2 + np.array(log_dict['z'])**2 + np.array(log_dict['z'])**2)
plt.plot(time_data, dist_calc)
plt.xlabel('Time (sec)')
plt.ylabel('Distance (cm)')
plt.title('Distance (uncalibrated) between aruco tag and drone over time')
plt.savefig("dist_1.png")

plt.figure()
plt.plot(time_data, log_dict['lr'])
plt.xlabel('Time (sec)')
plt.ylabel('LR control')
plt.title('LR rc control input over time')
plt.savefig("lr_1.png")

plt.figure()
plt.plot(time_data, log_dict['fb'])
plt.xlabel('Time (sec)')
plt.ylabel('FB control')
plt.title('FB rc control input over time')
plt.savefig("fb_1.png")

plt.figure()
plt.plot(time_data, log_dict['ud'])
plt.xlabel('Time (sec)')
plt.ylabel('UD control')
plt.title('UD rc control input over time')
plt.savefig("ud_1.png")

plt.figure()
plt.plot(time_data, log_dict['h'])
plt.xlabel('Time (sec)')
plt.ylabel('Height of drone (cm)')
plt.title('Height of drone over time')
plt.savefig("height_1.png")

plt.figure()
plt.plot(time_data, log_dict['xs'])
plt.plot(time_data, log_dict['ys'])
plt.plot(time_data, log_dict['zs'])
plt.xlabel('Time (sec)')
plt.ylabel('Speed of drone in XYZ (cm/s)')
plt.legend(['x speed', 'y speed', 'z speed'])
plt.title('Speed of drone in XYZ over time')
plt.savefig("speed_1.png")

plt.figure()
plt.plot(time_data, log_dict['roll'])
plt.plot(time_data, log_dict['pitch'])
plt.plot(time_data, log_dict['yaw'])
plt.xlabel('Time (sec)')
plt.ylabel('Roll, pitch, yaw of drone (deg)')
plt.legend(['roll', 'pitch', 'yaw'])
plt.title('roll, pitch, yaw of drone over time')
plt.savefig("rpy_1.png")

plt.figure()
plt.plot(time_data, log_dict['xe'])
plt.xlabel('Time (sec)')
plt.ylabel('Error in x direction (left/right) (pixels)')
plt.title('Error in x direction over time')
plt.savefig("xe_1.png")

plt.figure()
plt.plot(time_data, log_dict['ye'])
plt.xlabel('Time (sec)')
plt.ylabel('Error in y direction (up/down) (pixels)')
plt.title('Error in y direction over time')
plt.savefig("ye_1.png")

plt.figure()
plt.plot(time_data, log_dict['diste'])
plt.xlabel('Time (sec)')
plt.ylabel('Error in distance to the tag (forward/backward) (cm)')
plt.title('Error in distance over time')
plt.savefig("diste_1.png")
