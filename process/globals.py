import numpy as np

LIDAR_MAX_HEIGHT = 2
LIDAR_MIN_HEIGHT = -2

RES = (1.33, 0.2) #(vertical, horizontal)
VFOV = (-30.67, 10.67)

RES_RAD = np.array(RES) * (np.pi/180)
X_MIN = -360.0 / RES[1] / 2
Y_MIN = VFOV[0] / RES[0]
X_MAX = int(360.0 / RES[1])
Y_MAX = int(abs(VFOV[0] - VFOV[1]) / RES[0])

CAM_IMG_TOP = 430
CAM_IMG_BOTTOM = 942
