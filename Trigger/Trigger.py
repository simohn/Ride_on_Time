# Code written by Simon Schauppenlehner
# Last change: 22.06.2019

from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2 as cv
import numpy as np
import time
import os


# ---------global variables---------
res_x_full = 1280
res_y_full = res_x_full

res_x_small = 320
res_y_small = res_x_small

cor_x_full = res_x_full/res_x_small
cor_y_full = res_y_full/res_y_small

cnt = 0

# ---------Mouse Callback---------
active = True
save = False


def mouse_callback(event, x, y, flags, params):
    global active
    global save

    # End the program
    if event == cv.EVENT_RBUTTONDOWN:
        active = False

    # Activate/Deactivate save mode
    if event == cv.EVENT_LBUTTONDOWN:
        if save:
            save = False
            print("Save deactivated")
        else:
            save = True
            print("Save activated")


# Detects motion in the stream of images
# Stores an image of the rider when crossing the finish line
def motion_detector():
    # ---------Global Variables---------
    global active
    global save
    global res_x_full
    global res_y_full
    global res_x_small
    global res_y_small
    global cor_x_full
    global cor_y_full
    global cnt

    # ---------Parameter---------
    res_x_img = 640
    res_y_img = res_x_img
    area_min = 1000
    area_max = 30000

    # ---------Initialization---------
    camera = PiCamera()
    camera.resolution = (res_x_full, res_y_full)
    camera.rotation = 180

    camera_array = PiRGBArray(camera)

    print("[INFO] warming up...")
    time.sleep(1)

    filter_bgsub = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=3, detectShadows=True)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

    mask_empty = np.zeros((res_x_small, res_y_small, 1), np.uint8)
    mask_out = np.zeros((res_x_small, res_y_small, 1), np.uint8)

    cv.namedWindow("Mask")
    cv.setMouseCallback("Mask", mouse_callback)
    cv.imshow("Mask", mask_out)

    x = 0
    y = 0
    w = 0
    h = 0

    # ---------Process---------
    for f in camera.capture_continuous(camera_array, format="bgr", use_video_port=True):
        # Read image
        frame_org = f.array
        frame = cv.resize(frame_org, (res_x_small, res_y_small))

        # Calculate mask
        mask = filter_bgsub.apply(frame)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        rider_has_finished, mask = cv.threshold(mask, 200, 255, cv.THRESH_BINARY)

        # Find contours
        contours, hierachy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        areas = [cv.contourArea(c) for c in contours]

        if np.size(areas) != 0:
            max_index = np.argmax(areas)

            # Plausibility check
            if area_min < areas[int(max_index)] < area_max:
                contour_max = contours[max_index]

                # Geometric data of the bounding rectangular
                x, y, w, h = cv.boundingRect(contour_max)

            # Check if rider has crossed the finish line
            rider_has_finished, finish_line_a, finish_line_b = check_finish(x=x, y=y, w=w, h=h)

            if rider_has_finished:

                # Convert data (higher resolution)
                x_full = x * cor_x_full
                y_full = y * cor_y_full
                w_full = w * cor_x_full
                h_full = h * cor_y_full

                # Calculate geometric data for standardized bounding rectangular
                c_x = x_full + (w_full + 1) / 2.0
                c_y = y_full + (h_full + 1) / 2.0

                if c_x > res_x_full / 2:
                    if c_x + res_x_img / 2 <= res_x_full - 1:
                        d_x = int(c_x - res_x_img / 2)
                        e_x = d_x + res_x_img - 1
                    else:
                        d_x = res_x_full - res_x_img
                        e_x = res_x_full - 1
                else:
                    if c_x - res_x_img / 2 >= 0:
                        d_x = int(c_x - res_x_img / 2)
                        e_x = d_x + res_x_img - 1
                    else:
                        d_x = 0
                        e_x = res_x_img - 1

                if c_y > res_y_full / 2:
                    if c_y + res_y_img / 2 <= res_y_full - 1:
                        d_y = int(c_y - res_y_img / 2)
                        e_y = d_y + res_y_img - 1
                    else:
                        d_y = res_y_full - res_y_img
                        e_y = res_y_full - 1
                else:
                    if c_y - res_y_img / 2 >= 0:
                        d_y = int(c_y - res_y_img / 2)
                        e_y = d_y + res_y_img - 1
                    else:
                        d_y = 0
                        e_y = res_y_img - 1

                # Enter session ID manually
                session = 0
                if save:
                    cv.imwrite(os.path.join("Rider", "RID_" + str(session) + "_" + str(cnt) + ".jpg"),
                               frame_org[d_y:e_y+1, d_x:e_x+1, :])
                    cnt = cnt+1

                cv.imshow("Biker", frame_org[d_y:e_y+1, d_x:e_x+1, :])
                cv.waitKey(1)

            # Draw bounding rectangular
            cv.rectangle(mask, (x, y), (x+w, y+h), 255, 2)
            # cv.rectangle(mask, (int(d_x/4), int(d_y/4)), (int(e_x/4), int(e_y/4)), 200, 2)

            mask_out = mask

        else:
            mask_out = mask_empty
            rider_has_finished, finish_line_a, finish_line_b = check_finish(False)

        # Draw finish line
        cv.line(mask_out, finish_line_a, finish_line_b, 200, 2)

        # Display mask
        cv.imshow("Mask", mask_out)
        cv.waitKey(1)

        # Resize the stream to zero
        camera_array.truncate(0)

        if not active:
            break

    cv.destroyAllWindows()


# -----------static variables-------------
d_old = 0
state_CF = "no_motion"


# Checks if the rider has crossed the finish line
# Return values:
#   rider_has_finished ... True when rider has crossed the finish line
#   End points of the finish line
def check_finish(motion_detected=True, x=0, y=0, w=0, h=0):
    global res_x_full
    global res_y_full
    global cor_x_full
    global cor_y_full
    global d_old
    global state_CF

    rider_has_finished = False
    # Finish on left side (out of the camera perspective)
    finish_ls = False

    # -----------------------
    # |    |       |     .
    # |    |       by  .
    # |    |       | .
    # |----|---bx--B
    # |    |      .
    # |    ay   .
    # |    |  .
    # |    |.
    # |-ax-A
    # |  .
    # |.
    # -----------------------

    # End points of the finish line
    a_x = int(res_x_full/2)
    a_y = int(0)
    b_x = int(res_x_full/2)
    b_y = int(res_y_full-1)

    ab_x = b_x-a_x
    ab_y = b_y-a_y

    if ab_x == 0 and ab_y == 0:
        print("Error: Finish line")
        a_x = int(res_x_full/2)
        a_y = int(0)
        b_x = int(res_x_full/2)
        b_y = int(res_y_full-1)

        ab_x = b_x-a_x
        ab_y = b_y-a_y

    c_x = (x+w)*cor_x_full
    c_y = (y+h)*cor_y_full

    # Calculate distance to finish line
    # d is positive, when rider has crossed the finish line/has finished
    if ab_x == 0:
        d = (a_x-c_x)/ab_y
    elif ab_y == 0:
        d = (c_y-a_y)/ab_x
    else:
        d = (1/(ab_x**2 + ab_y**2))*((c_y-a_y)*ab_x + (a_x-c_x)*ab_y)

    if not finish_ls:
        d = -d

    # State machine
    if state_CF == "no_motion":
        if motion_detected and d < 0:
            d_old = d
            state_CF = "motion_detected"

    elif state_CF == "motion_detected":
        if d > 0:
            if d > d_old:
                rider_has_finished = True
                state_CF = "wait_for_next"
        d_old = d

    elif state_CF == "wait_for_next":
        if not motion_detected:
            state_CF = "no_motion"
        elif d < 0:
            d_old = d
            state_CF = "motion_detected"

    return rider_has_finished, (int(a_x/cor_x_full), int(a_y/cor_x_full)), (int(b_x/cor_x_full), int(b_y/cor_x_full))


if __name__ == "__main__":
    motion_detector()
