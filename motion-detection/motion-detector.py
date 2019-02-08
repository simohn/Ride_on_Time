from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2 as cv
import numpy as np
import time
import os

#---------global variables---------
res_x_full = 1280
res_y_full = res_x_full

res_x_small = 320
res_y_small = res_x_small

cor_x_full = res_x_full/res_x_small
cor_y_full = res_y_full/res_y_small

cnt = 0

t_shot_photo = 0
timeout_over = True

# ---------Mouse Callback---------
active = True
save = False

def mouse_callback(event, x, y, flags, params):
    global active
    global save

    if event == cv.EVENT_RBUTTONDOWN:
        active = False

    if event == cv.EVENT_LBUTTONDOWN:
        if save:
            save = False
            print("Save deactivated")
        else:
            save = True
            print("Save activated")


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
    global t_shot_photo
    global timeout_over
    
    # ---------Parameter---------

    # muss kleiner sein als res_xy_full !!
    res_x_soll = 640
    res_y_soll = res_x_soll

    cor_x_soll = res_x_soll/res_x_small
    cor_y_soll = res_y_soll/res_y_small

    bike_area_min = 100000
    finish_A = (0,res_y_full-1)
    finish_B = (res_x_full-1, 0) 

    # ---------Initialization---------
    camera = PiCamera()
    camera.resolution = (res_x_full, res_y_full)
    camera.rotation = 180

    camera_Array = PiRGBArray(camera)

    print("[INFO] warming up...")
    time.sleep(1)

    filter_bgsub = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=3, detectShadows=True)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

    mask_empty = np.zeros((res_x_small, res_y_small, 1), np.uint8)
    mask_out = np.zeros((res_x_small, res_y_small, 1), np.uint8)
    mask_eval = np.zeros((res_x_small, res_y_small, 1), np.uint8)
    mask_full = np.ones((res_x_small, res_y_small, 1), np.uint8)

    cv.namedWindow("Mask")
    cv.setMouseCallback("Mask", mouse_callback)
    cv.imshow("Mask", mask_out)

    # ---------Timing---------
    t_delta_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    index = 0
    t_old = time.time()

    # ---------Process---------
    for f in camera.capture_continuous(camera_Array, format="bgr", use_video_port=True):
        frame_org = f.array

        frame = cv.resize(frame_org,(res_x_small,res_y_small))

        mask = filter_bgsub.apply(frame)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        ret, mask = cv.threshold(mask, 200, 255, cv.THRESH_BINARY)

        contours, hierachy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        areas = [cv.contourArea(c) for c in contours]

        if np.size(areas) != 0:
            max_index = np.argmax(areas)
            contour_max = contours[max_index]

            x,y,w,h = cv.boundingRect(contour_max)

            x_full = x*cor_x_full
            y_full = y*cor_x_full
            w_full = w*cor_x_full
            h_full = h*cor_x_full

            c_x = x_full + (w_full+1)/2.0
            c_y = y_full + (h_full+1)/2.0

            d_x = 0
            d_y = 0
            e_x = res_x_soll-1
            e_y = res_y_soll-1

            if c_x > res_x_full/2:
                if c_x+res_x_soll/2 <= res_x_full-1:
                    d_x = int(c_x-res_x_soll/2)
                    e_x = d_x+res_x_soll-1
                else:
                    d_x = res_x_full-res_x_soll
                    e_x = res_x_full-1
            else:
                if c_x-res_x_soll/2 >= 0:
                    d_x = int(c_x-res_x_soll/2)
                    e_x = d_x+res_x_soll-1
                else:
                    d_x = 0
                    e_x = res_x_soll-1

            if c_y > res_y_full/2:
                if c_y+res_y_soll/2 <= res_y_full-1:
                    d_y = int(c_y-res_y_soll/2)
                    e_y = d_y+res_y_soll-1
                else:
                    d_y = res_y_full-res_y_soll
                    e_y = res_y_full-1
            else:
                if c_y-res_y_soll/2 >= 0:
                    d_y = int(c_y-res_y_soll/2)
                    e_y = d_y+res_y_soll-1
                else:
                    d_y = 0
                    e_y = res_y_soll-1

            if not timeout_over:
                if t_shot_photo+1 < time.time():
                    timeout_over = True

            ret, finish_A, finish_B = check_Finish(x=x, y=y, w=w, h=h)
            
            if ret and timeout_over:
                session = 34
                if save:
                    cv.imwrite(os.path.join("Rider", "RID_" + str(session) + "_" + str(cnt) + ".jpg"), frame_org[d_y:e_y+1,d_x:e_x+1,:])
                    cnt = cnt+1
                    
                cv.imshow("Biker", frame_org[d_y:e_y+1,d_x:e_x+1,:])
                cv.waitKey(1)

                #timeout_over = False
                #t_shot_photo = time.time()
            
            cv.rectangle(mask, (x,y), (x+w, y+h), 255, 2)
            cv.rectangle(mask, (int(d_x/4),int(d_y/4)), (int(e_x/4), int(e_y/4)), 200, 2)
            
            mask_out = mask

        else:
            mask_out = mask_empty
            ret, finish_A, finish_B = check_Finish(False)
            
        cv.line(mask_out, finish_A, finish_B, 200, 2)
        
        cv.imshow("Mask", mask_out)
        cv.waitKey(1)
        
        camera_Array.truncate(0)

        if not active:
            break

        # ----------TIMING-------------
        if index<10:
            t_new = time.time()
            t_delta = t_new-t_old
            t_old = t_new

            t_delta_list[index] = t_delta
            index = index + 1
        elif False:
            sum = 0
            for x in t_delta_list:
                sum = sum + x
            mean = sum/10.0
            sum_dif = 0
            for x in t_delta_list:
                sum_dif = sum_dif + (x-mean)
            var = sum_dif/10.0

            print('---------------------------')
            print('mean = %.2f pps' % (1/mean))
            print('var = %.2f ms' % (var))
            print('---------------------------')

            index = 0
            t_old = time.time()

    cv.destroyAllWindows()

#-----------static variables-------------
d_old = 0
state_CF = "no_motion"

def check_Finish(motion_detected = True, x=0, y=0, w=0, h=0):
    global res_x_full
    global res_y_full
    global cor_x_full
    global cor_y_full
    global d_old
    global state_CF

    ret = False
    Finish_RS = True
    # Ziel ist auf der rechten Seite
    # Aus Kamera Perspektive

    #-----------------------
    #|    |       |     .
    #|    |       by  .
    #|    |       | .
    #|----|---bx--B
    #|    |      .
    #|    ay   .
    #|    |  .
    #|    |.
    #|-ax-A
    #|  .
    #|.
    #-----------------------
    
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

    if ab_x == 0:
        d = (a_x-c_x)/ab_y
    elif ab_y == 0:
        d = (c_y-a_y)/ab_x
    else:
        d = (1/(ab_x**2 + ab_y**2))*((c_y-a_y)*ab_x + (a_x-c_x)*ab_y)

    if not Finish_RS:
        d = -d
    
    if state_CF == "no_motion":
        if motion_detected and d<0:
            d_old = d
            state_CF = "motion_detected"
            
    elif state_CF == "motion_detected":
        if d > 0:
            if d > d_old:
                ret = True
                state_CF = "wait_for_next"
        d_old = d

    elif state_CF == "wait_for_next":
        if not motion_detected:
            state_CF = "no_motion"
        elif d<0:
            d_old=d
            state_CF = "motion_detected"

    return ret, (int(a_x/cor_x_full), int(a_y/cor_x_full)), (int(b_x/cor_x_full), int(b_y/cor_x_full))


if __name__ == "__main__":
    motion_detector()


