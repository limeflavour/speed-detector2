import cv2
import numpy as np
import time
import copy
import os
import glob
import bgs
import multiprocessing as mpr
from datetime import datetime

from kalman_filter import KalmanFilter
from tracker import Tracker

if __name__ == '__main__':

    #FPS = 30

    #视频中下部到检测线的距离
    ROAD_DIST_MILES = 0.025


    #速度限制
    HIGHWAY_SPEED_LIMIT_MPH = 65
    HIGHWAY_SPEED_LIMIT_KMH = 50

    history = 100

    #背景建模方法
    algorithm = bgs.StaticFrameDifference()

    #字体
    font = cv2.FONT_HERSHEY_PLAIN


    centers = []

    #速度检测线的Y坐标
    Y_THRESH = 400

    #远处检测框的大小
    blob_min_width_far = 25
    blob_min_height_far = 25

    #近处检测框的大小
    blob_min_width_near = 30
    blob_min_height_near = 30

    frame_start_time = None

    #初始化一个Tracker
    tracker = Tracker(80, 3, 2, 1)


    cap = cv2.VideoCapture('/home/zxl/文档/speed-detector/TestVideo/t23.mp4')

    #视频帧宽
    frame_width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #视频FPS
    frame_FPS = round(cap.get(cv2.CAP_PROP_FPS))
    #视频每帧之间的等待时间
    pauseTime = round(1000 / frame_FPS)

    print("FPS: ",frame_FPS)
    print("pauseTime: ",pauseTime)

    while True:
        centers = []
        frame_start_time = datetime.utcnow()


        ret, frame = cap.read()

        if ret == False:
            break

        orig_frame = copy.copy(frame)


        #画出检测线
        cv2.line(frame, (0, Y_THRESH), (frame_width, Y_THRESH), (0, 139, 139), 2)


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray,(5,5,),0)

        #hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        #frame_H,frame_S,frame_V = cv2.split(hsv)

        #frame_B,frame_G,frame_R = cv2.split(frame)

        #fgmask = fgbg.apply(gray)

        #获取前景掩模
        fgmask = algorithm.apply(frame)



        # kernel = np.ones((4,4),np.uint8)
        # kernel_dilate = np.ones((5,5),np.uint8)
        # opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        # dilation = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel_dilate)

        #fgmask = cv2.adaptiveThreshold(fgmask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,0)

        #形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        erode = cv2.erode(fgmask,kernel)
        #erode = cv2.erode(erode, kernel)
        dilation = cv2.dilate(erode, kernel,None,None,3)
        #dilation = cv2.dilate(dilation, kernel)
        #dilation = cv2.dilate(dilation, kernel)
        #dilation = cv2.dilate(dilation, kernel)

        #寻找轮廓
        _, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



        #将检测到的汽车中心加入centers
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            #测试
            #print("矩形： ","x:", x, "y:", y, "w:" ,w , "h:", h)

            if y > Y_THRESH:
                if w >= blob_min_width_near and h >= blob_min_height_near:
                    center = np.array([[x + w / 2], [y + h / 2]])
                    centers.append(np.round(center))

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    #测试
                    #cv2.imshow("blob_min_width_near",frame)
                    #cv2.waitKey(0)
            else:
                if w >= blob_min_width_far and h >= blob_min_height_far:
                    center = np.array([[x + w / 2], [y + h / 2]])

                    centers.append(np.round(center))

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    #测试
                    #cv2.imshow("blob_min_width_far", frame)
                    #cv2.waitKey(0)

        if centers:
            tracker.update(centers)

            for vehicle in tracker.tracks:
                if len(vehicle.trace) > 1:
                    for j in range(len(vehicle.trace) - 1):
                        # 画出跟踪线
                        x1 = vehicle.trace[j][0][0]
                        y1 = vehicle.trace[j][1][0]
                        x2 = vehicle.trace[j + 1][0][0]
                        y2 = vehicle.trace[j + 1][1][0]

                        #画出上一矩形与当前帧矩形中心的连线
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)


                        #测试
                        #print("x1:",int(x1),"y1:",int(y1),"x2:",int(x2),"y2:",int(y2))
                        #print("-----------------------------------------")
                    try:

                        trace_i = len(vehicle.trace) - 1

                        trace_x = vehicle.trace[trace_i][0][0]
                        trace_y = vehicle.trace[trace_i][1][0]


                        if trace_y <= Y_THRESH + 5 and trace_y >= Y_THRESH - 5 and not vehicle.passed:
                            cv2.putText(frame, 'I PASSED!', (int(trace_x), int(trace_y)), font, 1, (0, 255, 255), 1,
                                        cv2.LINE_AA)
                            vehicle.passed = True

                            #load_lag为程序运行到这里的开销
                            load_lag = (datetime.utcnow() - frame_start_time).total_seconds()
                            print("-------------------------------------------------------")
                            print("frame_start_time:", frame_start_time,"datetime.utcnow():",datetime.utcnow(),"load_lag",load_lag)

                            time_dur = (datetime.utcnow() - vehicle.start_time).total_seconds() - load_lag
                            print("datetime.utcnow():",datetime.utcnow(),"vehicle.start_time:",vehicle.start_time,"load_lag:",load_lag)
                            print("time_dur",time_dur)
                            time_dur /= 60
                            time_dur /= 60

                            vehicle.mph = ROAD_DIST_MILES / time_dur
                            vehicle.kmh = vehicle.mph * 1.61


                            if vehicle.kmh > HIGHWAY_SPEED_LIMIT_KMH:
                                print('超速了!')
                                cv2.circle(orig_frame, (int(trace_x), int(trace_y)), 20, (0, 0, 255), 2)
                                #cv2.putText(orig_frame, 'MPH: %s' % int(vehicle.mph), (int(trace_x), int(trace_y)), font, 1,
                                #           (0, 0, 255), 1, cv2.LINE_AA)
                                cv2.putText(orig_frame, 'KMH: %s' % int(vehicle.kmh), (int(trace_x), int(trace_y)),
                                            font, 1, (0, 0, 255), 1, cv2.LINE_AA)
                                cv2.imwrite('speeding_%s.png' % vehicle.track_id, orig_frame)
                                print('超速照片已保存!')

                        if vehicle.passed:

                            cv2.putText(frame, 'KMH: %s' % int(vehicle.kmh), (int(trace_x), int(trace_y)), font, 1,
                                        (0, 255, 255), 1, cv2.LINE_AA)

                        else:

                            cv2.putText(frame, 'ID: ' + str(vehicle.track_id), (int(trace_x), int(trace_y)), font, 1,
                                        (255, 255, 255), 1, cv2.LINE_AA)
                    except:
                        pass


        cv2.imshow('original', frame)
        cv2.imshow('opening/erode', erode)
        cv2.imshow('opening/dilation', dilation)
        cv2.imshow('background subtraction', fgmask)

        #等待(1000/FPS)ms
        keyboard = cv2.waitKey(pauseTime)

        #按下'q'键推出
        if keyboard == 27:
            break
        #按下' '键暂停
        if keyboard == 32:
            cv2.waitKey(0)


        #time.sleep(1.0 / FPS)

    #释放资源
    cap.release()
    cv2.destroyAllWindows()

    #删除运行时产生的截图
    for file in glob.glob('speeding_*.png'):
        os.remove(file)
