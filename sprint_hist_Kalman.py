import argparse
import cv2
import sys
import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
import math


def integralSum(integral, top, left, bottom, right):
    return (integral[bottom, right] - integral[bottom, left] - integral[top, right] + integral[top, left])


def integralDensity(integral, top, left, bottom, right):
    area = integralArea(top, left, bottom, right)
    isum = integralSum(integral, top, left, bottom, right)
    return isum/area


def integralArea(top, left, bottom, right):
    area = abs(right - left) * abs(bottom - top)
    # print('area', area )
    return area


def processVideo(v):
    cap = cv2.VideoCapture(v)
    if not cap.isOpened():
        print("Error: unable to open video", v)

    bsmax = 0
    cnt = 1
    hist_init = [[]]
    hist_prev = [[]]

    fig, (ax_orig, ax_template, ax_corr) = plt.subplots(3, 1,
                                                        figsize=(10, 4))

    # ---- Set Kalman Parameters
    N = 30  # 30
    x = np.matrix('0. 0. 0. 0.').T
    P = np.matrix(np.eye(4)) * .000001000  # initial uncertainty
    R = 0.1**2

    kalman_x = np.zeros(N)
    kalman_y = np.zeros(N)

    # ---- Set up Helper Matrices
    F = np.matrix('''
                1. 0. 1. 0.;
                0. 1. 0. 1.;
                0. 0. 1. 0.;
                0. 0. 0. 1.
                ''')

    H = np.matrix('''
                1. 0. 0. 0.;
                0. 1. 0. 0.''')

    motion = np.matrix('0. 0. 0. 0.').T

    Q = np.matrix(np.eye(4))

    m = np.matrix('0. 0.').T

    # ----------- Optical Flow------------
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    # ------------------------------------
    kalman_previous = 0
    kalman_new = 0
    kalman_init = 0
    grad = 0
    # To save video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output#1.avi', fourcc, 20.0, (640, 480))

    while(cap.isOpened()):
        ret, frame = cap.read()
        OF_frame = frame
        print('read frame', cnt)
        print(frame.shape)

        height, width, depth = frame.shape

        frame = cv2.blur(frame, (5, 5))

        sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
       # sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
        #Laplacian = cv2.Laplacian(frame, cv2.CV_64F)
      #  cv2.imshow('frame', frame)
      #  cv2.imshow('Sobely', sobely)

        #frame = sobelx
        if (ret):
            # ----------- Optical Flow------------
            '''
            next = cv2.cvtColor(OF_frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow('Optical_Flow_Dense', bgr)
            #cv2.imshow('Image', OF_frame)

            prvs = next
            frame = bgr
            '''
            # ------------------------------------

            # frame[0:height, 600:630, :] = [ 0, 0, 0 ]
            # histr = cv2.calcHist([ frame[0:height//2,:,:] ], [1], None, [width//45],[0,200] )
            histr = []
            for bx in range(0, width - bwidth, bwidth):
                s = np.sum(frame[0:height, bx:bx+bwidth, 1])
                histr.append(s)
            # print(len(histr))
            # cv2.normalize( histr, histr, 0, height//2, cv2.NORM_MINMAX )
           # hist = [1000][1000]
            hist = np.array(histr)
            hist = np.int0(np.around(hist))/np.max(hist)
            cv2.normalize(hist, hist, 0, height//2, cv2.NORM_MINMAX)
            # print('hist', hist)
            px = 0
            py = height // 2
            dx = width // len(hist)
            for y in hist:
                x = px + dx
                y = int(height//2 - y)
                cv2.line(frame, (px, py), (x, y), (0, 255, 255), 10)
                cv2.line(frame, (px, py), (x, 10), (255, 255, 255), 1)
                px, py = x, y

            # Drow Compass line in button of video
            lineLenght = 100
            cv2.line(frame, (int(width / 2 - lineLenght), height - 50),
                     (int(width / 2 + lineLenght), height - 50), (255, 0, 0), 2)

            cv2.line(frame, (int(width / 2), height - 50 - 8),
                     (int(width / 2), height - 50 + 8), (0, 0, 255), 1)
            cv2.line(frame, (int(width / 2 + lineLenght), height - 50 - 10),
                     (int(width / 2 + lineLenght), height - 50 + 10), (0, 0, 255), 1)
            cv2.line(frame, (int(width / 2 - lineLenght), height - 50 - 10),
                     (int(width / 2 - lineLenght), height - 50 + 10), (0, 0, 255), 1)

            # The dynamic line to show deviation of movements
            if(math.sqrt(pow(kalman_previous-kalman_new, 2)) < 10):
                kalman_previous = kalman_new
                cv2.line(frame, (int(width / 2 + kalman_init-kalman_new), height - 50 - 8),
                         (int(width / 2 + kalman_init-kalman_new), height - 50 + 8), (0, 255, 0), 3)
            else:
                cv2.line(frame, (int(width / 2 + kalman_init-kalman_previous), height - 50 - 8),
                         (int(width / 2 + kalman_init-kalman_previous), height - 50 + 8), (0, 255, 0), 3)

            f = np.array([kalman_previous, kalman_new], dtype=float)
            grad = np.gradient(f)[1]

            #print(kalman_previous, kalman_new, 'grad: ', grad)

            cv2.imshow('Image', frame)

            # ---------------------------------
            # write the frame to hard
            out.write(frame)

            hist_new = np.vstack(hist)

            sliced = slice(round(len(hist_init)/3),
                           round(len(hist_init)/3) * 2, 1)

            #print('SLice: ', len(hist_new[sliced]))
            # print(sliced)

            if(cnt == 1 or cnt == 60):
                hist_init = hist_new
                hist_prev = hist_new

            if(cnt == 60):
                kalman_init = kalman_y[0, 0]
                kalman_previous = kalman_init

            corr_init = signal.correlate2d(
                hist_new, hist_init, boundary='symm', mode='same')
            y_i, x_i = np.unravel_index(np.argmax(corr_init), corr_init.shape)

            corr_prev = signal.correlate2d(
                hist_new, hist_prev, boundary='symm', mode='same')
            y_p, x_p = np.unravel_index(np.argmax(corr_prev), corr_prev.shape)

            corr_sliced_init = signal.correlate2d(
                hist_new, hist_init[sliced], boundary='fill', mode='valid')
            y_s_i, x_s_i = np.unravel_index(
                np.argmax(corr_sliced_init), corr_sliced_init.shape)

            corr_sliced_prev = signal.correlate2d(
                hist_new, hist_prev[sliced], boundary='symm', mode='same')
            y_s_p, x_s_p = np.unravel_index(
                np.argmax(corr_sliced_prev), corr_sliced_prev.shape)

            # blue is for initial frame cross correlated
            ax_orig.plot(cnt, y_i, 'bo', label='Frame to Frame')
            # red is for previous frame cross correlated
            ax_orig.plot(cnt, y_p, 'ro', label='Initial Capture to Frame')

            # blue is for divided initial frame cross correlated
            ax_template.plot(cnt, y_s_i, 'b*',
                             label='Frame to Frame (1/3 center)')
            # red is for divided previous frame cross correlated
            ax_template.plot(cnt, y_s_p, 'r+',
                             label='Initial Capture to Frame (1/3 center)')

            # ************ Kalman **************
            m[0] = y_s_i
            m[1] = y_s_i
            x, P = kalman(x, P, m, R, motion, Q, F, H)

            kalman_x = x[0]
            kalman_y = x[1]

            if(cnt == 1):
                kalman_previous = kalman_y[0, 0]

            ax_corr.plot(cnt, kalman_y[0, 0], 'g+',  label='Kalman')

            kalman_new = kalman_y[0, 0]

            #ax_corr.plot(cnt, kalman_y[0, 1], 'b+')
            #ax_corr.plot(cnt, kalman_x[0, 0], 'r+')
            #ax_corr.plot(cnt, kalman_x[0, 1], 'y+')

            plt.pause(0.00001)

            #print('x: ', x)
            #print('Ky: ', kalman_y[0, 0])
            #print('Kx: ', kalman_x)
            # print('Cor: ', corr)
            # print('NH: ', hist_new, 'OH:',  hist_old)

            hist_prev = hist_new

        cnt = cnt + 1

        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png', OF_frame)
            cv2.imwrite('opticalhsv.png', bgr)

    cap.release()
    cv2.destroyAllWindows()

# ---------------------------------


def detectVerticalLines():
    pass


def kalman(x, P, m, R, motion, Q, F, H):

    # ---- UPDATE x, P based on measurement m
    #            distance between measured and current position-belief

    y = m - H * x

    S = H * P * H.T + R  # residual convariance
    K = P * H.T * S.I    # Kalman gain

    x = x + K * y
    I = np.matrix(np.eye(F.shape[0]))  # identity matrix
    P = (I - K * H) * P

# ---- PREDICT x, P based on motion

    x = F * x + motion
    P = F * P * F.T + Q

# ---- Exit

    return x, P


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description='Track orientation via vision')
    parser.add_argument('videos', metavar='N', type=str, nargs='+',
                        help='videos')

    args = parser.parse_args(argv)
    for v in args.videos:
        processVideo(v)


if __name__ == "__main__":
    main()
