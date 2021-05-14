# Reference: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html
# Reference: https://vgg.fiit.stuba.sk/2015-02/2783/

import numpy as np
import cv2
import os
import argparse
import time
import datetime
import glob

parser = argparse.ArgumentParser(description='Monocular and stereo calibration. (recommend do monocular first, then do stereo)')

# Model infomation
parser.add_argument('-m',    '--mode',               type=str,   choices=['mono', 'stereo'],   help='mono or stereo calibration.', required=True)
parser.add_argument('-IDL',  '--left_ID',            type=int,   help='left camera device id', default=-1)
parser.add_argument('-IDR',  '--right_ID',           type=int,   help='right camera device id', default=-1)
parser.add_argument('-row',  '--chessboard_row',     type=int,   help='rows of intersection of chessboard (default:6)', default=6)
parser.add_argument('-col',  '--chessboard_column',  type=int,   help='columns of intersection of chessboard (default:9)', default=9)
parser.add_argument('-size', '--block_size',         type=int,   help='size of chessboard block(cm) (default:28)', default=28)

parser.add_argument('-mono',  '--mono_camera',         type=str,   choices=['left', 'right'], help='if mono mode is set. Please declare witch camera is going to use. (default:left)', default='left')
parser.add_argument('-intrL', '--left_intrinsic_dir',  type=str,   help='if stereo mode is set. left camera intrinsic parameters directory is needed.', default='')
parser.add_argument('-intrR', '--right_intrinsic_dir', type=str,   help='if stereo mode is set. right camera intrinsic parameters directory is needed.', default='')

args = parser.parse_args()

objp = np.zeros((args.chessboard_column * args.chessboard_row, 3), np.float32)
objp[:, :2] = np.mgrid[0:args.chessboard_column, 0:args.chessboard_row].T.reshape(-1, 2) * args.block_size

if args.mode == 'mono':
    # Checking argv
    camera_ID = args.left_ID if args.mono_camera == 'left' else args.right_ID
    if camera_ID == -1:
        parser.error("If mono mode is set, --IDL or --IDR must be set and match the --mono.")

    # Start monocular calibration
    print('\n Start {} calibration on {}_ID_{} camera...'.format(args.mode, args.mono_camera, camera_ID))

    # Checking device
    cap = cv2.VideoCapture(camera_ID)
    ret, frame = cap.read()
    if not ret:
        print(' Error: There is no device ID {} plese try another ID'.format(camera_ID))
        print('Exiting...')
        os._exit(0)
    height = frame.shape[0]
    width  = frame.shape[1]
    frame_shape = (width, height)
    #frame_shape = frame.shape[::-1]

    # Capture 40 images
    mono_frames = []
    while len(mono_frames) < 40:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Change BGR image to grayscale image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find & Draw chessboard
        ret_f, corners = cv2.findChessboardCorners(gray, (args.chessboard_column, args.chessboard_row), None)
        if ret_f:
            cv2.drawChessboardCorners(gray, (args.chessboard_column, args.chessboard_row), corners, ret_f)

        # Put text on image
        textToShow             = 'Clipped {} frames'.format(len(mono_frames))
        font                   = cv2.FONT_HERSHEY_COMPLEX
        bottomLeftCornerOfText = (50,50)
        fontScale              = 1
        fontColor              = (0,0,0)
        lineType               = 3
        gray = cv2.putText(gray, textToShow, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

        # Show image
        cv2.imshow('Cam {}'.format(camera_ID), gray)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == ord('Q'): # Press q/Q to break loop
            break
        elif key == ord(' '): # Press sapce to clip the frame
            if ret_f:
                mono_frames.append(frame)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
    # Write out frames
    now = datetime.datetime.now()
    current_time = '{:04d}_{:02d}_{:02d}_{:02d}{:02d}{:02d}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    images_dir = os.path.join(os.getcwd(), 'clips', 'monocular', args.mono_camera, current_time)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    for index, frame in enumerate(mono_frames):
        cv2.imwrite(os.path.join(images_dir, '{}.png'.format(index)), frame)
    
    # Check enough frames
    #current_time = '2018_11_14_210015'
    #images_dir = 'C:/Users/X550J/Desktop/clips/monocular/left/2018_11_14_210015'
    images = glob.glob(os.path.join(images_dir, '*.png'))
    if len(images) < 4:
        print('Error: Minimum 4 clips required...')
        os._exit(0)

    # Corner detection
    corners3D = []
    corners2D = []
    for image in images:
        gray = cv2.imread( image, cv2.IMREAD_GRAYSCALE)
        ret, corners = cv2.findChessboardCorners(gray, (args.chessboard_column, args.chessboard_row), None)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        if ret:
            rt = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            corners3D.append(objp)
            corners2D.append(corners)

    # Start calculate the Intrinsic parameters
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(corners3D, corners2D, frame_shape, None, None)
    calib_dir = os.path.join(os.getcwd(), 'calibs', 'monocular', current_time, args.mono_camera)
    if not os.path.exists(calib_dir):
        os.makedirs(calib_dir)
    np.save(os.path.join(calib_dir, 'mtx.npy'), mtx)
    np.save(os.path.join(calib_dir, 'dist.npy'), dist)
    np.save(os.path.join(calib_dir, 'rvecs.npy'), rvecs)
    np.save(os.path.join(calib_dir, 'tvecs.npy'), tvecs)

    # Undistortion
    img = cv2.imread(images[0])
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    np.save(os.path.join(calib_dir, 'roi.npy'), roi)
    np.save(os.path.join(calib_dir, 'newcameramtx.npy'), newcameramtx)

    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    np.save(os.path.join(calib_dir, 'mapx.npy'), mapx)
    np.save(os.path.join(calib_dir, 'mapy.npy'), mapy)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('calibresult.png', dst)

elif args.mode == 'stereo':
    # Checking argv
    if args.left_ID == -1 or args.right_ID == -1:
        parser.error("If stereo mode is set --IDL and --IDR both must be set.")

    # Start stereo calibration
    print('\n Start {} calibration on left_ID_{} & right_ID_{} camera...'.format(args.mode, args.left_ID, args.right_ID))

    # Checking device
    capL = cv2.VideoCapture(args.left_ID)
    capR = cv2.VideoCapture(args.right_ID)
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL:
        print(' Error: There is no device ID {} plese try another ID'.format(args.left_ID))
    if not retR:
        print(' Error: There is no device ID {} plese try another ID'.format(args.right_ID))
    if not retL or not retR:
        print('Exiting...')
        os._exit(0)
    height = frameL.shape[0]
    width  = frameL.shape[1]
    frame_shape = (width, height)

    # Capture 20 image pairs
    stereo_frames_L = []
    stereo_frames_R = []
    while len(stereo_frames_L) < 20:
        # Capture frame-by-frame
        retL, frameL = capL.read()
        retR, frameR = capR.read()

        # Change BGR image to grayscale image
        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        # Find & Draw chessboard
        retL, cornersL = cv2.findChessboardCorners(grayL, (args.chessboard_column, args.chessboard_row), None)
        retR, cornersR = cv2.findChessboardCorners(grayR, (args.chessboard_column, args.chessboard_row), None)
        if retL and retR:
            cv2.drawChessboardCorners(grayL, (args.chessboard_column, args.chessboard_row), cornersL, retL)
            cv2.drawChessboardCorners(grayR, (args.chessboard_column, args.chessboard_row), cornersR, retR)
        gray = np.concatenate((grayL, grayR), axis=1)

        # Put text on image
        textToShow             = 'Clipped {} frames'.format(len(stereo_frames_L))
        font                   = cv2.FONT_HERSHEY_COMPLEX
        bottomLeftCornerOfText = (50,50)
        fontScale              = 1
        fontColor              = (0,0,0)
        lineType               = 3
        gray = cv2.putText(gray, textToShow, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

        # Show image
        cv2.imshow('Cam {} & {}'.format(args.left_ID, args.right_ID), gray)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == ord('Q'): # Press q/Q to break loop
            break
        elif key == ord(' '): # Press sapce to clip the frame
            if retL and retR:
                stereo_frames_L.append(frameL)
                stereo_frames_R.append(frameR)

    # When everything done, release the capture
    capL.release()
    capR.release()
    cv2.destroyAllWindows()

    # Write out frames
    now = datetime.datetime.now()
    current_time = '{:04d}_{:02d}_{:02d}_{:02d}{:02d}{:02d}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    imagesL_dir = os.path.join(os.getcwd(), 'clips', 'stereo', current_time, 'left')
    imagesR_dir = os.path.join(os.getcwd(), 'clips', 'stereo', current_time, 'right')
    if not os.path.exists(imagesL_dir):
        os.makedirs(imagesL_dir)
    if not os.path.exists(imagesR_dir):
        os.makedirs(imagesR_dir)
    for index, frame in enumerate(stereo_frames_L):
        cv2.imwrite(os.path.join(imagesL_dir, '{}.png'.format(index)), frame)
    for index, frame in enumerate(stereo_frames_R):
        cv2.imwrite(os.path.join(imagesR_dir, '{}.png'.format(index)), frame)
    
    # Check enough frames
    imagesL = glob.glob(os.path.join(imagesL_dir, '*.png'))
    imagesR = glob.glob(os.path.join(imagesR_dir, '*.png'))
    if len(imagesL) < 4 or len(imagesR) < 4:
        print('Error: Minimum 4 clip pairs required...')
        os._exit(0)

    # Corner detection
    corners3D = []
    corners2DL = []
    corners2DR = []
    for imageL, imageR in zip(imagesL, imagesR):
        grayL = cv2.imread( imageL, cv2.IMREAD_GRAYSCALE)
        grayR = cv2.imread( imageR, cv2.IMREAD_GRAYSCALE)
        retL, cornersL = cv2.findChessboardCorners(grayL, (args.chessboard_column, args.chessboard_row), None)
        retR, cornersR = cv2.findChessboardCorners(grayR, (args.chessboard_column, args.chessboard_row), None)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        if retL and retR:
            rtL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            rtR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
            corners3D.append(objp)
            corners2DL.append(cornersL)
            corners2DR.append(cornersR)

    # Load Intrinsic parameters
    mono_mtxL = np.load(os.path.join(args.left_intrinsic_dir, 'mtx.npy'))
    mono_mtxR = np.load(os.path.join(args.right_intrinsic_dir, 'mtx.npy'))
    mono_distL = np.load(os.path.join(args.left_intrinsic_dir, 'dist.npy'))
    mono_distR = np.load(os.path.join(args.right_intrinsic_dir, 'dist.npy'))

    # Start calculate the Intrinsic parameters
    stereo_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    stereo_flags = (cv2.CALIB_FIX_ASPECT_RATIO | 
        cv2.CALIB_ZERO_TANGENT_DIST | 
        cv2.CALIB_SAME_FOCAL_LENGTH | 
        cv2.CALIB_RATIONAL_MODEL | 
        cv2.CALIB_FIX_K3 | 
        cv2.CALIB_FIX_K4 | 
        cv2.CALIB_FIX_K5)
    ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
        corners3D, 
        corners2DL, 
        corners2DR, 
        mono_mtxL, 
        mono_distL, 
        mono_mtxR, 
        mono_distR, 
        frame_shape, 
        criteria=stereo_criteria, 
        flags=stereo_flags)
    calib_dir = os.path.join(os.getcwd(), 'calibs', 'stereo', current_time)
    if not os.path.exists(calib_dir):
        os.makedirs(calib_dir)
    np.save(os.path.join(calib_dir, 'mtxL.npy'), mtxL)
    np.save(os.path.join(calib_dir, 'distL.npy'), distL)
    np.save(os.path.join(calib_dir, 'mtxR.npy'), mtxR)
    np.save(os.path.join(calib_dir, 'distR.npy'), distR)
    np.save(os.path.join(calib_dir, 'R.npy'), R)
    np.save(os.path.join(calib_dir, 'T.npy'), T)
    np.save(os.path.join(calib_dir, 'E.npy'), E)
    np.save(os.path.join(calib_dir, 'F.npy'), F)

    # Rectify images
    rectify_scale = 0 # 0=full crop, 1=no crop
    RL, RR, PL, PR, Q, roi1, roi2 = cv2.stereoRectify(
        mtxL, 
        distL, 
        mtxR, 
        distR, 
        frame_shape, 
        R, 
        T, 
        alpha = rectify_scale)
    mxL, myL = cv2.initUndistortRectifyMap(mtxL, distL, RL, PL, frame_shape, cv2.CV_32FC1)
    mxR, myR = cv2.initUndistortRectifyMap(mtxR, distR, RR, PR, frame_shape, cv2.CV_32FC1)
    np.save(os.path.join(calib_dir, 'RL.npy'), RL)
    np.save(os.path.join(calib_dir, 'RR.npy'), RR)
    np.save(os.path.join(calib_dir, 'PL.npy'), PL)
    np.save(os.path.join(calib_dir, 'Q.npy'), Q)
    np.save(os.path.join(calib_dir, 'roi1.npy'), roi1)
    np.save(os.path.join(calib_dir, 'roi2.npy'), roi2)
    np.save(os.path.join(calib_dir, 'mxL.npy'), mxL)
    np.save(os.path.join(calib_dir, 'myL.npy'), myL)
    np.save(os.path.join(calib_dir, 'mxR.npy'), mxR)
    np.save(os.path.join(calib_dir, 'myR.npy'), myR)

    # Undistort
    for imageL, imageR in zip(imagesL, imagesR):
        imgL = cv2.imread(imageL)
        imgR = cv2.imread(imageR)
        imgL_rectified = np.stack((cv2.remap(imgL[:,:,0], mxL, myL, cv2.INTER_LINEAR), cv2.remap(imgL[:,:,1], mxL, myL, cv2.INTER_LINEAR), cv2.remap(imgL[:,:,2], mxL, myL, cv2.INTER_LINEAR)), axis=2)
        imgR_rectified = np.stack((cv2.remap(imgR[:,:,0], mxR, myR, cv2.INTER_LINEAR), cv2.remap(imgR[:,:,1], mxR, myR, cv2.INTER_LINEAR), cv2.remap(imgR[:,:,2], mxR, myR, cv2.INTER_LINEAR)), axis=2)
        orig = np.concatenate((imgL, imgR), axis=1)
        show = np.concatenate((imgL_rectified, imgR_rectified), axis=1)
        cv2.imshow('original', orig)
        cv2.imshow('rectified', show)
        cv2.waitKey(0)
