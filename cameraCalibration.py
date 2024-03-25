import cv2
import numpy as np
import glob


CHECKERBOARD = (8,11)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = []
imgpoints = []

objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)


images = glob.glob('./images7/*.jpg')
images = sorted(images)
for i,fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+
    	cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2,ret)
        cv2.imshow(fname+" succeed", img)
    else:
        print(f"第{i}张图，{fname}未发现足够角点")
        cv2.imshow(fname + " failed", img)
    cv2.waitKey(1)
cv2.destroyAllWindows()

h,w = img.shape[:2]

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print("相机内参:")
print(mtx,"\n")
print("畸变参数:")
print(dist,"\n")
print("旋转矩阵:")
print(rvecs,"\n")
print("平移矩阵:")
print(tvecs,"\n")
