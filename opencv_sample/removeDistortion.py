import numpy as np
import os, json
import cv2

name = '11.png'
out = '11_new.png'

f = open('intrinsic.json')
data = json.load(f)
mtx = np.array(data['Kmtx'])
dist = np.array(data['Dist'])
f.close()

# do only once
img = cv2.imread(name)
img_size = (img.shape[1], img.shape[0])
newcameramtx = mtx.copy()
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, np.eye(3), newcameramtx, img_size, 5)

# undistortion
img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
# img = cv2.undistort(img, mtx, dist, None, newcameramtx) # alternative
cv2.imwrite(out, img)
