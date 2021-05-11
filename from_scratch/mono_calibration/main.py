import numpy as np
import os
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image

def compute_view_based_homography(correspondence):
    image_points = correspondence[0]
    object_points = correspondence[1]

    N = len(image_points)
    M = np.zeros((2*N, 9), dtype=np.float64)

    # create row wise allotment for each 0-2i rows
    # that means 2 rows.. 
    for i in range(N):
        X, Y = object_points[i] #A
        u, v = image_points[i] #B

        row_1 = np.array([ -X, -Y, -1, 0, 0, 0, X*u, Y*u, u])
        row_2 = np.array([ 0, 0, 0, -X, -Y, -1, X*v, Y*v, v])
        M[2*i] = row_1
        M[(2*i) + 1] = row_2

    # M.h  = 0 . solve system of linear equations using SVD
    u, s, vh = np.linalg.svd(M)
    h_norm = vh[np.argmin(s)]
    h_norm = h_norm.reshape(3, 3)
    h = h_norm[:,:]/h_norm[2, 2]

    return h

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# (8,6) is for the given testing images.
# If you use the another data (e.g. pictures you take by your smartphone), 
# you need to set the corresponding numbers.
corner_x = 7
corner_y = 7
objp = np.zeros((corner_x*corner_y,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('ourdata/*.jpg')
img = cv2.imread(images[0])

if not os.path.isfile("objpoints.npy") or not os.path.isfile("imgpoints.npy"):
    # Step through the list and search for chessboard corners
    print('Start finding chessboard corners...')
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #Find the chessboard corners
        print('find the chessboard corners of',fname)
        ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)

    np.save('objpoints.npy', objpoints)
    np.save('imgpoints.npy', imgpoints)
objpoints = np.load('objpoints.npy')
imgpoints = np.load('imgpoints.npy')

#######################################################################################################
#                                Homework 1 Camera Calibration                                        #
#               You need to implement camera calibration(02-camera p.76-80) here.                     #
#   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
#                                          H I N T                                                    #
#                        1.Use the points in each images to find Hi                                   #
#                        2.Use Hi to find out the intrinsic matrix K                                  #
#                        3.Find out the extrensics matrix of each images.                             #
#######################################################################################################
print('Camera calibration...')
img_size = (img.shape[1], img.shape[0])
# You need to comment these functions and write your calibration function from scratch.
# Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
# In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
Vr = np.array(rvecs)
Tr = np.array(tvecs)
extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)
"""
Write your code here
"""

# For each image, find the Homography Matrix
correspondences = []
Homography_Matrixs = []
for (o, i) in zip(objpoints, imgpoints):
    correspondences.append([i.reshape(-1,2), o[:, :-1]])
print('correspondences',np.array(correspondences).shape)

for correspondence in correspondences:
    Homography_Matrixs.append(compute_view_based_homography(correspondence))


# Use the Homography_Matrixs to calculate intrinsic matrix K
def get_K_matrix(H_mtx_list):
    # b = (b11 , b12 , b13 , b22 , b23 , b33 )
    #     (b[0], b[1], b[2], b[3], b[4], b[5])
    H_mtx_num = len(H_mtx_list)
    V_mtx = np.zeros((2*H_mtx_num, 6), np.float64)

    def cal_V_vec(a, b, H_mtx):
        # a,b: the column of H_mtx
        V_vec = np.array([
            H_mtx[0,a]*H_mtx[0,b],
            H_mtx[1,a]*H_mtx[0,b] + H_mtx[0,a]*H_mtx[1,b],
            H_mtx[2,a]*H_mtx[0,b] + H_mtx[0,a]*H_mtx[2,b],
            H_mtx[1,a]*H_mtx[1,b],
            H_mtx[2,a]*H_mtx[1,b] + H_mtx[1,a]*H_mtx[2,b],
            H_mtx[2,a]*H_mtx[2,b]
        ])
        return V_vec
    
    # get V vector from each homography matrix
    for i, H_mtx in enumerate(H_mtx_list):
        # [h0.T][B][h1] = 0
        V_mtx[2*i+0] = cal_V_vec(0, 1, H_mtx)
        # [h0.T][B][h0] - [h1.T][B][h1] = 0
        V_mtx[2*i+1] = np.subtract(
            cal_V_vec(0, 0, H_mtx), cal_V_vec(1, 1, H_mtx)
        )
    
    # solve [V][b]=0 using SVD
    _, s, vh = np.linalg.svd(V_mtx)
    b = vh[np.argmin(s)]

    B_mtx = np.array(
        [
            [b[0], b[1], b[2]],
            [b[1], b[3], b[4]],
            [b[2], b[4], b[5]]
        ]
    )
#    print("the b matrix is: ")
#    print(B_mtx)
    def is_pos_def(x):
        return np.all(np.linalg.eigvals(x) > 0)
    if not is_pos_def(B_mtx):
        B_mtx = B_mtx*(-1)
    # guess, when B is real, B's cholesky factorization is [L][L.T]
    K_inv_T = np.linalg.cholesky(B_mtx)
    #print (K_inv_T.T)
    K_mtx = np.linalg.inv(K_inv_T.T)
    K_mtx = K_mtx/K_mtx[2,2]
#    print("the K matrix, which is intrinsic matrix, is: ")
#    print(K_mtx)
    return K_mtx

# ===ref. calibration example===
def get_intrinsic_parameters(H_r):
    M = len(H_r)
    V = np.zeros((2*M, 6), np.float64)

    def v_pq(p, q, H):
        v = np.array([
                H[0, p]*H[0, q],
                H[0, p]*H[1, q] + H[1, p]*H[0, q],
                H[1, p]*H[1, q],
                H[2, p]*H[0, q] + H[0, p]*H[2, q],
                H[2, p]*H[1, q] + H[1, p]*H[2, q],
                H[2, p]*H[2, q]
            ])
        return v

    for i in range(M):
        H = H_r[i]
        V[2*i] = v_pq(p=0, q=1, H=H)
        V[2*i + 1] = np.subtract(v_pq(p=0, q=0, H=H), v_pq(p=1, q=1, H=H))

    # solve V.b = 0
    u, s, vh = np.linalg.svd(V)
    # print(u, "\n", s, "\n", vh)
    b = vh[np.argmin(s)]
#    print("V.b = 0 Solution : ", b.shape)
#    print(b)
    # according to zhangs method
    vc = (b[1]*b[3] - b[0]*b[4])/(b[0]*b[2] - b[1]**2)
    l = b[5] - (b[3]**2 + vc*(b[1]*b[2] - b[0]*b[4]))/b[0]
    alpha = np.sqrt((l/b[0]))
    beta = np.sqrt(((l*b[0])/(b[0]*b[2] - b[1]**2)))
    gamma = -1*((b[1])*(alpha**2) *(beta/l))
    uc = (gamma*vc/beta) - (b[3]*(alpha**2)/l)

    '''    
    print([vc,
            l,
            alpha,
            beta,
            gamma,
        uc])
    '''
    A = np.array([
            [alpha, gamma, uc],
            [0, beta, vc],
            [0, 0, 1.0],
        ])
#    print("Intrinsic Camera Matrix is :")
#    print(A)
    return A

get_intrinsic_parameters(Homography_Matrixs)
Intrinsic_matrix = get_K_matrix(Homography_Matrixs)
#print("the intrinsic matrix from cv calibration: ")
#print(mtx)
# ===ref. calibration example===

# Find out the extrensics matrix of each images

'''
print ("extrinsic")
print (extrinsics)
'''
Intrinsic_inv = np.linalg.inv(Intrinsic_matrix)
H_mtx_num = len(Homography_Matrixs)
R_mtxs = np.zeros((3, 3), np.float64)
R_vecs = np.zeros((H_mtx_num, 3), np.float64)
T_mtxs = np.zeros((H_mtx_num, 3), np.float64)

for i, H_mtx in enumerate(Homography_Matrixs):
    multiple = Intrinsic_inv@H_mtx[:,0]
    lamda = 1/np.linalg.norm(multiple, ord=None, axis=None, keepdims=False)
    R_mtxs[:,0] = np.array(lamda*(Intrinsic_inv@H_mtx[:,0]))
    R_mtxs[:,1] = np.array(lamda*(Intrinsic_inv@H_mtx[:,1]))
    R_mtxs[:,2] = np.cross(R_mtxs[:,0], R_mtxs[:,1])
    T_mtxs[i] = np.array(lamda*(Intrinsic_inv@H_mtx[:,2]))
    cv2.Rodrigues(R_mtxs, R_vecs[i], jacobian=0)  
'''
print ("Cal R: ")
print (R_vecs)
print ("rvecs")
print (rvecs)
print ("Cal T")
print (T_mtxs)
print ("tvecs")
print (tvecs)
'''
Our_extrinsics = np.concatenate((R_vecs, T_mtxs), axis=1).reshape(-1,6)



"""
"""
# show the camera extrinsics
print('Show the camera extrinsics, figure by ourselves')
# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')
# camera setting
camera_matrix = mtx
cam_width = 0.064/0.1
cam_height = 0.032/0.1
scale_focal = 1600
# chess board setting
board_width = 8
board_height = 6
square_size = 1
# display
# True -> fix board, moving cameras
# False -> fix camera, moving boards
min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, Our_extrinsics, board_width,
                                                board_height, square_size, True)

X_min = min_values[0]
X_max = max_values[0]
Y_min = min_values[1]
Y_max = max_values[1]
Z_min = min_values[2]
Z_max = max_values[2]
max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

mid_x = (X_max+X_min) * 0.5
mid_y = (Y_max+Y_min) * 0.5
mid_z = (Z_max+Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, 0)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('-y')
ax.set_title('Extrinsic Parameters Visualization')
plt.show()

