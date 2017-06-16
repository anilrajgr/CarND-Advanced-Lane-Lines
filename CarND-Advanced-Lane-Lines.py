import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
images = glob.glob('camera_cal\calibration*.jpg')
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane


# In[4]:

# Prepare object points
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)


# In[5]:

# get_ipython().magic('matplotlib inline')

for fname in images:
    img = mpimg.imread(fname)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    
    # If corners are found, add object points, image points
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        
        # draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        # plt.imshow(img)
        # plt.show()
        


# In[6]:

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

def abs_sobel_thresh(img_ch, orient='x', thresh_min=0, thresh_max=255):
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img_ch, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img_ch, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

def mag_thresh(img_ch, sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(img_ch, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img_ch, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def dir_threshold(img_ch, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(img_ch, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img_ch, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def abs_select(img_ch, thresh=(0, 255)):
    binary_output = np.zeros_like(img_ch)
    binary_output[(img_ch > thresh[0]) & (img_ch <= thresh[1])] = 1
    return binary_output
def compute_perspective_xform():
    img = mpimg.imread(fname)
    img_size = (img.shape[1], img.shape[0])
    # Four source coordinates
    src = np.float32([
        [595, 450],
        [687, 450],
        [1050, 680],
        [275, 680]
    ])
    # Four desired coordinates
    dst = np.float32([
        [300, 0],
        [980, 0],
        [980, 720],
        [300, 720],
    ])
    # Compute the perspective transform, M
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    # Draw lines
    # cv2.line(img, (600, 447), (679, 447), [255,0,0], 4)
    # cv2.line(img, (679, 447), (1037, 681), [255,0,0], 4)
    # cv2.line(img, (1037, 681), (259, 681), [255,0,0], 4)
    # cv2.line(img, (259, 681), (600, 447), [255,0,0], 4)
    # 
    # cv2.line(warped, (429, 0), (858, 0), [255,0,0], 4)
    # cv2.line(warped, (858, 0), (858, 719), [255,0,0], 4)
    # cv2.line(warped, (858, 719), (429, 719), [255,0,0], 4)
    # cv2.line(warped, (429, 719), (429, 0), [255,0,0], 4)
    # 
    # f, (ax0, ax1) = plt.subplots(1, 2, figsize=(24,9))
    # ax0.imshow(img)
    # ax0.set_title('Original Image', fontsize=20)
    # ax1.imshow(warped)
    # ax1.set_title('Warped Image', fontsize=20)
    # plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0)
    
    return M, Minv

def warp(img, M):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

M, Minv = compute_perspective_xform()
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None


import copy
def mark_lanes_experiment(img, left_line_inst, right_line_inst):
    global left_curverad_prev
    global right_curverad_prev
    
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)
    
    hls_s_binary = abs_select(cv2.cvtColor(undist_img, cv2.COLOR_RGB2HLS)[:,:,2], thresh=(80, 255))
    ksize = 3 # Choose a larger odd number to smooth gradient measurements
    gradx = abs_sobel_thresh(cv2.cvtColor(undist_img, cv2.COLOR_RGB2GRAY), orient='x', thresh_min=20, thresh_max=150)
    grady = abs_sobel_thresh(cv2.cvtColor(undist_img, cv2.COLOR_RGB2GRAY), orient='y', thresh_min=20, thresh_max=255)
    mag_binary = mag_thresh(cv2.cvtColor(undist_img, cv2.COLOR_RGB2GRAY), sobel_kernel=7, thresh=(50,200))
    dir_binary = dir_threshold(cv2.cvtColor(undist_img, cv2.COLOR_RGB2GRAY), sobel_kernel=15, thresh=(0.6, 1.3))
    combined_gradient = np.zeros_like(dir_binary)
    combined_gradient[((gradx == 1) & (grady == 1)) | ((dir_binary == 1) & (mag_binary == 1))] = 1
    combined = np.zeros_like(gradx)
    combined[(hls_s_binary == 1) | (combined_gradient == 1)] = 1
    
    warped = warp(combined, M) # undist_img, M)
    
    # Take a histogram of the image
    histogram = np.sum(warped[warped.shape[0]/2:,:], axis=0)
    #### plt.plot(histogram)
    
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((warped, warped, warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped.shape[0] - (window+1)*window_height
        win_y_high = warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    # Plot the result
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Measure the curvature
    y_eval = warped.shape[0]
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')
    
    # Create an image to drw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([[left_fitx[y], y] for y in np.arange(720)])
    pts_right = np.array([[right_fitx[y], y] for y in np.arange(719,0,-1)])

    pts = np.concatenate((pts_left, pts_right))    
    cv2.polylines(color_warp, np.int_([pts_left]), False, (0, 0, 255), 30)
    cv2.polylines(color_warp, np.int_([pts_right]), False, (255, 0, 0), 30)
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    
    # Draw the line onto the warped blank image
    # warp the blank bak to original image space using inverse perspective matrix
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist_img.shape[1], undist_img.shape[0]))
    
    # Combine the result with the original image
    filled_img = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)
    return filled_img

left_lane_inst = Line()
right_lane_inst = Line()

def mark_lanes(img):
    global left_lane_inst
    global right_lane_inst
    filled_img = mark_lanes_experiment(img, left_lane_inst, right_lane_inst)
    return filled_img
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import inspect

proj_out = 'proj_out.mp4'
clip = VideoFileClip("project_video.mp4")
proj_clip = clip.fl_image(mark_lanes)
proj_clip.write_videofile(proj_out, audio=False)
