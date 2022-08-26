import cv2 as cv
import numpy as np

# Load the predefined dictionary
dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)

# Generate the marker
markerImage = np.zeros((200, 200), dtype=np.uint8)
markerImage = cv.aruco.drawMarker(dictionary, 33, 200, markerImage, 1);
cv.imwrite("marker33.png", markerImage);

#Load the dictionary that was used to generate the markers.
dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)

# Initialize the detector parameters using default values
parameters =  cv.aruco.DetectorParameters_create()

# Detect the markers in the image
markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(frame, dictionary, parameters=parameters)
# Calculate Homography
h, status = cv.findHomography(pts_src, pts_dst)

# Warp source image to destination based on homography
warped_image = cv.warpPerspective(im_src, h, (frame.shape[1],frame.shape[0]))

# Prepare a mask representing region to copy from the warped image into the original frame.
mask = np.zeros([frame.shape[0], frame.shape[1]], dtype=np.uint8);
cv.fillConvexPoly(mask, np.int32([pts_dst_m]), (255, 255, 255), cv.LINE_AA);

# Erode the mask to not copy the boundary effects from the warping
element = cv.getStructuringElement(cv.MORPH_RECT, (3,3));
mask = cv.erode(mask, element, iterations=3);

# Copy the mask into 3 channels.
warped_image = warped_image.astype(float)
mask3 = np.zeros_like(warped_image)
for i in range(0, 3):
    mask3[:,:,i] = mask/255

# Copy the masked warped image into the original frame in the mask region.
warped_image_masked = cv.multiply(warped_image, mask3)
frame_masked = cv.multiply(frame.astype(float), 1-mask3)
im_out = cv.add(warped_image_masked, frame_masked)