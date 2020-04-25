'''
This file contains helper functions for the 16-662 Robot Autonomy
Final Project. This was completed by Bryson Jones, Quentin Cheng,
and Shaun Ryer.

Carnegie Mellon University
MRSD Class of 2021
'''

import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion
import cv2
import imutils
from imutils import perspective
from imutils import contours

def generate_bounding_box(rgb_img, lower_bound, upper_bound):
    '''
    :param rgb_img: an rgb image with an object that should have a bounding
                    box overlaid onto it
    :param lower_bound: lower thresholding bound (1 x 3)
    :param upper_bound: upper thresholding bound (1 x 3)

    :return: bgr_modified_img: image in bgr format, which has bounding box
                               overlaid
    '''

    # convert rgb to hsv
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    # create bounded mask
    mask_red = cv2.inRange(bgr, lower_bound, upper_bound)

    # bitwise mask
    redBoxOnlyFrame = cv2.bitwise_and(bgr.astype(np.uint8), bgr.astype(np.uint8), mask=mask_red)

    # convert image to gray
    gray = cv2.cvtColor(redBoxOnlyFrame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edges = cv2.Canny(gray, 50, 100)
    edges = cv2.dilate(edges, None, iterations=1)
    edges = cv2.erode(edges, None, iterations=1)

    # find contours in the edge map
    cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    try:
        (cnts, _) = contours.sort_contours(cnts)

        # loop over the contours individually
        bgr_modified_img = bgr.copy()
        for c in cnts:
            # if the contour is not sufficiently large, ignore it
            if cv2.contourArea(c) < 100:
                continue
            # compute the rotated bounding box of the contour
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding
            # box
            box = perspective.order_points(box)
            cv2.drawContours(bgr_modified_img, [box.astype("int")], -1, (0, 255, 0), 2)

        cv2.imwrite('test.jpg', bgr_modified_img)

        return bgr_modified_img

    except ValueError:
        # print('No image to return')
        1


def get_objects(state, shape, obs, object_pos, box_pos):
    stepsize = 0.02
    if state == 0:
        if abs(object_pos[0] - obs.gripper_pose[0]) > .008 or abs(object_pos[1] - obs.gripper_pose[1]) > .008:
            movementVector = np.asarray([(object_pos[0] - obs.gripper_pose[0]),
                                         (object_pos[1] - obs.gripper_pose[1]),
                                         .1 * (object_pos[2] - obs.gripper_pose[2])])
        else:
            movementVector = np.asarray([(object_pos[0] - obs.gripper_pose[0]),
                                         (object_pos[1] - obs.gripper_pose[1]),
                                         (object_pos[2] - obs.gripper_pose[2])])


        if np.linalg.norm(movementVector) < 0.006:
            state = 1
            return [0, 0, 0, 0, 0, 0, 1, 0], state, shape

        unitMovementVector = movementVector / np.linalg.norm(movementVector)
        robotStep = unitMovementVector * stepsize
        delta_quat = np.asarray([0, 0, 0, 1])  # xyzw
        gripper_pos = np.asarray([1])

        return np.concatenate((robotStep, delta_quat, gripper_pos)), state, shape

    elif state == 1:
        if obs.gripper_pose[2] > .95:
            state = 2
        return [0, 0, stepsize, 0, 0, 0, 1, 0], state, shape

    elif state == 2:
        # This was code to try and determine whether the object was actually picked up or not. It causes an error right now
        # if np.linalg.norm(np.asarray([(object_pos[0] - obs.gripper_pose[0]),
        #                               (object_pos[1] - obs.gripper_pose[1]),
        #                               (object_pos[2] - obs.gripper_pose[2])])) > .1:
        #     state = 0

        if abs(box_pos[0] - obs.gripper_pose[0]) > .005 or abs(box_pos[1] - obs.gripper_pose[1]) > .005:
            movementVector = np.asarray([(box_pos[0] - obs.gripper_pose[0]),
                                         (box_pos[1] - obs.gripper_pose[1]),
                                         0])
        else:
            movementVector = np.asarray([(box_pos[0] - obs.gripper_pose[0]),
                                         (box_pos[1] - obs.gripper_pose[1]),
                                         (box_pos[2] - obs.gripper_pose[2])])


        unitMovementVector = movementVector / np.linalg.norm(movementVector)
        robotStep = unitMovementVector * stepsize
        delta_quat = np.asarray([0, 0, 0, 1])  # xyzw
        gripper_pos = np.asarray([0])

        if np.linalg.norm(movementVector) < 0.05:
            state = 3
            return [0, 0, 0, 0, 0, 0, 1, 1], state, shape

        return np.concatenate((robotStep, delta_quat, gripper_pos)), state, shape

    elif state == 3:
        if obs.gripper_pose[2] > 1:
            state = 0
            shape = str(int(shape) + 2)
        return [0, 0, stepsize, 0, 0, 0, 1, 1], state, shape

