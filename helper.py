'''
This file contains helper functions for the 16-662 Robot Autonomy
Final Project. This was completed by Bryson Jones, Quentin Cheng,
and Shaun Ryer.
Carnegie Mellon University
MRSD Class of 2021
'''

import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion, as_euler_angles, from_euler_angles, as_float_array
import cv2
import imutils
from imutils import perspective
from imutils import contours
from collections import namedtuple
from shapely.geometry import Point, Polygon
import time
import warnings


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






def pick_up_box_variables(large_container_pos,obs,z,small_container_pos):
    above_large_container = large_container_pos
    above_large_container[2] += 0.3

    notflipped = quaternion(obs.gripper_pose[3],obs.gripper_pose[4],obs.gripper_pose[5],obs.gripper_pose[6])
    notflipped_array = as_float_array(notflipped)
    flipped_euler = as_euler_angles(notflipped)
    

    amount_2_flip = -2.4
    if np.cos(z)<0:
        amount_2_flip = -amount_2_flip
    flipped_euler[2] += amount_2_flip
    flipped = from_euler_angles(flipped_euler)
    flipped_array = as_float_array(flipped)

    above_large_container[0] += 0.070*np.sin(z)
    above_large_container[1] += 0.070*np.cos(z)

    small_container_pos_original = small_container_pos + 0*small_container_pos
    small_container_pos_original[0] = small_container_pos[0]+0.070*np.sin(z)
    small_container_pos_original[1] = small_container_pos[1]+0.070*np.cos(z)

    return above_large_container,small_container_pos_original,notflipped_array,flipped_array

def pick_up_box(state,obs,gripper,small_container0_obj,z,small_container_pos,small_container_pos_original,gripper_pose,above_large_container,flipped_array,notflipped_array):
    # move to red box
    # state = 5
    if state == -1:
        print('state -1')
        action = np.concatenate((small_container_pos_original+[0,0,.3], gripper_pose[3:7], np.array([1])))
        state = 0
        return state,action

    elif state == 0:
        print('state 0')
        action = np.concatenate((small_container_pos_original+[0,0,0.01], gripper_pose[3:7], np.array([1])))
        state = 1
        return state,action
    # wait
    elif state == 1:
        print('state 1')
        action = np.concatenate((obs.gripper_pose, np.array([1])))
        # action[-1] = 0
        
        state = 2
        return state,action
    # moving sideways
    elif state == 2:
        pose = obs.gripper_pose
        pose[0] += np.sin(z)*0.02
        pose[1] += np.cos(z)*0.02
        action = np.concatenate((pose, np.array([0])))
        
        state = 3
        return state,action
    # moving up
    elif state == 3:
        pose = obs.gripper_pose
        pose[2] += 0.1
        gripper.grasp(small_container0_obj)
        action = np.concatenate((pose, np.array([0])))
        state = 4
        return state,action
    # above big container
    elif state == 4:
        action = np.concatenate((above_large_container, obs.gripper_pose[3:7], np.array([0])))
        state = 5
        return state,action
    #flip box
    elif state ==5:
        # print(flipped_array)
        action = np.concatenate((above_large_container, flipped_array, np.array([0])))
        state = 6
        return state,action
    elif state ==6:
        # print(flipped_array)
        action = np.concatenate((above_large_container, notflipped_array, np.array([0])))
        state = 7
        return state,action
    elif state ==7:
        # print(flipped_array)
        action = np.concatenate((small_container_pos_original+[0,0,0.1], notflipped_array, np.array([0])))
        state = 8
        return state,action
    elif state ==8:
        # print(flipped_array)
        action = np.concatenate((small_container_pos_original, notflipped_array, np.array([0])))
        state = 9
        return state,action
    elif state ==9:
        # print(flipped_array)
        action = np.concatenate((small_container_pos_original, notflipped_array, np.array([1])))
        state = 10
        return state,action
    elif state ==10:
        # print(flipped_array)
        action = np.concatenate((small_container_pos_original+[0,0,0.3], notflipped_array, np.array([1])))
        state = 11
        return state,action
    else:
        action = np.concatenate((obs.gripper_pose, np.array([1])))
        return state,action

    


def get_objects(state, shape, obs, object_pos, box_pos):
    
    #move above object
    if state == 0:
        
        action = np.concatenate((object_pos[0:3]+[0,0,0.15], obs.gripper_pose[3:7], np.array([1])))
        state = 1

        return action, state, shape
    #move to object
    elif state == 1:
        
        action = np.concatenate((object_pos[0:3]+[0,0,0.001], obs.gripper_pose[3:7], np.array([1])))
        
        state = 2

        return action, state, shape
    #move up
    elif state == 2:
        
        action = np.concatenate((object_pos[0:3]+[0,0,0.15], obs.gripper_pose[3:7], np.array([0])))
        state = 3

        return action, state, shape

    #move above box
    elif state == 3:
        
        action = np.concatenate((box_pos[0:3]+[0,0,0.15], obs.gripper_pose[3:7], np.array([0])))
        state = 4

        return action, state, shape

    #move close to box
    if state == 4:
        
        action = np.concatenate((box_pos[0:3]+[0,0,0.07], obs.gripper_pose[3:7], np.array([0])))
        state = 5

        return action, state, shape

    #drop and move away

    else:
        shape = str(int(shape) + 2)
        state=0
        action = np.concatenate((box_pos[0:3]+[0,0,0.15], obs.gripper_pose[3:7], np.array([1])))
        return action, state, shape

    

    # elif state == 1:
    #     if obs.gripper_pose[2] > .95:
    #         state = 2
    #     return [0, 0, stepsize, 0, 0, 0, 1, 0], state, shape

    # elif state == 2:
    #     if abs(box_pos[0] - obs.gripper_pose[0]) > .005 or abs(box_pos[1] - obs.gripper_pose[1]) > .005:
    #         action = np.concatenate((np.array([box_pos[0], box_pos[1], 0]), obs.gripper_pose[3:7], np.array([1])))

    #     else:
    #         action = np.concatenate((box_pos, obs.gripper_pose[3:7], np.array([1])))

    #     if np.isclose(box_pos, obs.gripper_pose, .05):
    #         state = 3
    #         return [0, 0, 0, 0, 0, 0, 1, 1], state, shape

    #     return action, state, shape

    # elif state == 3:
    #     if obs.gripper_pose[2] > 1:
    #         state = 0
    #         shape = str(int(shape) + 2)
    #     return [0, 0, stepsize, 0, 0, 0, 1, 1], state, shape


def checkShapePosition(obj_poses, obs):
    # have while loop to check for all of shape positions being accessible
    shapePosAccessible = False
    while not shapePosAccessible:
        try:
            shape0_pos = obj_poses['Shape0'][:3]
            print('Shape0 position accessed')
            shape2_pos = obj_poses['Shape2'][:3]
            print('Shape2 position accessed')
            shape4_pos = obj_poses['Shape4'][:3]
            print('All shape positions are available')
            shapePosAccessible = True

        except KeyError:
            warnings.warn('Can''t access all shapes yet.')
            time.sleep(1)

    # once we can access the object shape positions, we need to
    # we need to check if there are within the bounds of the large container
    largeContainerPosition = obj_poses['large_container'][:3]
    largeContainerOrientation = obj_poses['large_container'][3:]

    largeContainerOrientation = quaternion(largeContainerOrientation[0],
                                           largeContainerOrientation[1],
                                           largeContainerOrientation[2],
                                           largeContainerOrientation[3])
    largeContainerOrientation = as_euler_angles(largeContainerOrientation)

    transformValues = namedtuple('transformValues', ['x', 'y', 'yaw'])
    transformValuesPosition = transformValues(x=0, y=0, yaw=largeContainerOrientation[1])

    H_box = getTransform(transformValuesPosition)

    # get dimensions of box !!!!!!!!!!!!! THESE ARE RANDOM VALUES RIGHT NOW!!!!!!!!
    largeContainerLength = .28  # x dimension
    largeContainerWidth = .28  # y dimension

    boxCornerPoints = np.array([[largeContainerLength / 2, -largeContainerLength / 2, -largeContainerLength / 2, largeContainerLength / 2],
                                [largeContainerWidth / 2, largeContainerWidth / 2, -largeContainerWidth / 2, -largeContainerWidth / 2],
                                [1, 1, 1, 1]])

    rotatedBoxCornerPoints = H_box @ boxCornerPoints

    finalBoxCornerPoints = rotatedBoxCornerPoints + np.array([[largeContainerPosition[0]],
                                                              [largeContainerPosition[1]],
                                                              [1]])

    finalBoxCornerCoords = [(finalBoxCornerPoints[0, 0], finalBoxCornerPoints[1, 0]),
                            (finalBoxCornerPoints[0, 1], finalBoxCornerPoints[1, 1]),
                            (finalBoxCornerPoints[0, 2], finalBoxCornerPoints[1, 2]),
                            (finalBoxCornerPoints[0, 3], finalBoxCornerPoints[1, 3])]

    boxPolygon = Polygon(finalBoxCornerCoords)

    # check if points are in bounds
    shapes = [shape0_pos, shape2_pos, shape4_pos]
    shapeNum = ['0', '2', '4']
    shapesToBeReset = []
    for i in range(3):
        pointToCheck = Point(shapes[i][0], shapes[i][1])
        isWithinBox = pointToCheck.within(boxPolygon)
        print('Shape' + shapeNum[i], 'within bounds: ', isWithinBox)
        if not isWithinBox:
            shapesToBeReset.append(shapeNum[i])

    mode = 3

    return mode, shapesToBeReset


def getTransform(ArgStruct):
    '''
    getTransformation

       Returns the 2D homogeneous transformation for the given pose.

       Args: ArgStruct
           ArgStruct.x: x axis position of frame 2's origin in frame 1 [m]
           ArgStruct.y: y axis position of frame 2's origin in frame 1 [m]
           ArgStruct.yaw: angle between frames [rad]

       Returns: H
           H: 2D homogeneous transformation matrix [m]

       Raises:
           none

    '''
    H = np.asarray([[np.cos(ArgStruct.yaw), -np.sin(ArgStruct.yaw), ArgStruct.x],
                    [np.sin(ArgStruct.yaw), np.cos(ArgStruct.yaw), ArgStruct.y],
                    [0, 0, 1]])

    return H

