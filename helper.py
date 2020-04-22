import numpy as np
from quaternion import from_rotation_matrix, quaternion, as_euler_angles, from_euler_angles, as_float_array



def pick_up_box_variables(large_container_pos,obs,z,small_container_pos):
    above_large_container = large_container_pos
    above_large_container[2] += 0.3

    notflipped = quaternion(obs.gripper_pose[3],obs.gripper_pose[4],obs.gripper_pose[5],obs.gripper_pose[6])
    notflipped_array = as_float_array(notflipped)
    flipped_euler = as_euler_angles(notflipped)
    

    amount_2_flip = -2
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
    if state == 0:
        action = np.concatenate((small_container_pos, gripper_pose[3:7], np.array([1])))
        state = 1
    # grasp box
    elif state == 1:
        action = np.concatenate((obs.gripper_pose, np.array([0])))
        # action[-1] = 0
        gripper.grasp(small_container0_obj)
        state = 2
    # moving sideways
    elif state == 2:
        pose = obs.gripper_pose
        pose[0] += np.sin(z)*0.02
        pose[1] += np.cos(z)*0.02
        action = np.concatenate((pose, np.array([0])))
        state = 3
    # moving up
    elif state == 3:
        pose = obs.gripper_pose
        pose[2] += 0.1
        action = np.concatenate((pose, np.array([0])))
        state = 4
    # above big container
    elif state == 4:
        action = np.concatenate((above_large_container, obs.gripper_pose[3:7], np.array([0])))
        state = 5
    #flip box
    elif state ==5:
        # print(flipped_array)
        action = np.concatenate((above_large_container, flipped_array, np.array([0])))
        state = 6
    elif state ==6:
        # print(flipped_array)
        action = np.concatenate((above_large_container, notflipped_array, np.array([0])))
        state = 7
    elif state ==7:
        # print(flipped_array)
        action = np.concatenate((small_container_pos_original+[0,0,0.1], notflipped_array, np.array([0])))
        state = 8
    elif state ==8:
        # print(flipped_array)
        action = np.concatenate((small_container_pos_original, notflipped_array, np.array([0])))
        state = 9
    elif state ==9:
        # print(flipped_array)
        action = np.concatenate((small_container_pos_original, notflipped_array, np.array([1])))
        state = 10
    elif state ==10:
        # print(flipped_array)
        action = np.concatenate((small_container_pos_original+[0,0,0.3], notflipped_array, np.array([1])))
        state = 11
    else:
        action = np.concatenate((obs.gripper_pose, np.array([1])))

    return state,action
        