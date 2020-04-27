import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion, as_euler_angles, from_euler_angles, as_float_array
import time
from collections import namedtuple

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *

from helper import pick_up_box, pick_up_box_variables, generate_bounding_box, get_objects
from helper import *

def skew(x):
    return np.array([[0, -x[2], x[1]],
                    [x[2], 0, -x[0]],
                    [-x[1], x[0], 0]])

def sample_normal_pose(pos_scale, rot_scale):
    '''
    Samples a 6D pose from a zero-mean isotropic normal distribution
    '''
    pos = np.random.normal(scale=pos_scale)
        
    eps = skew(np.random.normal(scale=rot_scale))
    R = sp.linalg.expm(eps)
    quat_wxyz = from_rotation_matrix(R)

    return pos, quat_wxyz

class MoveAgent:

    def act(self, obs, target_pos):
        stepsize = 0.015
        movementVector = np.asarray([(target_pos[0]-obs.gripper_pose[0]),
                                     (target_pos[1]-obs.gripper_pose[1]),
                                     (target_pos[2]-obs.gripper_pose[2])])
        unitMovementVector = movementVector / np.linalg.norm(movementVector)
        robotStep = unitMovementVector * stepsize
        delta_quat = np.asarray([0, 0, 0, 1]) # xyzw
        gripper_pos = np.asarray([0])

        return np.concatenate((robotStep, delta_quat, gripper_pos))

class NoisyObjectPoseSensor:

    def __init__(self, env):
        self._env = env

        self._pos_scale = [0.005] * 3
        self._rot_scale = [0.01] * 3

    def get_poses(self):
        objs = self._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)
        obj_poses = {}

        for obj in objs:
            name = obj.get_name()
            # print(name)
            pose = obj.get_pose()

            pos, quat_wxyz = sample_normal_pose(self._pos_scale, self._rot_scale)
            gt_quat_wxyz = quaternion(pose[6], pose[3], pose[4], pose[5])
            perturbed_quat_wxyz = quat_wxyz * gt_quat_wxyz

            pose[:3] += pos
            pose[3:] = [perturbed_quat_wxyz.x, perturbed_quat_wxyz.y, perturbed_quat_wxyz.z, perturbed_quat_wxyz.w]

            obj_poses[name] = pose

        return obj_poses
    
    def get_objs(self):
        objs = self._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)
        return objs


if __name__ == "__main__":

    mode = 2
    action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN) # See rlbench/action_modes.py for other action modes
    env = Environment(action_mode, '', ObservationConfig(), False)
    task = env.get_task(EmptyContainer) # available tasks: EmptyContainer, PlayJenga, PutGroceriesInCupboard, SetTheTable
    agent = MoveAgent()
    obj_pose_sensor = NoisyObjectPoseSensor(env)
    update = 0
    count = 0
    

    gripper = task._robot.gripper
    objs = obj_pose_sensor.get_objs()
    for obj in objs:
        if (obj.get_name()=='small_container0'):
            small_container0_obj = obj

    descriptions, obs = task.reset()
    print(descriptions)
    global state
    state = 0
    global shape
    shape = '0'

    # initialize shape poses
    obj_poses = obj_pose_sensor.get_poses()
    shape_pos = obj_poses['Shape' + shape][:3]
    most_recent_shape_pos = shape_pos

    # Getting noisy object poses
    small_container_pos = obj_poses['small_container0'][:3]
    small_container_pos_original = obj_poses['small_container0'][:3]
    large_container_pos = obj_poses['large_container'][:3]
    small_container_quat2 = obj_poses['small_container0'][3:7]
    small_container_pos[2] -= 0.01
    small_container_quat = quaternion(obj_poses['small_container0'][3], obj_poses['small_container0'][4], obj_poses['small_container0'][5],obj_poses['small_container0'][6])
    small_container_euler = as_euler_angles(small_container_quat)

    z = small_container_euler[0]

    above_large_container,small_container_pos_original, notflipped_array,flipped_array = pick_up_box_variables(large_container_pos,obs,z,small_container_pos)

    while True:
        if mode==0:

            # Getting noisy object poses
            obj_poses = obj_pose_sensor.get_poses()
            small_container_pos = obj_poses['small_container0'][:3]


            # Perform action and step simulation
            if int(shape) < 5:  # shape goes from 0, 2, 4
                try:
                    shape_pos = obj_poses['Shape' + shape][:3]
                    most_recent_shape_pos = shape_pos
                except KeyError:
                    shape_pos = most_recent_shape_pos

            print(int(shape))

            action, state, shape = get_objects(state, shape, obs, shape_pos, small_container_pos)
            print("done")
            if int(shape) > 4:
                mode = 0.5

        if mode == 0.5:
            obj_poses = obj_pose_sensor.get_poses()
            small_container_pos = obj_poses['small_container0'][:3]
            gripper_pose = obs.gripper_pose
            small_container_pos[0] += 0.070*np.sin(z)
            small_container_pos[1] += 0.070*np.cos(z)
            update=-1
            mode = 1
        
        if mode == 1:
            update, action = pick_up_box(update,obs,gripper,
                                         small_container0_obj,z,
                                         small_container_pos,
                                         small_container_pos_original,
                                         gripper_pose,above_large_container,
                                         flipped_array,notflipped_array)

            print('state update (mode 1, pick-up box): ', update)

            shape = '0'
            if update == 11:
                mode = 2

        if mode == 2:
            mode, shapesToBeReset = checkShapePosition(obj_poses, obs)
            action = np.concatenate((obs.gripper_pose, np.array([1])))
            print(mode)
            state = 0
            i=0
            do_once_per_loop=0
            # mode = 3

        if mode ==3:
            obj_poses = obj_pose_sensor.get_poses()
            large_container_pos = obj_poses['large_container'][:3]
            
            
            
            
            if do_once_per_loop==0:
                print(shapesToBeReset)
                shape = shapesToBeReset[i]
                shape_pos = obj_poses['Shape' + shape][:3]

            print("shape",shape)
            action, state, do_once_per_loop = put_in_big_container(state, shape, obs, shape_pos, large_container_pos)
            if (do_once_per_loop==0):
                if ((i+1)>=len(shapesToBeReset)) :
                    mode=4
                else:   
                    i+=1
                    shape = shapesToBeReset[i]
                
        if mode==4:
            print("end")
            action = np.concatenate((obs.gripper_pose, np.array([1])))

        obs, reward, terminate = task.step(action)

    env.shutdown()
