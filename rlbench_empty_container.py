import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion, as_euler_angles, from_euler_angles, as_float_array

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *

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


class RandomAgent:

    def act(self, obs):
        delta_pos = [(np.random.rand() * 2 - 1) * 0.005, 0, 0]
        delta_quat = [0, 0, 0, 1] # xyzw
        gripper_pos = [np.random.rand() > 0.5]
        return delta_pos + delta_quat + gripper_pos

class MoveAgent:

    def move2box(self, obs, target_pos):
        global state
        stepsize = 0.005
        movementVector = np.asarray([(target_pos[0]-obs.gripper_pose[0]),
                                     (target_pos[1]-obs.gripper_pose[1]),
                                     (target_pos[2]-obs.gripper_pose[2])])
        
        if np.linalg.norm(movementVector)<0.01:
            state = 1
            return [0, 0, 0, 0, 0, 0, 1, 0]

        unitMovementVector = movementVector / np.linalg.norm(movementVector)
        robotStep = unitMovementVector * stepsize
        # delta_quat = np.asarray([0, 0, 0, 1]) # xyzw
        delta_quat = target_pos[3:7]
        gripper_pos = np.asarray([1])

        return np.concatenate((robotStep, delta_quat, gripper_pos))

    def moveleft(self, obs, x, y):
        global state, count
        count += 1
        print(count)
        if count >= 10:
            state = 2
        stepsize = 0.005
        robotStep = np.asarray([stepsize*x, stepsize*y, 0])
        delta_quat = np.asarray([0, 0, 0, 1]) # xyzw
        # delta_quat = quat
        gripper_pos = np.asarray([0])

        return np.concatenate((robotStep, delta_quat, gripper_pos))
    
    def moveup(self, obs, length):
        global state
        stepsize = 0.005
        robotStep = np.asarray([0, 0, stepsize])
        delta_quat = np.asarray([0, 0, 0, 1]) # xyzw
        # delta_quat = quat
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
    action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN) # See rlbench/action_modes.py for other action modes
    env = Environment(action_mode, '', ObservationConfig(), False)
    task = env.get_task(EmptyContainer) # available tasks: EmptyContainer, PlayJenga, PutGroceriesInCupboard, SetTheTable
    agent = MoveAgent()
    obj_pose_sensor = NoisyObjectPoseSensor(env)
    state = 0
    count = 0

    gripper = task._robot.gripper
    objs = obj_pose_sensor.get_objs()
    for obj in objs:
        if (obj.get_name()=='small_container0'):
            small_container0_obj = obj

    descriptions, obs = task.reset()
    # print(descriptions)

    # Getting noisy object poses
    obj_poses = obj_pose_sensor.get_poses()
    small_container_pos = obj_poses['small_container0'][:3]
    small_container_pos_original = obj_poses['small_container0'][:3]
    large_container_pos = obj_poses['large_container'][:3]
    small_container_quat2 = obj_poses['small_container0'][3:7]
    small_container_pos[2] -= 0.01
    small_container_quat = quaternion(obj_poses['small_container0'][3], obj_poses['small_container0'][4], obj_poses['small_container0'][5],obj_poses['small_container0'][6])
    small_container_euler = as_euler_angles(small_container_quat)
    # print(small_container_euler)
    z = small_container_euler[0]
    # small_container_pos[2] += 0.1
    above_large_container = large_container_pos
    above_large_container[2] += 0.3

    notflipped = quaternion(obs.gripper_pose[3],obs.gripper_pose[4],obs.gripper_pose[5],obs.gripper_pose[6])
    notflipped_array = as_float_array(notflipped)
    flipped_euler = as_euler_angles(notflipped)
    print(flipped_euler)

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

    while True:
        # Get shapes
        shape0_pos = obj_poses['Shape'][:3]
        shape1_pos = obj_poses['Shape1'][:3]
        shape3_pos = obj_poses['Shape3'][:3]

        # The size of the small container is .25x by .14y by .07z (m)
        small_container_pos[0] += 0.070*np.sin(z)
        small_container_pos[1] += 0.070*np.cos(z)

        

        

        # Getting various fields from obs
        current_joints = obs.joint_positions
        gripper_pose = obs.gripper_pose
        rgb = obs.wrist_rgb
        depth = obs.wrist_depth
        mask = obs.wrist_mask

        # Perform action and step simulation
        # if state == 2:
        #     action = agent.moveup(obs, 1)
        # if state == 1:
        #     action = agent.moveleft(obs, np.sin(z), np.cos(z))
        # if state == 0:
        #     action = agent.move2box(obs, small_container_pos)
        
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
        
            
        obs, reward, terminate = task.step(action)

        # if terminate:
        #     break

    env.shutdown()
