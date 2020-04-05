import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion
import cv2

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


if __name__ == "__main__":
    action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE) # See rlbench/action_modes.py for other action modes
    env = Environment(action_mode, '', ObservationConfig(), False)
    task = env.get_task(EmptyContainer) # available tasks: EmptyContainer, PlayJenga, PutGroceriesInCupboard, SetTheTable
    agent = MoveAgent()
    obj_pose_sensor = NoisyObjectPoseSensor(env)
   
    descriptions, obs = task.reset()
    print(descriptions)
    while True:
        # Getting noisy object poses
        obj_poses = obj_pose_sensor.get_poses()
        small_container_pos = obj_poses['small_container0'][:3]

        # Getting various fields from obs
        current_joints = obs.joint_positions
        gripper_pose = obs.gripper_pose
        rgb = obs.wrist_rgb * 255
        depth = obs.wrist_depth
        mask = obs.wrist_mask

        # convert rgb to hsv
        hsv_image = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

        # establish bounds for thresholding for red object
        lower_bound = np.array([0, 0, 100])
        upper_bound = np.array([80, 80, 255])

        # create bounded mask
        mask_red = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # bitwise mask
        res = cv2.bitwise_and(rgb.astype(np.uint8), rgb.astype(np.uint8), mask_red)

        # convert image to gray
        gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        # # perform edge detection, then perform a dilation + erosion to
        # # close gaps in between object edges
        edges = cv2.Canny(gray, 50, 100)
        # edges = cv2.dilate(edges, None, iterations=1)
        # edges = cv2.erode(edges, None, iterations=1)
        cv2.imwrite('test.jpg', res)

        # Perform action and step simulation
        action = agent.act(obs, small_container_pos)
        obs, reward, terminate = task.step(action)

        # if terminate:
        #     break

    env.shutdown()
