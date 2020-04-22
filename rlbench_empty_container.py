import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion, as_euler_angles, from_euler_angles, as_float_array

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *

from helper import pick_up_box, pick_up_box_variables

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

    above_large_container,small_container_pos_original,notflipped_array,flipped_array = pick_up_box_variables(large_container_pos,obs,z,small_container_pos)

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
        
        state,action = pick_up_box(state,obs,gripper,small_container0_obj,z,small_container_pos,small_container_pos_original,gripper_pose,above_large_container,flipped_array,notflipped_array)
    
            
        obs, reward, terminate = task.step(action)

        # if terminate:
        #     break

    env.shutdown()
