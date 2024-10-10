import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def generate_coordinates(last_pos, delta, fps): # [0, 1]
    x = last_pos[0] + delta[0] * fps
    y = last_pos[1] + delta[1] * fps
    z = last_pos[2] + delta[2] * fps
    # return np.array([x, y, z]) # x y z
    return (x, y, z)


def prepare_offset(rotation, translation):
    def func(pts):
        return (torch.from_numpy(rotation).float().cuda().detach() @ pts.permute(1, 0)).permute(1, 0) + torch.from_numpy(translation).float().cuda().detach()
    return func


def find_rotation_matrix(x, y, z):
    """
    Parameters:
    - x: rotation angle in axis x.
    - y: rotation angle in axis y.
    - z: rotation angle in axis z.

    Returns:
    - The rotation matrix.
    """
    rotation_matrix = R.from_euler('xyz', [x, y, z], degrees=True).as_matrix()
        
    return rotation_matrix # [3, 3]


def get_rotation(init_angle, rotations, rotations_time, t0, fps, frame_num):
    last_angle, time = init_angle, t0
    rotate_speed = rotations[0]
    rotation_list = []
    for i in range(frame_num):
        rotation_list.append(find_rotation_matrix(last_angle[0], last_angle[1], last_angle[2]))
        last_angle = (last_angle[0]+rotate_speed[0], last_angle[1]+rotate_speed[1], last_angle[2]+rotate_speed[2])
        if len(rotations_time) != 0:
            if time >= rotations_time[0]:
                _ = rotations_time.pop(0)
                if len(rotations) > 1:
                    _ = rotations.pop(0)
                    rotate_speed = rotations[0]
        time += fps
    return rotation_list


def query_trajectory(init_pos, move_list, move_time, t0, fps, frame_num):
    # get_location = lambda t: np.array((R * np.sin(2 * np.pi * t * rot_speed), 0, R * np.cos(2 * np.pi * t * rot_speed)))
    last_pos, time = init_pos, t0
    speed = move_list[0]
    translation_list = []
    for i in range(frame_num):
        translation_list.append(np.array([last_pos[0], last_pos[1], last_pos[2]]))
        last_pos = generate_coordinates(last_pos, speed, fps)
        if len(move_time) != 0:
            if time >= move_time[0]:
                _ = move_time.pop(0)
                if len(move_list) > 1:
                    _ = move_list.pop(0)
                    speed = move_list[0]
        time += fps
    return translation_list


def get_appear_list(appear_init, appear_trans_time, t0, fps, frame_num):
    appear_last, time = appear_init, t0
    appear_list = []
    for i in range(frame_num):
        appear_list.append(appear_last)
        if len(appear_trans_time) != 0:
            if time >= appear_trans_time[0]:
                _ = appear_trans_time.pop(0)
                appear_last = 1 - appear_last
        time += fps
    return appear_list


def get_region_func(move_list, rotation_list, start_time, end_time, frame_num=49):
    full_video_fnum = len(rotation_list)
    start_fnum, end_fnum = int(full_video_fnum*start_time), int(full_video_fnum*end_time)
    start_pos, end_pos, start_rotation, end_rotation = move_list[start_fnum], move_list[end_fnum], rotation_list[start_fnum], rotation_list[end_fnum]
    region_move_list = [start_pos + (j*(end_pos-start_pos)/frame_num) for j in range(frame_num)]
    region_rotation_list = [start_rotation + (j*(end_rotation-start_rotation)/frame_num) for j in range(frame_num)]
    return region_move_list, region_rotation_list
