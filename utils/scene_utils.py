import torch
import os
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']

import numpy as np

import copy
@torch.no_grad()
def render_training_image(scene, gaussians, viewpoints, render_func, pipe, background, stage, iteration, time_now):
    def render(gaussians, viewpoint, path, scaling):
        # scaling_copy = gaussians._scaling
        render_pkg = render_func(viewpoint, gaussians, pipe, background, stage=stage)
        label1 = f"stage:{stage},iter:{iteration}"
        times =  time_now/60
        if times < 1:
            end = "min"
        else:
            end = "mins"
        label2 = "time:%.2f" % times + end
        image = render_pkg["render"]
        depth = render_pkg["depth"]
        image_np = image.permute(1, 2, 0).cpu().numpy()
        depth_np = depth.permute(1, 2, 0).cpu().numpy()
        depth_np /= depth_np.max()
        depth_np = np.repeat(depth_np, 3, axis=2)
        image_np = np.concatenate((image_np, depth_np), axis=1)
        image_with_labels = Image.fromarray((np.clip(image_np,0,1) * 255).astype('uint8'))  
        draw1 = ImageDraw.Draw(image_with_labels)

        font = ImageFont.truetype('./utils/TIMES.TTF', size=40)

        text_color = (255, 0, 0)

        label1_position = (10, 10)
        label2_position = (image_with_labels.width - 100 - len(label2) * 10, 10)

        draw1.text(label1_position, label1, fill=text_color, font=font)
        draw1.text(label2_position, label2, fill=text_color, font=font)
        
        image_with_labels.save(path)
    render_base_path = os.path.join(scene.model_path, f"{stage}_render")
    point_cloud_path = os.path.join(render_base_path,"pointclouds")
    image_path = os.path.join(render_base_path,"images")
    if not os.path.exists(os.path.join(scene.model_path, f"{stage}_render")):
        os.makedirs(render_base_path)
    if not os.path.exists(point_cloud_path):
        os.makedirs(point_cloud_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    # image:3,800,800
    
    # point_save_path = os.path.join(point_cloud_path,f"{iteration}.jpg")
    for idx in range(len(viewpoints)):
        image_save_path = os.path.join(image_path,f"{iteration}_{idx}.jpg")
        # time = torch.tensor([idx]).unsqueeze(0)
        # render(gaussians,viewpoints[idx]['pose0_cam'],image_save_path,scaling=1,time=time)
        render(gaussians,viewpoints[idx]['t0_cam'],image_save_path,scaling = 1)
    # render(gaussians,point_save_path,scaling = 0.1)

    pc_mask = gaussians.get_opacity
    pc_mask = pc_mask > 0.1
    xyz = gaussians.get_xyz.detach()[pc_mask.squeeze()].cpu().permute(1,0).numpy()
    # visualize_and_save_point_cloud(xyz, viewpoint.R, viewpoint.T, point_save_path)
    # return image
    # image_with_labels_tensor = torch.tensor(image_with_labels, dtype=torch.float32).permute(2, 0, 1) / 255.0


def visualize_and_save_point_cloud(point_cloud, R, T, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    R = R.T
    T = -R.dot(T)
    transformed_point_cloud = np.dot(R, point_cloud) + T.reshape(-1, 1)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(transformed_point_cloud.T)
    # transformed_point_cloud[2,:] = -transformed_point_cloud[2,:]
    # 
    ax.scatter(transformed_point_cloud[0], transformed_point_cloud[1], transformed_point_cloud[2], c='g', marker='o')
    ax.axis("off")
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')

    plt.savefig(filename)
