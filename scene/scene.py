#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model_nogrid import GaussianModel_nogrid as GaussianModel
# from scene.dataset import FourDGSdataset
from scene.text_dataset import Text4Ddataset
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.utils.data import Dataset
import numpy as np

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None,shuffle=True, resolution_scales=[1.0], load_coarse=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.names = [os.path.basename(x) for x in args.cloud_path]
        
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}
        self.cameras_extent = 1 # scene_info.nerf_normalization["radius"]

        print("Loading Training Cameras")
        # if args.imagedream:
        #     ds = ImageDreamdataset
        # else:
        #     ds = FourDGSdataset
        ds = Text4Ddataset
        self.train_camera = ds(split='train', radius=args.radius, W=args.render_W, H=args.render_H, frame_num=args.frame_num,name=args.name,rife=args.rife,static=args.static)
        print("Loading Test Cameras")
        self.maxtime = self.train_camera.pose0_num
        self.test_camera = ds(split='test', radius=args.radius, W=args.render_W, H=args.render_H, frame_num=args.frame_num,name=args.name,rife=args.rife,static=args.static)
        print("Loading Video Cameras")
        self.video_cameras = ds(split='video', radius=args.radius, W=args.render_W, H=args.render_H, frame_num=args.frame_num,name=args.name,rife=args.rife,static=args.static)

        if self.loaded_iter:
            for idx, _ in enumerate(self.gaussians):
                if os.path.exists(os.path.join(self.model_path, "point_cloud_refine")):
                    load_iter_refine = searchForMaxIteration(os.path.join(self.model_path, "point_cloud_refine"))
                    if os.path.exists(os.path.join(self.model_path, "point_cloud_refine", "iteration_" + str(load_iter_refine), self.names[idx])):
                        _.load_model(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), f"{self.names[idx].replace('.ply', '')}"))
                        _.load_luciddreamer_ply(os.path.join(self.model_path, "point_cloud_refine", "iteration_" + str(load_iter_refine), self.names[idx]))
                    else:
                        _.load_ply(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), f"{self.names[idx]}"))
                        _.load_model(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), f"{self.names[idx].replace('.ply', '')}"))

                else:
                    _.load_ply(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), f"{self.names[idx]}"))
                    _.load_model(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), f"{self.names[idx].replace('.ply', '')}"))
                # sub directory for each object
        else:
            for idx, _ in enumerate(self.gaussians):
                _.load_3studio_ply(args.cloud_path[idx], spatial_lr_scale=1, time_line=self.maxtime, pts_num=int(2e4), position_scale=1/2.5, load_color=True) ## 4dfy
            # self.gaussians.load_3studio_ply(cloud_path, spatial_lr_scale=1, time_line=self.maxtime, step=1, position_scale=1, load_color=True) ## imagedream

    def save(self, iteration, stage):
        # if stage == "coarse":
        #     point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))

        # else:
        # if not os.path.exists()
        # TODO: save offset and rotation for objects
        for idx, _ in enumerate(self.gaussians):
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
            _.save_ply(os.path.join(point_cloud_path, f"{self.names[idx]}")) # comes with .ply
            _.save_deformation(os.path.join(point_cloud_path,  f"{self.names[idx].replace('.ply', '')}"))
            # _.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
            # _.save_deformation(point_cloud_path)
    
    def save_refine(self, iteration, stage):
        for idx, _ in enumerate(self.gaussians):
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
            _.save_ply(os.path.join(point_cloud_path, f"refine_{self.names[idx]}")) # comes with .ply

    def getTrainCameras(self, scale=1.0):
        return self.train_camera

    def getTestCameras(self, scale=1.0):
        return self.test_camera
    def getVideoCameras(self, scale=1.0):
        return self.video_cameras

    def get_total_points(self):
        return sum([_.get_xyz.shape[0] for _ in (self.gaussians)])