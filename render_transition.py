import imageio
import numpy as np
import torch
import os
import cv2
import sys
from tqdm import tqdm
from os import makedirs
from time import time

from trajs_utils.trajs_utils import *
from scene.scene import Scene
from scene.gaussian_model_nogrid import GaussianModel_nogrid as GaussianModel
from scene.transition import Transition
from gaussian_renderer.renderer import render

from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams,OptimizationParams, get_combined_args, ModelHiddenParams, TrajParams


to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
def render_set_fixcam(model_path, name, iteration, views, gaussians, transitions, trajs, pipeline, background,multiview_video, fname='video_rgb.mp4', funcs=None, scales=None, appears=[], pre_scale=False, cam_idx=25):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []
    print(len(views))

    transition_idx = 0
    g_id_list = trajs.trans_list[transition_idx]
    trans_start, trans_end = trajs.trans_period[transition_idx][0], trajs.trans_period[transition_idx][1]

    ####
    fnum = 48
    for idx in tqdm(range (fnum)):
        view = views[cam_idx]
        if idx == 0:time1 = time()
        ww = torch.tensor([trans_start + (idx * (trans_end - trans_start) / fnum)]).unsqueeze(0)

        ###### here to determine whether appear ######
        appear_list = [appear[int((trans_start + (idx * (trans_end - trans_start) / fnum))*48)] for appear in appears]
        ##############################################

        trans_list = [0] * len(gaussians)
        if (idx > 0.1 * fnum) and (idx < 0.9 * fnum):
            for t_idx in g_id_list: trans_list[t_idx] = 1

        # Optional: set transition iter rates will make the results better
        iter_rates = [1.0]*len(gaussians)
        iter_rate = idx/fnum
        for rate_i, _ in enumerate(iter_rates):
            if len(trajs.appear_trans_time[rate_i]) == 0: continue
            if len(trajs.appear_trans_time[rate_i])%2 == 1:
                iter_rates[rate_i] = iter_rate if trajs.appear_init[rate_i]==0 else 1-iter_rate
            elif len(trajs.appear_trans_time[rate_i])%2 == 0:
                iter_rate = iter_rate * 2 if iter_rate < 0.5 else (1-iter_rate) * 2
                iter_rates[rate_i] = iter_rate if trajs.appear_init[rate_i]==0 else 1-iter_rate

        rendering = render(view['cur_cam'], gaussians, pipeline, background,\
                           transitions=transitions, time=ww, stage='fine', offset=[func[idx] for func in funcs],\
                           scales_list=scales, appear_list=appear_list, transition_list=trans_list, pre_scale=pre_scale, iter_rate=iter_rates)["render"]

        render_images.append(to8b(rendering).transpose(1,2,0))
        render_list.append(rendering)
    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))
    print('Len', len(render_images))
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), fname), render_images, fps=20, quality=8)


from importlib import import_module
def render_sets(dataset : ModelParams, hyperparam, opt, trajs, iteration : int, pipeline : PipelineParams, output_path : str, skip_train : bool, skip_test : bool, skip_video: bool,multiview_video: bool):

    init_pos, move_list, move_time = trajs.init_pos, trajs.move_list, trajs.move_time
    init_angle, rotations, rotations_time = trajs.init_angle, trajs.rotations, trajs.rotations_time
    appear_list, appear_trans_time = trajs.appear_init, trajs.appear_trans_time
    funcs = []
    appears = []

    transition_idx = 0
    transition_start, transition_end = trajs.trans_period[transition_idx][0], trajs.trans_period[transition_idx][1]     # new test
    for i, _ in enumerate(dataset.cloud_path):
        translation_list = query_trajectory(init_pos[i], move_list[i][:], move_time[i][:], 0, 1 / 48, 48 + 1)
        rotation_list = get_rotation(init_angle[i], rotations[i][:], rotations_time[i][:], 0, 1 / 48, 48 + 1)
        region_trans_list, region_rotat_list = get_region_func(translation_list, rotation_list, transition_start, transition_end, 48+1)
        func = [prepare_offset(region_rotat_list[j], region_trans_list[j]) for j in range(len(region_rotat_list))]
        funcs.append(func)
        appears.append(get_appear_list(appear_list[i], appear_trans_time[i][:], 0, 1 / 48, 48 + 1))

    with torch.no_grad():
        gaussians = [GaussianModel(dataset.sh_degree, hyperparam) for __ in dataset.cloud_path]
        transitions = [Transition().to("cuda") for _ in dataset.cloud_path]
        # gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        if iteration == -1:
            scene = Scene(dataset, gaussians, load_coarse=None)
        else:
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
            if os.path.exists(os.path.join(dataset.model_path, "transition_network")):
                for c_idx, trans_net in enumerate(transitions):
                    save_name = f"{os.path.basename(dataset.cloud_path[c_idx]).replace('.ply', '')}.pth"
                    iteration_trans = searchForMaxIteration(os.path.join(dataset.model_path, "transition_network"))
                    load_transition_weight = torch.load(os.path.join(dataset.model_path, "transition_network", "iteration_" + str(iteration_trans), save_name), map_location="cpu")
                    trans_net.load_state_dict(load_transition_weight)
        offset_list = []
        for gs in scene.gaussians:
            offset_list.append(lambda x:x)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not skip_video:
            #origin
            for cam_idx in range(0, 100, 5):
                render_set_fixcam(output_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,transitions,trajs,pipeline,background,multiview_video=False, fname=f"pose_{cam_idx}.mp4", funcs=funcs, scales=opt.scales, appears=appears, pre_scale=opt.pre_scale, cam_idx=cam_idx)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    trajparam = TrajParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--output_path", default="output/test", type=str)
    parser.add_argument('--multiview_video',default=False,action="store_true")
    parser.add_argument("--configs", type=str)
    # args = get_combined_args(parser)
    args = parser.parse_args(sys.argv[1:])

    if args.configs:
        # import mmcv
        import mmengine
        from utils.params_utils import merge_hparams
        # config = mmcv.Config.fromfile(args.configs)
        config = mmengine.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), op.extract(args), trajparam.extract(args), args.iteration, pipeline.extract(args), args.output_path, args.skip_train, args.skip_test, args.skip_video,args.multiview_video)
