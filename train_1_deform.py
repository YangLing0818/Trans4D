import numpy as np
import random
import os
import torch
import sys
import gc
from argparse import ArgumentParser, Namespace
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams, TrajParams
from torch.utils.data import DataLoader
from trajs_utils.trajs_utils import *
from gaussian_renderer.renderer import render
from scene.scene import Scene
from scene.gaussian_model_nogrid import GaussianModel_nogrid as GaussianModel

from guidance.sd_utils import StableDiffusion

from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from utils.timer import Timer

from render import render_set_fixcam, render_sets

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def scene_reconstruction(dataset, opt, hyper, pipe, trajs, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter,timer, args):
    first_iter = 0

    torch.cuda.empty_cache()
    gc.collect()
    print(f'Start training of stage {stage}: ')
    obj_prompts = []
    if opt.video_sds_type == 'zeroscope':
        from guidance.zeroscope_utils import ZeroScope
        zeroscope = ZeroScope('cuda', fp16=True)
        emb_zs = zeroscope.get_text_embeds([opt.prompt])
        for ww in opt.obj_prompt:
            obj_prompts.append(zeroscope.get_text_embeds([ww]))
    else:
        from videocrafter.scripts.evaluation.videocrafter2_utils import VideoCrafter2
        from omegaconf import OmegaConf
        vc_model_config = OmegaConf.load('videocrafter/configs/inference_t2v_512_v2.0.yaml').pop("model", OmegaConf.create())
        vc2 = VideoCrafter2(vc_model_config, ckpt_path='model.ckpt', weights_dtype=torch.float16, device='cuda')
        emb_zs = vc2.model.get_learned_conditioning([opt.prompt])
        neg_emb_zs = vc2.model.get_learned_conditioning(["text, watermark, copyright, blurry, nsfw"])
        cond = {"c_crossattn": [emb_zs], "fps": torch.tensor([6]*emb_zs.shape[0]).to(vc2.model.device).long()}
        un_cond = {"c_crossattn": [neg_emb_zs], "fps": torch.tensor([6]*emb_zs.shape[0]).to(vc2.model.device).long()}
        
        for ww in opt.obj_prompt:
            emb_zs = vc2.model.get_learned_conditioning([ww])
            obj_prompts.append({"c_crossattn": [emb_zs], "fps": torch.tensor([6]*emb_zs.shape[0]).to(vc2.model.device).long()})

    sd = StableDiffusion('cuda', fp16=True, sd_version='2.1')
    sd.get_text_embeds([opt.prompt], negative_prompts=['static statue, text, watermark, copyright, blurry, nsfw'])
    sd.get_objects_text_embeds(opt.obj_prompt, negative_prompts=['static statue, text, watermark, copyright, blurry, nsfw'])
    
    stage_ = ['fine']
    train_iter_ = [opt.iterations]
    white_bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda", requires_grad=False)
    black_bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda", requires_grad=False)

    for cur_stage, train_iter in zip(stage_, train_iter_):
        for gs in gaussians:
            gs.training_setup(opt)
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            for gs in gaussians:
                gs.restore(model_params, opt)
        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)
        viewpoint_stack = None
        ema_loss_for_log = 0.0

        final_iter = train_iter

        progress_bar = tqdm(range(first_iter, final_iter), desc=f"[{args.expname}] Training progress")
        offset_list = []
        for gs in gaussians:
            offset_list.append(lambda x:x)

        init_pos, move_list, move_time = trajs.init_pos, trajs.move_list, trajs.move_time
        init_angle, rotations, rotations_time = trajs.init_angle, trajs.rotations, trajs.rotations_time
        appear_list, appear_trans_time = trajs.appear_init, trajs.appear_trans_time
        funcs = []
        appears = []
        for i, _ in enumerate(gaussians):
            translation_list = query_trajectory(init_pos[i], move_list[i][:], move_time[i][:], 0, 1 / 48, 48 + 1)
            rotation_list = get_rotation(init_angle[i], rotations[i][:], rotations_time[i][:], 0, 1 / 48, 48 + 1)
            print('translation', translation_list)
            print('rotation', rotation_list)
            func = [prepare_offset(rotation_list[j], translation_list[j]) for j in range(len(rotation_list))]
            funcs.append(func)
            appears.append(get_appear_list(appear_list[i], appear_trans_time[i][:], 0, 1 / 48, 48 + 1))

        for iteration in range(first_iter, final_iter+1):
            stage = cur_stage
            loss_weight = 1
            if np.random.random() < 0.5:
                background = white_bg
            else:
                background = black_bg

            iter_start.record()
            for gs in gaussians:
                gs.update_learning_rate(iteration)
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras()
                viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=1,shuffle=True,num_workers=4,collate_fn=list)
                frame_num = viewpoint_stack.pose0_num

                loader = iter(viewpoint_stack_loader)

            try:
                data = next(loader)
            except StopIteration:
                print("reset dataloader")
                batch_size = 1
                loader = iter(viewpoint_stack_loader)
            if (iteration - 1) == debug_from:
                pipe.debug = True
            images = []
            radii_list = []
            visibility_filter_list = []
            viewspace_point_tensor_list = []
            dx = []
            out_pts = []
            viewpoint_cam = data[0]['rand_poses']
            fps = 1 / frame_num
            t0 = 0
            sds_idx_list = range(frame_num)

            if np.random.random() < 1.0:
                use_comp = True
            else:
                use_comp = False
            lower_bound = random.randint(0, 48 - frame_num)     # fix the bug, the low bound should before the video gen
            # render individual object
            gs_idx = random.choice(range(len(gaussians)))
            for i in sds_idx_list:
                new_i = i + lower_bound

                time = torch.tensor([t0 + new_i * (1/48)]).unsqueeze(0).float()
                for j, func in enumerate(funcs):
                    offset_list[j] = func[new_i]
                if use_comp:
                    appear_list = [appear[new_i] for appear in appears]
                    render_pkg = render(viewpoint_cam[0], gaussians, pipe, background, stage=stage, time=time, offset=offset_list, scales_list=opt.scales, appear_list=appear_list, pre_scale=opt.pre_scale)
                else:
                    render_pkg = render(viewpoint_cam[0], [gaussians[gs_idx]], pipe, background, stage=stage, time=time, offset=[offset_list[gs_idx]], scales_list=[opt.scales[gs_idx]], appear_list=[1], pre_scale=opt.pre_scale)

                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                fg_mask = render_pkg['alpha']
                rgba = torch.cat([image, fg_mask], dim=0)
                images.append(rgba.unsqueeze(0))
                if 'dx' in render_pkg:
                    dx.append(render_pkg['dx'])
                radii_list.append(radii.unsqueeze(0))
                visibility_filter_list.append(visibility_filter.unsqueeze(0))
                viewspace_point_tensor_list.append(viewspace_point_tensor)
            image_tensor = torch.cat(images,0)

            if use_comp:
                if opt.video_sds_type == 'zeroscope':
                    loss = zeroscope.train_step(image_tensor[:, :3], emb_zs)
                else:
                    loss = vc2.train_step(image_tensor[:, :3].unsqueeze(0).permute(0, 2, 1, 3, 4), cond, un_cond, cfg=opt.cfg, cfg_temporal=opt.cfg_temporal, as_latent=False)

                # img loss for comp renderings
                randints = list(range(16))
                np.random.shuffle(randints)
                img_loss = sd.train_step(image_tensor[randints[0]:randints[0]+1, :3], background=background) + sd.train_step(image_tensor[randints[1]:randints[1]+1, :3], background=background) \
                    + sd.train_step(image_tensor[randints[2]:randints[2]+1, :3], background=background) + sd.train_step(image_tensor[randints[3]:randints[3]+1, :3], background=background)
                print(f"origin loss is {loss}, image_loss with weight {opt.image_weight} is {img_loss * opt.image_weight}")
                loss = img_loss * opt.image_weight + loss * loss_weight

                if opt.with_reg:
                    dx_nn_loss = []
                    for dx_index, cur_dx in enumerate(dx):
                        tot_record = 0
                        ## rewrite here for multi object appear and disappear
                        for gs_idx in range(len(gaussians)):
                            if appears[gs_idx][dx_index + lower_bound] == 1:
                                ## the pts num in stage 1 is settled to 20000
                                dx_nn_loss.append(gaussians[gs_idx].get_nn_loss(cur_dx[20000*tot_record:20000*(tot_record+1)]))
                                tot_record += 1

                    # values inside the list are already mean-ed
                    loss_nn = torch.stack(dx_nn_loss).sum()
                    tb_writer.add_scalar(f'{stage}/dx_nn_comp', loss_nn.item(), iteration)
                    print(f'in comp loss_nn with weight {opt.nn_weight} is {loss_nn * opt.nn_weight}')
                    loss += loss_nn * opt.nn_weight
            else:
                # print(len(obj_prompts), gs_idx)
                if opt.video_sds_type == 'zeroscope':
                    loss = zeroscope.train_step(image_tensor[:, :3], obj_prompts[gs_idx])
                else:
                    loss = vc2.train_step(image_tensor[:, :3].unsqueeze(0).permute(0, 2, 1, 3, 4), obj_prompts[gs_idx], un_cond, cfg=opt.cfg, cfg_temporal=opt.cfg_temporal, as_latent=False)
                
                randints = list(range(16))
                np.random.shuffle(randints)
                img_loss = sd.train_step(image_tensor[randints[0]:randints[0]+1, :3], background=background, obj_id=gs_idx) + sd.train_step(image_tensor[randints[1]:randints[1]+1, :3], background=background, obj_id=gs_idx) \
                    + sd.train_step(image_tensor[randints[2]:randints[2]+1, :3], background=background, obj_id=gs_idx) + sd.train_step(image_tensor[randints[3]:randints[3]+1, :3], background=background, obj_id=gs_idx)
                print(f"origin loss is {loss}, image_loss with weight {opt.image_weight} is {img_loss * opt.image_weight}")
                loss = img_loss * opt.image_weight + loss * loss_weight

                if opt.with_reg:
                    dx_nn_loss = []
                    for cur_dx in dx:
                        dx_nn_loss.append(gaussians[gs_idx].get_nn_loss(cur_dx))
                    loss_nn = torch.stack(dx_nn_loss).sum()
                    tb_writer.add_scalar(f'{stage}/dx_nn_sep', loss_nn.item(), iteration)
                    print(f'in seperate loss_nn with weight {opt.nn_weight} is {loss_nn * opt.nn_weight}')
                    loss += loss_nn * opt.nn_weight
        
            if stage == 'fine':
                if (not use_comp) and gs_idx == 0:
                    loss_dx0 = torch.stack(dx).mean().abs()
                    tb_writer.add_scalar(f'{stage}/loss_dx0_mean', loss_dx0.item(), iteration)
                    loss_dx0 = torch.stack(dx).abs().sum()
                    loss += loss_dx0 * opt.loss_dx_weight
                    tb_writer.add_scalar(f'{stage}/loss_dx-first', loss_dx0.item(), iteration)
                else:
                    loss_dx0 = torch.stack([cur_dx.abs().sum() for cur_dx in dx]).abs().sum()
                    loss += loss_dx0 * opt.loss_dx_weight
                    tb_writer.add_scalar(f'{stage}/loss_dx-first', loss_dx0.item(), iteration)

            if stage == "fine" and hyper.time_smoothness_weight != 0:
                tv_loss = torch.sum([gs.compute_regulation(hyper.time_smoothness_weight, hyper.plane_tv_weight, hyper.l1_time_planes) for gs in gaussians])
                loss += tv_loss
                tb_writer.add_scalar(f'{stage}/loss_tv', tv_loss.item(), iteration)
            loss.backward()
            viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
            for idx in range(0, len(viewspace_point_tensor_list)):
                if viewspace_point_tensor_list[idx].grad is not None:
                    viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
            iter_end.record()
            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

                total_point = sum([gs._xyz.shape[0] for gs in gaussians])
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                            "point":f"{total_point}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()
                timer.pause()
                if iteration % 1000 == 0:
                    intermediate_save_path = os.path.join(args.model_path, "Intermediate_results")
                    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
                    background_val = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                    render_set_fixcam(intermediate_save_path, "video_stage1", iteration, scene.getVideoCameras(), gaussians, pipe, background_val, multiview_video=False, fname=f"pose_{0}.mp4", funcs=funcs, scales=opt.scales, appears=appears, pre_scale=opt.pre_scale, cam_idx=0)

                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration, stage)
                timer.start()

                if iteration < opt.iterations:
                    for gs in gaussians:
                        gs.optimizer.step()
                        gs.optimizer.zero_grad(set_to_none = True)


def training(dataset, hyper, opt, pipe, trajs, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname, args):
    tb_writer = prepare_output_and_logger(expname)
    gaussians = [GaussianModel(dataset.sh_degree, hyper) for __ in dataset.cloud_path] # init one GS model for each ply (object)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians,load_coarse=None)
    timer.start()
    scene_reconstruction(dataset, opt, hyper, pipe, trajs, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer, args)


from datetime import datetime

def prepare_output_and_logger(expname):
    if not args.model_path:
        unique_str = str(datetime.today().strftime('%Y-%m-%d')) + '/' + expname + '_' + datetime.today().strftime('%H:%M:%S')
        args.model_path = os.path.join("./output/", unique_str)
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    trajparam = TrajParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*50 for i in range(0,300)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2500, 3000, 3500, 4000, 4500, 5000, 7000, 8000, 9000, 14000, 20000, 25000, 30000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('-e', "--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "arguments/comp.py")
    parser.add_argument("--yyypath", type=str, default = "")
    parser.add_argument("--t0_frame0_rate", type=float, default = 1)
    parser.add_argument("--name_override", type=str, default="")
    parser.add_argument("--sds_ratio_override", type=float, default=-1)
    parser.add_argument("--sds_weight_override", type=float, default=-1)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--image_weight_override', type=float, default=-1)
    parser.add_argument('--nn_weight_override', type=float, default=-1)
    parser.add_argument('--cfg_override', type=float, default=-1)
    parser.add_argument('--cfg_temporal_override', type=float, default=-1) 
    parser.add_argument('--loss_dx_weight_override', type=float, default=-1)
    parser.add_argument('--with_reg_override', action='store_true', default=False)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations - 1)
    if args.configs:
        # import mmcv
        import mmengine
        from utils.params_utils import merge_hparams
        # config = mmcv.Config.fromfile(args.configs)
        config = mmengine.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    if args.name_override != '':
        args.name = args.name_override
    if args.sds_ratio_override != -1:
        args.fine_rand_rate = args.sds_ratio_override
    if args.sds_weight_override != -1:
        args.lambda_zero123 = args.sds_weight_override
    if args.image_weight_override != -1:
        args.image_weight = args.image_weight_override
    if args.nn_weight_override != -1:
        args.nn_weight = args.nn_weight_override
    if args.cfg_override != -1:
        args.cfg = args.cfg_override
    if args.cfg_temporal_override != -1:
        args.cfg_temporal = args.cfg_temporal_override
    if args.loss_dx_weight_override != -1:
        args.loss_dx_weight = args.loss_dx_weight_override
    if args.with_reg_override:
        args.with_reg = args.with_reg_override

    # print(args.name)
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    timer1 = Timer()
    timer1.start()
    print('Configs: ', args)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), trajparam.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname, args)
    print("\nTraining complete.")
    print('training time:',timer1.get_elapsed_time())

    render_sets(lp.extract(args), hp.extract(args), op.extract(args), trajparam.extract(args), args.iterations, pp.extract(args), lp.extract(args).model_path, skip_train=True, skip_test=True, skip_video=False, multiview_video=True)
    print("\Rendering complete.")
