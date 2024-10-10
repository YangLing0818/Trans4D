## 3d objects training

### Stage 1
python 4dfy-main/launch.py --config 4dfy-main/configs/fourdfy_stage_1.yaml --train --gpu $gpu exp_root_dir=$exp_root_dir seed=123 system.prompt_processor.prompt="a panda dancing"
### Stage 2
ckpt=/path/to/fourdfy_stage_1/a_panda_dancing@timestamp/ckpts/last.ckpt
python 4dfy-main/launch.py --config 4dfy-main/configs/fourdfy_stage_2.yaml --train --gpu $gpu exp_root_dir=$exp_root_dir seed=123 system.prompt_processor.prompt="a panda dancing" system.prompt_processor.negative_prompt="" system.weights=$ckpt


## transform to ply

### transform into mesh (.obj)
python 4dfy-main/launch.py --config /path_to_4dfy/output/fourdfy_stage_2_low_vram/a_flower@timestamp/configs/parsed.yaml --export --gpu 0 resume=/path_to_4dfy/output/fourdfy_stage_2_low_vram/a_flower@timestamp/ckpts/last.ckpt system.exporter_type=mesh-exporter system.exporter.context_type=cuda system.exporter.fmt=obj
### transform into 3d cloud (.ply)
python mesh2ply_8w.py /path_to_4dfy/output/fourdfy_stage_2_low_vram/a_flower@timestamp/save/iterations-export/model.obj ./4D_data/input/butterfly_flower/a_flower.ply


## optimize 4D scene

### train
#### stage_1
python train_1_deform.py --configs ./arguments/drama_cut_tree_composer.py --expname tree_cut --image_weight_override 0.02 --nn_weight 1000 --with_reg --cfg_override 100.0 --loss_dx_weight_override 0.01 --model_path ./4D_data/output/drama_tree_cut
#### stage_2
python train_2_transition.py --configs ./arguments/drama_cut_tree_composer.py --expname tree_cut --image_weight_override 0.02 --nn_weight 1000 --with_reg --cfg_override 100.0 --loss_dx_weight_override 0.01 --model_path ./4D_data/output/drama_tree_cut
#### stage_3
python train_3_refine --opt ./arguments/top_hat_pigeon/pigeon.yaml --model_path ./4D_data/output/drama_top_hat_pigeon/ 

### render
python render_full_process.py --skip_train --configs ./arguments/drama_missile_plane_cloud_composer.py --skip_test --output_path ./4D_data/output/test_full_process_missile_explode --iteration 4500 --model_path ./4D_data/output/drama_top_hat_pigeon/
