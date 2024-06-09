# install instructions
Python 3.8
Pytorch 1.7.1
Torchvision 0.8.2
timm 0.4.12

# comments
# if you use quant_vision_transformer_original.py and Quant_original.py, that means no quantization on nonlinear kernels,
# and the results can be better.


conda activate QViT
tmux attach -t 0

python -m torch.distributed.launch --master_port=12345 --nproc_per_node=8 --use_env main.py --model fourbits_deit_tiny_patch16_224 --epochs 100 \
--warmup-epochs 0 --weight-decay 0. --batch-size 172 --data-path /data/dataset/image_net --lr 3e-4 --no-repeated-aug --pin-mem --output_dir ./dist_4bit_tiny_lamb_3e-4_300_512 \
--distillation-type hard --teacher-model deit_tiny_distilled_patch16_224
#--resume ./dist_4bit_tiny_lamb_3e-4_300_512/checkpoint.pth
#--opt fusedlamb



python -m torch.distributed.launch --master_port=12345 --nproc_per_node=8 --use_env main.py --model eightbits_deit_tiny_patch16_224 --epochs 100 \
--warmup-epochs 0 --weight-decay 0. --batch-size 156 --data-path /data/dataset/image_net --lr 3e-4 --no-repeated-aug --pin-mem --output_dir ./dist_8bit_tiny_lamb_3e-4_300_512 \
--distillation-type hard --teacher-model deit_tiny_distilled_patch16_224
#--resume ./dist_8bit_tiny_lamb_3e-4_300_512/checkpoint.pth



python -m torch.distributed.launch --master_port=12345 --nproc_per_node=8 --use_env main.py --model fourbits_deit_small_patch16_224 --epochs 100 \
--warmup-epochs 0 --weight-decay 0. --batch-size 84 --data-path /data/dataset/image_net  --lr 3e-4 --no-repeated-aug --pin-mem --output_dir ./dist_8bit_small_lamb_3e-4_300_512 \
--distillation-type hard --teacher-model deit_small_distilled_patch16_224





