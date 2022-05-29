#! /bin/bash
# for first
#rm -rf Stripe1LPkernel
#python3 -m grid Stripe1LPkernel --n 2 "python3 main.py --device cpu --max_wall 10000 --d 2 --seed_trainset 0 --seed_testset 0 \
#--arch fc --act relu --loss hinge  --L 1 --dataset stripe --ptk 1000 --pte 50000 --tau_alpha_crit 1e3 --tau_over_h 1e-3 \
#--save_weight 1 --bias 1 --save_neurons 100 --save_function 1 --final_kernel 1 --final_features 1 --delta_kernel 1 --init_kernel 1
#" --seed_init 0 --alpha 1e-4 1e-2 1e0 1e2 1e4 --ptr 5000 7000 9000 --h:int 10000

## for second
#rm -rf Stripe1LPkernel2
#python3 -m grid Stripe1LPkernel2 --n 2 "python3 main.py --device cpu --max_wall 10000 --d 2 --seed_trainset 0 --seed_testset 0 \
#--arch fc --act relu --loss hinge  --L 1 --dataset stripe --ptk 1000 --pte 50000 --tau_alpha_crit 1e3 --tau_over_h 1e-3 \
#--save_weight 1 --bias 1 --save_neurons 100 --save_function 1 --final_kernel 1 --final_features 1 --delta_kernel 1 --init_kernel 1
#" --seed_init 0 --alpha 1e-4 1e-2 1e0 1e2 1e4 --ptr 2000 3000 4000 6000 8000 --h:int 10000

# for third
rm -rf Stripe1LPkernel3
python3 -m grid Stripe1LPkernel3 --n 2 "python3 main.py --device cpu --max_wall 1 --d 5 --seed_trainset 0 --seed_testset 0 \
--arch fc --act relu --loss hinge  --L 1 --dataset stripe --ptk 1000 --pte 10000 --tau_alpha_crit 1e3 --tau_over_h 1e-3 \
--save_weight 1 --bias 1 --save_neurons 100 --save_function 1 2 3 --final_kernel 1 --final_features 1 --delta_kernel 1 --init_kernel 1
--init_features_ptr 1 --init_kernel_ptr 1 --final_kernel_ptr 1 --stretch_kernel 1 --final_features_ptr 1 --final_headless 1
--final_headless_ptr 1" --seed_init 0  --alpha 1e-6 --ptr 100 200 400 600 800 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000  --h:int 10000