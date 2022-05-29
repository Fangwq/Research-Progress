#! /bin/bash
# for first
#rm -rf Stripe1LalphaKernel
#python3 -m grid Stripe1LalphaKernel --n 2 "python3 main.py --device cpu --max_wall 10000 --d 2 --seed_trainset 0 --seed_testset 0 \
#--arch fc --act relu --loss hinge  --L 1 --dataset stripe --ptr 10000 --ptk 1000 --pte 50000 --tau_alpha_crit 1e3 --tau_over_h 1e-3 \
#--save_weight 1 --bias 1 --save_neurons 100 --save_function 1 --final_kernel 1 --final_features 1 --delta_kernel 1 --init_kernel 1
#" --seed_init 0  --alpha 1e-6 1e6 --h:int 10000


# for second
#rm -rf Stripe1LalphaKernel
#python3 -m grid Stripe1LalphaKernel --n 2 "python3 main.py --device cpu --max_wall 30000 --d 2 --seed_trainset 0 --seed_testset 0 \
#--arch fc --act relu --loss hinge  --L 1 --dataset stripe --ptr 10000 --ptk 1000 --pte 50000 --tau_alpha_crit 1e3 --tau_over_h 1e-3 \
#--save_weight 1 --bias 1 --save_neurons 1 --final_kernel 1 --final_features 1 --delta_kernel 1 --init_kernel 1" \
#--seed_init 0  --alpha 1e-6 1e6 --h:int 10000

# for third
#rm -rf Stripe1LalphaKernel2
#python3 -m grid Stripe1LalphaKernel2 --n 2 "python3 main.py --device cpu --max_wall 20000 --d 2 --seed_trainset 0 --seed_testset 0 \
#--arch fc --act relu --loss hinge  --L 1 --dataset stripe --ptr 1000 --ptk 1000 --pte 50000 --tau_alpha_crit 1e3 --tau_over_h 1e-3 \
#--save_weight 1 --bias 1 --save_neurons 100 --save_function 1 2 3 --final_kernel 1 --final_features 1 --delta_kernel 1 --init_kernel 1
#--init_features_ptr 1 --init_kernel_ptr 1 --final_kernel_ptr 1 --stretch_kernel 1 --final_features_ptr 1 --final_headless 1
#--final_headless_ptr 1" --seed_init 0  --alpha 1e-6 1e6 --h:int 10000

## for fourth
#rm -rf Stripe1LalphaKernel3
#python3 -m grid Stripe1LalphaKernel3 --n 2 "python3 main.py --device cpu --max_wall 20000 --d 2 --seed_trainset 0 --seed_testset 0 \
#--arch fc --act relu --loss hinge  --L 1 --dataset stripe --ptr 1000 --ptk 1000 --pte 50000 --tau_alpha_crit 1e3 --tau_over_h 1e-3 \
#--save_weight 1 --bias 1 --save_neurons 100 --save_function 1 2 3 --final_kernel 1 --final_features 1 --delta_kernel 1 --init_kernel 1
#--init_features_ptr 1 --init_kernel_ptr 1 --final_kernel_ptr 1 --stretch_kernel 1 --final_features_ptr 1 --final_headless 1
#--final_headless_ptr 1" --seed_init 0  --alpha 1e-4 1e4 --h:int 10000

## for fifth
#rm -rf Stripe1LalphaKernel4
#python3 -m grid Stripe1LalphaKernel4 --n 2 "python3 main.py --device cpu --max_wall 20000 --d 2 --seed_trainset 0 --seed_testset 0 \
#--arch fc --act relu --loss hinge  --L 1 --dataset stripe --ptr 5000 --ptk 5000 --pte 50000 --tau_alpha_crit 1e3 --tau_over_h 1e-3 \
#--save_weight 1 --bias 1 --save_neurons 100 --save_function 1 2 3 --final_kernel 1 --final_features 1 --delta_kernel 1 --init_kernel 1
#--init_features_ptr 1 --init_kernel_ptr 1 --final_kernel_ptr 1 --stretch_kernel 1 --final_features_ptr 1 --final_headless 1
#--final_headless_ptr 1" --seed_init 0  --alpha 1e-4 1e4 --h:int 10000

##for sixth
#rm -rf Stripe1LalphaKernel5
#python3 -m grid Stripe1LalphaKernel5 --n 2 "python3 main.py --device cpu --max_wall 20000 --d 2 --seed_trainset 0 --seed_testset 0 \
#--arch fc --act relu --loss hinge  --L 1 --dataset stripe --ptr 5000 --ptk 5000 --pte 50000 --tau_alpha_crit 1e3 --tau_over_h 1e-3 \
#--save_weight 1 --bias 1 --save_neurons 100 --save_function 1 2 3 --final_kernel 1 --final_features 1 --delta_kernel 1 --init_kernel 1
#--init_features_ptr 1 --init_kernel_ptr 1 --final_kernel_ptr 1 --stretch_kernel 1 --final_features_ptr 1 --final_headless 1
#--final_headless_ptr 1" --seed_init 0  --alpha 1e-2 1e2 --h:int 10000

## for seventh
#rm -rf Stripe1LalphaKernel6
#python3 -m grid Stripe1LalphaKernel6 --n 2 "python3 main.py --device cpu --max_wall 30000 --d 2 --seed_trainset 0 --seed_testset 0 \
#--arch fc --act relu --loss hinge  --L 1 --dataset stripe --ptr 5000 --ptk 5000 --pte 50000 --tau_alpha_crit 1e3 --tau_over_h 1e-3 \
#--save_weight 1 --bias 1 --var_bias 1 --save_neurons 100 --save_function 1 2 3 --final_kernel 1 --final_features 1 --delta_kernel 1 --init_kernel 1
#--init_features_ptr 1 --init_kernel_ptr 1 --final_kernel_ptr 1 --stretch_kernel 1 --final_features_ptr 1 --final_headless 1
#--final_headless_ptr 1" --seed_init 0  --alpha 1e-6 --h:int 10000


## for debug
#rm -rf Stripe1LalphaKernel_test
#python3 -m grid Stripe1LalphaKernel_test --n 1 "python3 main.py --device cpu --max_wall 30 --d 2 --seed_trainset 0 --seed_testset 0 \
#--arch fc --act relu --loss hinge  --L 1 --dataset stripe --ptr 1000 --ptk 1000 --pte 50000 --tau_alpha_crit 1e3 --tau_over_h 1e-3 \
#--save_weight 1 --bias 1 --save_neurons 100 --save_function 1 2 3 --final_kernel 1 --final_features 1 --delta_kernel 1 --init_kernel 1
#--init_features_ptr 1 --init_kernel_ptr 1 --final_kernel_ptr 1 --stretch_kernel 1 --final_features_ptr 1 --final_headless 1
#--final_headless_ptr 1" --seed_init 0  --alpha 1e-6 --h:int 10000


# for debug
rm -rf Stripe1LalphaKernel_test1
python3 -m grid Stripe1LalphaKernel_test1 --n 2 "python3 main.py --device cpu --max_wall 300 --d 2 --seed_trainset 0 --seed_testset 0 \
--arch fc --act relu --loss hinge  --L 1 --dataset stripe --ptr 5000 --ptk 5000 --pte 50000 --tau_alpha_crit 1e3 --tau_over_h 1e-3 \
--save_weight 1 --bias 1 --var_bias 1 --save_neurons 100 --save_function 1 2 3 --final_kernel 1 --final_features 1 --delta_kernel 1 --init_kernel 1
--init_features_ptr 1 --init_kernel_ptr 1 --final_kernel_ptr 1 --stretch_kernel 1 --final_features_ptr 1 --final_headless 1
--final_headless_ptr 1" --seed_init 0  --alpha 1e-6 --h:int 10000
