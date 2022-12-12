python -u train_single_spinn.py\
            --layer-sizes 20 20  --group-lasso-param 0.1 --data-classes 1 --is-relu 1 --use-bias --adam-learn-rate 1e-2\
            --num-p 20 --n-obs 2000 --num-groups 1 --batch-size 64 --max-iters 5000 --print-every 500 --seed 888 > logs/train2.log 2>&1 &