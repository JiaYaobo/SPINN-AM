python -u train.py\
            --layer-sizes 6   --group-lasso-param 0.1 --data-classes 1 --is-relu 1 --use-bias\
            --num-p 6 --n-obs 2000 --num-groups 2 --batch-size 128 --max-iters 5000 --print-every 500 --seed 888 > logs/train.log 2>&1 &