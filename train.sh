python -u train.py\
            --layer-sizes 1000 20  --group-lasso-param 0.1 --data-classes 1 --is-relu 1 --use-bias\
            --num-p 1000 --n-obs 2000 --num-groups 2 --batch-size 256 --max-iters 5000 --print-every 500 --seed 888 > logs/train.log 2>&1 &