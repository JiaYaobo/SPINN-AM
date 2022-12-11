python -u train.py\
            --layer-sizes 100 30  --group-lasso-param 0.1 --data-classes 1 --is-relu 1 --use-bias\
            --num-p 100 --n-obs 2000 --num-groups 2 --batch-size 256 --max-iters 5000 --print-every 500 --seed 520 > logs/train.log 2>&1 &