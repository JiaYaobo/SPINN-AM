python -u train.py\
            --layer-sizes 100 50  --group-lasso-param 0.9 --data-classes 1 --is-relu 1 --use-bias\
            --num-p 100 --n-obs 2000 --num-groups 2 --batch-size 32 --max-iters 3000 --print-every 500 --seed 520 > logs/train.log 2>&1 &