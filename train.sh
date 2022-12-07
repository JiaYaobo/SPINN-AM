python -u train.py\
            --layer-sizes 100 20  --group-lasso-param 0.9 --data-classes 1 --is-relu 1 --use-bias\
            --num-p 100 --n-obs 2000 --num-groups 1 --batch-size 32 --max-iters 5000 --print-every 1000 --seed 11451 > logs/train.log 2>&1 &