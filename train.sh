#!bin bash
python train.py --data SMD --win_size 100 --batch_size 32 --epochs 5
python train.py --data NYC --win_size 100 --batch_size 1100 --epochs 1
# 不知道为啥不行
python train.py --data GECCO --win_size 100 --batch_size 256 --epochs 1
python train.py --data DLR --win_size 512  --batch_size 256 --epochs 5
python train.py --data CICIDS --win_size 100 --batch_size 200 --epochs 1
#一般
python train.py --data PSM --win_size 100 --batch_size 32 --epochs 10
python train.py --data Creditcard --win_size 256 --batch_size 600 --epochs 1
# python train.py --data Creditcard --win_size 32 --batch_size 600 --epochs 1
python train.py --data MSL --win_size 100 --batch_size 64 --epochs 8
python train.py --data SWaT --win_size 100 --batch_size 32 --epochs 3


