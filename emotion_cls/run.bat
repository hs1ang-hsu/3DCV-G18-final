REM python train.py
REM python train.py --feature_dim 16 --using_trans
REM python train.py --dataset dataset/data_p64.npz --kp 64 --feature_dim 16
REM python train.py --dataset dataset/data_p64.npz --kp 64 --feature_dim 16 --using_trans

:: naive
REM python train.py

:: transfomer
REM python train.py --dataset dataset/data_p64.npz --kp 64 --feature_dim 16 REM with tcn
REM python train.py --dataset dataset/data_p64.npz --kp 64 --feature_dim 16 REM with pure transformer

:: original
REM python train.py --dataset dataset/data_p64.npz --kp 64 --feature_dim 16 --channels 2048 --num_epoch 30
REM python train.py --feature_dim 16 --channels 2048 --num_epoch 30
REM python train.py --feature_dim 16 --using_trans --channels 2048 --num_epoch 30


REM \item MLP + MLP
REM \item MLP + TDCN
REM \item T-Net + TDCN
REM \item transformer + transformer

:: frame-wise
:: exp1: train: 0.415270, eval: 0.531708, time: 144.34it/s epoch_60
:: exp2: train: 0.395208, eval: 0.479274, time: 74.34it/s
:: exp3: train: 0.412774, eval: 0.439526, time: 111.87it/s
:: exp4: train: 0.357979, eval: 0.415124, time: 51.13it/s
:: exp5: train: 0.160785, eval: 0.223198, time: 183.43it/s

:: exp9: train: 0.394048, eval: 0.471834, time: 68.83it/s best
:: exp8: train: 0.408824, eval: 0.470738, time: 61.40it/s best

