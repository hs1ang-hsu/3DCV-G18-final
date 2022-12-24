# 3DCV Final

## Emotion classification

### Environment

- Package
	- pytorch
	- einops
	- numpy
	- timm
	- collections

- Download dataset
	- Download the dataset from Google drive and put it in the dataset directory.
	https://drive.google.com/drive/folders/1tKcpg4yJ652zaM68L_R6fiS_JLmfiPny?usp=share_link
	- run ```python download.py```

### Train

```shell
python train.py \
    [--frame 27] \
    [--kp 34] \
    [--feature_dim 3] \
    [--hidden_dim 256] \
    [--channels 1024] \
    [--out_dim 64] \
    [--num_classes 7] \
    [--using_trans] \
    [--lr 1e-3] \
    [--lrd 0.95] \
    [--batch_size 64] \
    [--device cuda] \
    [--num_epoch 60] \
	[--export_training_curves True]
```
- frame: the number of frame as video sequence input
- kp: the number of facial keypoint
- feature_dim: the expanded dimension of input feature
- hidden_dim: the dimension of hidden layer in transform net
- channels: the expanded dimension of spatial feature in temporal convolution net
- out_dim: the output dimension of temporal convolution net
- num_classes: the total class of facial emotions
- using_trans: whether to use transform net
- lr: learning rate
- lrd: learning rate drop
- batch_size: batch size
- device: specify from "cpu, cuda, cuda:0, cuda:1"
- num_epoch: the number of training epoch
- export_training_curves: whether to plot training curves

### Inference
```shell
python inference.py
python inference.py --emotion_cls_model checkpoint/model_Tnet.bin --feature_dim 16 --using_trans
```

## Object Pose Estimation

### Environment
- Download pretrain model
	- Download the pretrain model here
	https://drive.google.com/file/d/1tmp_iob-mx-mrbdpVS0K7sKE12ERJy7X/view?usp=share_link 

### Inference
demo video
```shell
python inference_o.py --device cuda --demo ./demo_vid/mug1_s.mp4 --load_model <path to pretrain model>
```
webcam
```shell
python inference_o.py --device cuda --demo 'webcam' --load_model <path to pretrain model>
```
