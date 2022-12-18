# 3DCV Final

## Emotion classification + object pose estimation

Only list the change about object pose
object pose estimation origin repo: 
https://github.com/NVlabs/CenterPose
### Environment
- Package 
	- pytorch=1.11.0
	- torchvision=0.12.0
	- other follow origin repo requirement.txt

### Inference
- Download pre-trained model [here](https://drive.google.com/drive/folders/16HbCnUlCaPcTg4opHP_wQNPsWouUlVZe) to checkpoints/
- image or video
`python inference_o.py --device cuda --demo <path_to image/video> --load_model <path_to_model>`
- webcam
`python inference_o.py --device cuda --demo 'webcam' --load_model <path_to_model>`
