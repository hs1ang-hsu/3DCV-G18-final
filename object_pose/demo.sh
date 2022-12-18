#example
#python demo.py --demo ../images/CenterPose/chair/ --arch dlav1_34 --load_model ../models/CenterPoseTrack/chair_v1_140.pth
#python demo.py --demo ../images/CenterPoseTrack/bike_batch-1_25.mp4 --arch dlav1_34 --load_model ../models/CenterPoseTrack/bike_v1_140.pth --tracking_task
#python demo.py --demo webcam --arch dlav1_34 --load_model ../models/CenterPoseTrack/chair_v1_140.pth

#images
#python demo.py --demo ../images/CenterPose/my_chair/ --arch dlav1_34 --load_model ../models/CenterPoseTrack/chair_v1_140.pth --debug 4

#video
python demo.py --demo ../images/CenterPose/my_chair_vid/004_s.mp4 --arch dlav1_34 --load_model ../models/CenterPoseTrack/chair_v1_140.pth
