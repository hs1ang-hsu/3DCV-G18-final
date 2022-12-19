import sys
import numpy as np
import torch
import cv2

sys.path.append('./emotion_cls')
from common.model import EmotionClassifierInference

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

#===object pose===#
import os
from object_pose.lib.opts import opts
from object_pose.lib.detectors.detector_factory import detector_factory
import glob
import time
import torch.multiprocessing as multiprocessing
multiprocessing.set_start_method('spawn', force=True)
from scipy.spatial.transform import Rotation as R

# facial keypoints
mouth = [0, 61, 17, 291, 13, 14]
nose = [122, 351, 48, 4, 278, 2]
cheek_r = [50, 206, 192]
cheek_l = [280, 426, 416]
eye_r = [133, 159, 130, 145]
eye_l = [362, 386, 359, 374]
eyebrow_r = [107, 105, 46]
eyebrow_l = [336, 334, 276]

iris_r = [473]
iris_l = [468]
kp_list = mouth + nose + cheek_r + cheek_l + eye_r + iris_r + eye_l + iris_l + eyebrow_r + eyebrow_l


def pred_emotion(emotion_cls_model, face_mesh):
    face_mesh = torch.from_numpy(face_mesh).float()
    face_mesh = face_mesh.unsqueeze(0)
    with torch.no_grad():
        pred = emotion_cls_model(face_mesh)
        return pred.max(0)[1]

def get_pose(ret, prev_pose, frame_idx):
    box = ret['boxes']
    if len(box) == 0: #no object detected
        t = None
        quat = None
        return t, quat, prev_pose
    for i in range(0, len(box)):
        t = np.array(box[i][4]['location']) #translation
        quat = np.array(box[i][4]['quaternion_xyzw']) #rotation, in quaternion xyzw
        quat = R.from_quat(quat)
        prev_pose.append((frame_idx, t, quat))
        if len(prev_pose) >= 3:
            del prev_pose[0] #only keep most recent 2 pose
        return t, quat, prev_pose
    

def pose_extrapolation(frame_idx, prev_pose):
    if frame_idx - prev_pose[-2][0] > 30: #previous pose not updating
        t, quat = None, None
        return t, quat
    else:
        #translation: linear extrapolation
        t = prev_pose[-1][1] + (prev_pose[-1][1] - prev_pose[-2][1]) * (frame_idx - prev_pose[-1][0]) / (prev_pose[-1][0] - prev_pose[-2][0])
        #rotation:
        rot = prev_pose[-2][2] * prev_pose[-1][2].inv()
        axis = rot.as_rotvec()
        ang = np.linalg.norm(rot.as_rotvec())
        axisn = axis / ang
        ang = ang *  (frame_idx - prev_pose[-1][0]) / (prev_pose[-1][0] - prev_pose[-2][0])
        ang -= np.pi * int(ang/np.pi)
        axis = axisn * ang
        axis = R.from_rotvec(axis)
        quat = prev_pose[-1][2] * axis
        return t, quat
        

def main(args, meta):
    OBJECT_FLAG = len(args.load_model) > 0

    emotion_cls_model = EmotionClassifierInference(args.kp, args.feature_dim, args.hidden_dim, args.channels,
                    args.out_dim, args.num_classes, args.using_trans)
    checkpoint = torch.load(args.emotion_cls_model)
    emotion_cls_model.load_state_dict(checkpoint)
    emotion_cls_model.eval()
    
    #print("Start webcam")
    #cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #windows
    #cap = cv2.VideoCapture(0, cv2.CAP_V4L2) #linux
    cap = cv2.VideoCapture(0 if args.demo == 'webcam' else args.demo)
    if args.demo == 'webcam':
        print("webcam mode")
    else:
        print("video mode")
        #cap.set(cv2.CAP_PROP_FPS, 20) 
    print(f"video opened: {cap.isOpened()}")
    
    if OBJECT_FLAG:
        image_ext = ['jpg', 'jpeg', 'png', 'webp']
        video_ext = ['mp4', 'mov', 'avi', 'mkv']
        time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'pnp', 'track']
        Detector = detector_factory[args.task]
        detector = Detector(args)
        detector.pause = False
    
    img_size = [cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)]
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        query_face_mesh = []
        
        #object pose
        frame_idx = 0
        st = time.time()
        pool = multiprocessing.Pool(processes=1)
        first_frame = True
        prev_pose = []
        #===========
        
        while cap.isOpened():
            if cv2.waitKey(30) == 27:
                break
            success, frame = cap.read()
            if not success:
                et = time.time()
                print(f"total time: {et-st} sec")
                break
                #continue
            
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #"""
            results = face_mesh.process(frame)
            
            tmp = []
            annotated_image = frame.copy()
            for face_landmarks in results.multi_face_landmarks:
                for i, landmark in enumerate(face_landmarks.landmark):
                    if i in kp_list:
                        tmp.append([landmark.x, landmark.y, landmark.z])
                        x = int(landmark.x * img_size[1])
                        y = int(landmark.y * img_size[0])
                        cv2.circle(annotated_image, (x, y), radius=3, color=(0, 0, 255), thickness=1)
                
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                cv2.imshow('window', annotated_image)
                break
            tmp = np.array(tmp)
            query_face_mesh.append(tmp - tmp[0])
            if len(query_face_mesh) == 27:
                pred = pred_emotion(emotion_cls_model, np.array(query_face_mesh))
                print(pred)
                query_face_mesh.pop(0)
            #"""
            
            #object pose
            """
            #object pose without multiprocessing
            if first_frame:
                ret = detector.run(frame, meta_inp=meta, filename='')
                t, quat, prev_pose = get_pose(ret, prev_pose, frame_idx)
                st2 = time.time()
                first_frame=False
                continue
            ret = detector.run(frame, meta_inp=meta, filename='')
            t, quat, prev_pose = get_pose(ret, prev_pose, frame_idx)
            frame_idx = frame_idx + 1
            """
            
            #"""
            #object pose with multiprocessing
            if first_frame:
                #print("init pose!")
                ret = pool.apply(detector.run, (frame, '', meta)) #blocked execution
                t, quat, prev_pose = get_pose(ret, prev_pose, frame_idx)
                ret = pool.apply_async(detector.run, (frame, '', meta))
                st2 = time.time()
                first_frame = False
            if (ret.ready()):
                #print("pose finish!")
                #get result
                ret = ret.get()
                t, quat, prev_pose = get_pose(ret, prev_pose, frame_idx)
                #new pose from current frame
                ret = pool.apply_async(detector.run, (frame, '', meta))
            else:
                #print("pose running!")
                t, quat = None, None
                pass
            #extrapolation
            if (t is None) and (len(prev_pose) >= 2):
                t, quat = pose_extrapolation(frame_idx, prev_pose)
            frame_idx = frame_idx + 1
            #"""
            #===========
            if t is not None:
                #object render
                print("object pose acquired, should render object here")
                pass
            else:
                #no object, no render
                pass
            
    cap.release()

if __name__ == "__main__":
    opt = opts().parser.parse_args()
    # Default setting
    opt.nms = True
    opt.obj_scale = True
    # PnP related
    meta = {}
    if opt.cam_intrinsic is None:
        meta['camera_matrix'] = np.array(
            [[663.0287679036459, 0, 300.2775065104167], [0, 663.0287679036459, 395.00066121419275], [0, 0, 1]])
        opt.cam_intrinsic = meta['camera_matrix']
    else:
        meta['camera_matrix'] = np.array(opt.cam_intrinsic).reshape(3, 3)

    opt.use_pnp = True
    opt = opts().parse(opt)
    args = opts().init(opt)
    main(args, meta)
