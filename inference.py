from argparse import ArgumentParser
from pathlib import Path
import os
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

def main(args):
    emotion_cls_model = EmotionClassifierInference(args.kp, args.feature_dim, args.hidden_dim, args.channels,
                    args.out_dim, args.num_classes, args.using_trans)
    checkpoint = torch.load(args.emotion_cls_model)
    emotion_cls_model.load_state_dict(checkpoint)
    emotion_cls_model.eval()
    
    print("Start webcam")
    #cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #windows
    #cap = cv2.VideoCapture(0, cv2.CAP_V4L2) #linux
    cap = cv2.VideoCapture("./demo/face1_s.mp4")
    print(f"video opened: {cap.isOpened()}")
    img_size = [cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)]
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        query_face_mesh = []
        while cap.isOpened():
            if cv2.waitKey(5) == 27:
                break
            success, frame = cap.read()
            if not success:
                continue
            
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            
    cap.release()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--emotion_cls_model",
        type=Path,
        help="Directory to the emotion classification model file.",
        default="./checkpoints/model.bin",
    )
    
    # model
    parser.add_argument("--frame", type=int, default=27)
    parser.add_argument("--kp", type=int, default=34)
    parser.add_argument("--feature_dim", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--channels", type=int, default=1024)
    parser.add_argument("--out_dim", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument('--using_trans', action='store_true')

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
