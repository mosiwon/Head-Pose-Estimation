import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pickle
import glob
import scipy.io

import psutil
import time
from collections import deque

import argparse

# 인자 파서 설정
parser = argparse.ArgumentParser(description="Face Angle Detection")
parser.add_argument('-d', '--display', action='store_true', help="CPU Usage and RAM Usage Display")
parser.add_argument('-c', '--coordination', action='store_true', help="Coordination Display")
args = parser.parse_args()

face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cpu_usage_history = deque(maxlen=10)
cap = cv2.VideoCapture(0)  # From Camera

model = pickle.load(open('./model.pkl', 'rb'))

cols = []
for pos in ['nose_', 'forehead_', 'left_eye_', 'mouth_left_', 'chin_', 'right_eye_', 'mouth_right_']:
    for dim in ('x', 'y'):
        cols.append(pos+dim)
        
def extract_features(img, face_mesh):
    NOSE = 1
    FOREHEAD = 10
    LEFT_EYE = 33
    MOUTH_LEFT = 61
    CHIN = 199
    RIGHT_EYE = 263
    MOUTH_RIGHT = 291

    result = face_mesh.process(img)
    face_features = []
    
    if result.multi_face_landmarks != None:
        for face_landmarks in result.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [FOREHEAD, NOSE, MOUTH_LEFT, MOUTH_RIGHT, CHIN, LEFT_EYE, RIGHT_EYE]:
                    face_features.append(lm.x)
                    face_features.append(lm.y)

    return face_features

def normalize(poses_df):
    normalized_df = poses_df.copy()
    
    for dim in ['x', 'y']:
        # Centerning around the nose 
        for feature in ['forehead_'+dim, 'nose_'+dim, 'mouth_left_'+dim, 'mouth_right_'+dim, 'left_eye_'+dim, 'chin_'+dim, 'right_eye_'+dim]:
            normalized_df[feature] = poses_df[feature] - poses_df['nose_'+dim]
        
        
        # Scaling
        diff = normalized_df['mouth_right_'+dim] - normalized_df['left_eye_'+dim]
        for feature in ['forehead_'+dim, 'nose_'+dim, 'mouth_left_'+dim, 'mouth_right_'+dim, 'left_eye_'+dim, 'chin_'+dim, 'right_eye_'+dim]:
            normalized_df[feature] = normalized_df[feature] / diff
    
    return normalized_df

def draw_axes(img, pitch, yaw, roll, tx, ty, size=50):
    yaw = -yaw
    rotation_matrix = cv2.Rodrigues(np.array([pitch, yaw, roll]))[0].astype(np.float64)
    axes_points = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=np.float64)
    axes_points = rotation_matrix @ axes_points
    axes_points = (axes_points[:2, :] * size).astype(int)
    axes_points[0, :] = axes_points[0, :] + tx
    axes_points[1, :] = axes_points[1, :] + ty
    
    new_img = img.copy()
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 0].ravel()), (255, 0, 0), 3)    
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 1].ravel()), (0, 255, 0), 3)    
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 2].ravel()), (0, 0, 255), 3)
    return new_img

def get_average_cpu_usage(cpu_usage_history, new_usage, max_len=10):
    cpu_usage_history.append(new_usage)
    if len(cpu_usage_history) > max_len:
        cpu_usage_history.popleft()
    return sum(cpu_usage_history) / len(cpu_usage_history)


def renderFacePoseStats(img, text, pitch_pred_deg, yaw_pred_deg, roll_pred_deg, average_cpu_usage, ram_usage):
    cv2.putText(img, text, (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(img, f'Pitch: {pitch_pred_deg:.2f}', (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(img, f'Yaw: {yaw_pred_deg:.2f}', (25, 175), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(img, f'Roll: {roll_pred_deg:.2f}', (25, 225), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(img, f"CPU: {average_cpu_usage:.1f}%", (25, 275), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(img, f"RAM: {ram_usage}%", (25, 325), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return img

while(cap.isOpened()):

    ret, img = cap.read()
    if ret:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 1)
        img_h, img_w, img_c = img.shape
        text = ''

        face_features = extract_features(img, face_mesh)
        if len(face_features):
            face_features_df = pd.DataFrame([face_features], columns=cols)
            face_features_normalized = normalize(face_features_df)
            pitch_pred, yaw_pred, roll_pred = model.predict(face_features_normalized).ravel()
            nose_x = face_features_df['nose_x'].values * img_w
            nose_y = face_features_df['nose_y'].values * img_h

            if args.coordination:
                img = draw_axes(img, pitch_pred, yaw_pred, roll_pred, int(nose_x), int(nose_y))
                        
            if pitch_pred > 0.3:
                text = 'Top'
                if yaw_pred > 0.3:
                    text = 'Top Left'
                elif yaw_pred < -0.3:
                    text = 'Top Right'
            elif pitch_pred < -0.3:
                text = 'Bottom'
                if yaw_pred > 0.3:
                    text = 'Bottom Left'
                elif yaw_pred < -0.3:
                    text = 'Bottom Right'
            elif yaw_pred > 0.3:
                text = 'Left'
            elif yaw_pred < -0.3:
                text = 'Right'
            else:
                text = 'Forward'

        # rad to degree
        pitch_pred_deg = pitch_pred * 180 / np.pi
        yaw_pred_deg = yaw_pred * 180 / np.pi
        roll_pred_deg = roll_pred * 180 / np.pi
                
        new_cpu_usage = psutil.cpu_percent(interval=0)
        average_cpu_usage = get_average_cpu_usage(cpu_usage_history, new_cpu_usage)
        ram_usage = psutil.virtual_memory().percent
        
        if args.display:
                img = renderFacePoseStats(img, text, pitch_pred_deg, yaw_pred_deg, roll_pred_deg, average_cpu_usage, ram_usage)


        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('img', img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()