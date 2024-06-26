from FaceAngleDetector import FaceAngleDetector

import cv2
import numpy as np 
import psutil

# FaceAngleDetector 클래스 인스턴스화
detector = FaceAngleDetector()

# 예시 1: 이미지 파일에서 얼굴 각도 및 방향을 계산
image_path = 'image.jpg'
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 얼굴 각도 및 방향 계산
pitch_pred, yaw_pred, roll_pred, nose_x, nose_y = detector.get_face_angles(img)
if pitch_pred is not None:
    # 얼굴 방향 텍스트 생성
    text = detector.get_face_direction(pitch_pred, yaw_pred)
    # 라디안을 각도로 변환
    pitch_pred_deg = pitch_pred * 180 / np.pi
    yaw_pred_deg = yaw_pred * 180 / np.pi
    roll_pred_deg = roll_pred * 180 / np.pi

    # 얼굴 방향 축 그리기
    img = detector.draw_axes(img, pitch_pred, yaw_pred, roll_pred, int(nose_x * img.shape[1]), int(nose_y * img.shape[0]))

    # 얼굴 방향 텍스트 및 각도 그리기
    img = detector.render_face_pose_stats(img, text, pitch_pred_deg, yaw_pred_deg, roll_pred_deg)

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 예시 2: 웹캠에서 프레임을 읽어 얼굴 각도와 방향을 계산(카메라를 사용하면 실행이 더 느려질 수 있습니다.)
cap = cv2.VideoCapture(-1)  # 웹캠 인덱스
while cap.isOpened():
    ret, img = cap.read()
    if ret:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 1)
        
        # 얼굴 각도 및 방향 계산
        pitch_pred, yaw_pred, roll_pred, nose_x, nose_y = detector.get_face_angles(img)
        
        if pitch_pred is not None:
            
            # 얼굴 방향 텍스트 생성
            text = detector.get_face_direction(pitch_pred, yaw_pred)
            pitch_pred_deg = pitch_pred * 180 / np.pi
            yaw_pred_deg = yaw_pred * 180 / np.pi
            roll_pred_deg = roll_pred * 180 / np.pi

            # 얼굴 방향 축 그리기
            img = detector.draw_axes(img, pitch_pred, yaw_pred, roll_pred, int(nose_x * img.shape[1]), int(nose_y * img.shape[0]))

            img = detector.render_face_pose_stats(img, text, pitch_pred_deg, yaw_pred_deg, roll_pred_deg)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('img', img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()


