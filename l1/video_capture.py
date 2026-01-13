import cv2
from retinaface import RetinaFace
from deepface import DeepFace
import tensorflow as tf
import numpy
from emotion_detection import predict_emotion_from_crop

# Initialize webcam
video_path = "/home/rama/cv_bootcamp/pictures/sample.mp4"   # absolute path is safer in WSL
cap = cv2.VideoCapture(video_path)
frame_count = 0
detection_interval = 20
last_faces = []
last_emotions =[]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % detection_interval == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = RetinaFace.detect_faces(rgb)
        last_faces = []
        last_emotions =[]
        if faces:
            for face in faces.values():
                x1, y1, x2, y2 = face["facial_area"]
                #-- face crop--#
                face_crop=frame[y1:y2,x1:x2]
                emotion = predict_emotion_from_crop(face_crop)
                
                last_faces.append((x1, y1, x2, y2))
                last_emotions.append(emotion)
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    """for (x1, y1, x2, y2) in last_faces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)"""
    print(frame_count)
    print(x1)
    #print(len(last_emotions))    
    for (x1, y1, x2, y2), emotion in zip(last_faces, last_emotions):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            emotion,
            (x1+50, y1-20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )
    frame_count += 1


    cv2.imshow("Video", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

"""if not cap.isOpened():
    print("Error: Cannot open video")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break

    cv2.imshow("Video", frame)

    # slow down playback if needed
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break"""
"""while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret :
        print("End of video")
        break
    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break"""

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()