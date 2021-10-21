"""
face detection with mtcnn on a video
"""
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
import cv2

# create the detector, using default weights
detector = MTCNN()

# function to draw boxes and landmarks
def draw_boxes_with_landmarks(frame, result_list, min_conf):
    for result in result_list:
        # draw only face with confidence more than min_conf 
        if result['confidence'] < min_conf:
            continue
        # get box info
        x, y, w, h = result['box']
        # draw face rectangle
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
        # draw face landmarks
        for _, value in result['keypoints'].items():
            cv2.circle(frame, value, 2, (0,0,255), -1)
    return frame

# read video
cap = cv2.VideoCapture('data/people.mp4')

# show video
while cap.isOpened():
    success, img = cap.read()
    if success :
        # detect faces in the image
        faces = detector.detect_faces(img)
        img = draw_boxes_with_landmarks(img, faces, 0.9)
        cv2.imshow("Result", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
             break
    else :
        break
cap.release()
cv2.destroyAllWindows()