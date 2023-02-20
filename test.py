import cv2
import os


cap = cv2.VideoCapture('vtest.avi')
currentframe = 0

frame_rate = 30
frame_interval = int(cap.get(cv2.CAP_PROP_FPS)/frame_rate)
if frame_interval <=0 :
    frame_interval = 30

os.makedirs('data', exist_ok=True)

frame_count = 0


while(cap.isOpened()):
    success , frame = cap.read()
    if not success :
        break
    elif frame_count%frame_interval == 0 :
        cv2.imwrite('./data/dataframe'+str(frame_count) + '.jpg' , frame)
    
    
    cv2.imshow('vid', frame)   
    frame_count +=1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()