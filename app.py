import streamlit as st
import numpy as np
import cv2
from tensorflow import keras


CLASS_LABELS = ['Abuse','Arrest','Arson','Assault','Burglary','Explosion','Fighting',"Normal",'RoadAccidents','Robbery','Shooting','Shoplifting','Stealing','Vandalism']
preprocess_fun = keras.applications.densenet.preprocess_input
model = keras.models.load_model(r'.\anomaly_model.h5')

def process_frame(frame):
    
    img = cv2.resize(frame , (64, 64))
    img = np.array(img , dtype='float32')
    img = preprocess_fun(img)
    img = np.expand_dims(img , axis=0)
    val = model.predict(img)
    label = CLASS_LABELS[np.argmax(val)]
    print(label)
    if not label == "Normal":
        print('----------------------------------', label)
  
    
    return label

def app():
    st.set_page_config(page_title='Final Project', layout = 'wide')
    
    st.title('Detector APP')
    uploaded_file = st.file_uploader('Upload a video file', type = ['mp4', 'avi', 'mpv'])
    
    
    if uploaded_file is not None:
        
        file_extension = uploaded_file.name.split('.')[-1]
        with open('temp.'+file_extension , 'wb') as f:
            f.write(uploaded_file.read())
         
        video_stream = cv2.VideoCapture('temp.avi')

        frame_rate = 30
        frame_interval = int(video_stream.get(cv2.CAP_PROP_FPS)/frame_rate)
        if frame_interval <=0 :
            frame_interval = 3000
        
        frame_count = 0
        st.header('video frames')
        while True:
            
            success , frame = video_stream.read()
            
            if not success:
                break
            elif frame_count% frame_interval == 0:
                label = process_frame(frame)
                st.image(frame , caption=f'frame {label}' , use_column_width = True)
            
            
            
            
            
            
if __name__=='__main__':
    app()
            
            
            