import mediapipe as mp
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import numpy as np
import pandas as pd
import av
import queue
import time

from landmark import landmark

landmark_df= landmark()

def app_pose_tracking():

    #drawing style 설정
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    holistic = mp_holistic.Holistic(
        enable_segmentation=True,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    #실시간 프로세싱 코드
    def process(image):
        image.flags.writeable = False 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        # landmark 그리기
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                     )
        try : 
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
            row = pose_row
        except:
            pass
            #row = np.zeros(len(landmarks))
        return cv2.flip(image, 1), row
        

    # 서버
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # webrtc_streamer의 video_processor_factory 부분.
    class VideoProcessor:
        def recv(self, frame): 
            img = frame.to_ndarray(format="bgr24")
            img = process(img)[0]
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    def callback(frame):
        img = frame.to_ndarray(format="bgr24")
   
        img = process(img)[0]
        try:
            row = process(img)[1]
        except :
            time.sleep(3)
        #df = pd.DataFrame(row).T
        #df.columns=landmarks
        #pred = model.predict(df)
        #data_queue.put(pred)
        
        global landmark_df
        landmark_df = landmark_df.append(pd.Series(row,index=landmark_df.columns),ignore_index=True)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    # 캠이 실행되는 부분
    webrtc_ctx = webrtc_streamer(
        key="WYH",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory = VideoProcessor,
        video_frame_callback = callback,
        async_processing=True,
    )
    if webrtc_ctx.state.playing:
        labels_placeholder_2 = st.empty()
        while True:
            labels_placeholder_2.dataframe(landmark_df)
            st.session_state.landmark_df = landmark_df
    #return landmark_df
    # # 사이드바에 체크박스 : 현재 좌표에서 모델을 적용시킨 값을 즉석으로 보여주고, 데이터프레임에 어떤 값이 들어가는지 가시적으로 보여줌
    # if st.sidebar.checkbox("Show the detected labels", value=True):
    #     if webrtc_ctx.state.playing:
    #         #labels_placeholder_1 = st.empty()
    #         labels_placeholder_2 = st.empty()

    #         while True:
    #             #try:
    #                 #result = data_queue.get(timeout=1.0)
    #             #except queue.Empty:
    #                 #result = None
    #             #labels_placeholder_1.table(result)
    #             labels_placeholder_2.dataframe(landmark_df)
                
    #             # session_state 에 landmark_df 값을 저장
    #             #st.session_state.landmark_df = landmark_df
    # # return webrtc_ctx, landmark_df