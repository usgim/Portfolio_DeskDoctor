import streamlit as st
import matplotlib.pyplot as plt
import time
import plotly.express as px
import cv2
import numpy as np
import pandas as pd
import keras 
import os
from landmark import landmark
    
def app_model_attempt():
    
    if 'landmark_df' in st.session_state:
        landmark_df = st.session_state.landmark_df
    
    #모델 불러오기
    MODEL_ROOT_DIR = "C:/Users/user/desktop/cv_project/pose_tracking_project_class1/"
    MODEL_NAME = "Maded_model_4.h5"
    MODEL_DIR = os.path.join(MODEL_ROOT_DIR,MODEL_NAME)
    model = keras.models.load_model(MODEL_DIR)

    # sample code 1 (model predicttion to csv)
    st.title('Pose reader')
    # Add a placeholder progress bar
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
        #update the progress bar with each iteration
        latest_iteration.text(f'iteration{i+1}')
        bar.progress(i+1)
        time.sleep(0.01)
              
    if st.sidebar.button("적용 시작"):        
        #global landmark
        df = landmark_df
        if "Unnamed: 0" in df:
            df = df.drop(labels="Unnamed: 0",axis=1)
        df2 = [df]
        df3 = df.copy()
        # Make predictions
        pred = model.predict(df2)    
        results = []
        for i in pred:
            results.append(i.argmax()) # onehot decoding
        output = round(np.mean(results),4)    
        map_dict = {0: "left", 1: "middle", 2: "right"}
        output_2 = output - 1
        if output_2 > 0:
            bias="오른쪽"
        else:
            bias="왼쪽"
        st.title(f"나의 앉은 자세 점수는 {output}입니다.")
        st.write("점수가 1에 가까울수록 바른 앉은자세입니다.")        
        st.title(f"나의 점수 {output}는 {round(output_2,4)}정도 {bias}으로 기울어져있음을 의미합니다.")


        df["Predictions"] = results
        pred_name = []
        for i in results:
            pred_name.append(map_dict[i])
        df["Predictions_name"] = pred_name

        label = df["Predictions_name"].value_counts()             ## bar 차트 그리기
#         st.dataframe( df )
        st.bar_chart(label)
        #plotly pie차트
        df4 = pd.DataFrame()
        for i in range(0,33):
            df5 = df3.iloc[:,[4*i, 4*i + 1]]
            df4 = pd.concat((df4,df5),ignore_index=True,axis=1)

        # 히트맵 구현
        canvas = np.zeros((300, 300), np.uint8)
        canvas.fill(255)

        for i in range(df4.shape[0]):
            for j in range(int(df4.shape[1]/2)):
                x, y = df4.iloc[i,[2*j, 2*j +1]]
                x = x * 300  #denormalize
                y = y * 300
                if x >= 0 and x <= 300 and y >= 0 and y <= 300:
                    # opencv canvas 사용
                    canvas[int(y),int(x)] += 1

        # 히트맵 가시화
        canvas = cv2.applyColorMap(canvas, cv2.COLORMAP_JET)
        st.image(canvas, use_column_width=True)

        df["Counts"] = 1
        df2 = df[["Predictions_name","Counts"]]
        data = df.groupby("Predictions_name").sum()
        fig1 = px.pie(data, values='Counts', names= data.index, title='자세 유형 분포도')
        st.plotly_chart(fig1)

        # 경로를 입력하면 csv 파일로 저장
        save_path = st.text_input("저장하시려면 파일이 저장될 경로와 파일 이름을 입력해주세요!.:")
        if st.button("저장"):
            df.to_csv(f"{save_path}.csv", index=False)
            st.success("파일이 성공적으로 저장되었습니다!")