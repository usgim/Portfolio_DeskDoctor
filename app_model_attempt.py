import streamlit as st
import matplotlib.pyplot as plt
import time
import plotly.express as px
import cv2
import numpy as np
import pandas as pd
import keras 
import os
    
def app_model_attempt():
    
    if 'landmark_df' in st.session_state:
        landmark_df = st.session_state.landmark_df
    
    #모델 불러오기
    MODEL_ROOT_DIR = "C:/Users/user/DD/Page2/"
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

#     # load the model
#     st.title("Load the Model")
#     stream = st.file_uploader('TF.Keras model file (.h5py.zip)', type='zip')
#     if stream is not None:
#       myzipfile = zipfile.ZipFile(stream)
#       with tempfile.TemporaryDirectory() as tmp_dir:
#         myzipfile.extractall(tmp_dir)
#         root_folder = myzipfile.namelist()[0] # e.g. "model.h5py"
#         model_dir = os.path.join(tmp_dir, root_folder)
#         #st.info(f'trying to load model from tmp dir {model_dir}...')
#         model = tf.keras.models.load_model(model_dir)

#     # Load data from CSV
#     st.title("Upload your csv file.")
#     file_path = st.file_uploader("Upload a CSV file", type="csv")
#     if file_path is not None:

    if st.sidebar.button("측정값 확인"):
        #global landmark
        a1 = st.empty()
        a1.dataframe(landmark_df)
                
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
            bias="right"
        else:
            bias="left"
        st.title(f"Your Pose value is {output}. 0<=Value<=2, 0=left, 1=middle, 2=right. The value close to 1 is good.")
        st.title(f"Your Pose is {map_dict[round(output)]}, but {round(output_2,4)} biased to {bias}.")

        df["Predictions"] = results
        pred_name = []
        for i in results:
            pred_name.append(map_dict[i])
        df["Predictions_name"] = pred_name

        label = df["Predictions_name"].value_counts()             ## bar 차트 그리기
        st.dataframe( df.head() )
        st.bar_chart(label)
        #plotly pie차트
        df4 = pd.DataFrame()
        for i in range(0,33):
            df5 = df3.iloc[:,[4*i, 4*i + 1]]
            df4 = pd.concat((df4,df5),ignore_index=True,axis=1)
        # df4 구성
        # 이제 df4 에다가 x*width, y*height 해서 좌표 구해주면 되겠다. 

        # Create a blank image to be used as the canvas for the heatmap
        canvas = np.zeros((300, 300), np.uint8)
        canvas.fill(255)
        # Loop through the landmark data
        for i in range(df4.shape[0]):
            for j in range(int(df4.shape[1]/2)):
                x, y = df4.iloc[i,[2*j, 2*j +1]]
                x = x * 300  #denormalize
                y = y * 300
                if x >= 0 and x <= 300 and y >= 0 and y <= 300:
                    # Plot a circle on the canvas using OpenCV's circle function
                    canvas[int(y),int(x)] += 1

        # Show the heatmap
        canvas = cv2.applyColorMap(canvas, cv2.COLORMAP_JET)
        st.image(canvas, use_column_width=True)

        df["Counts"] = 1
        df2 = df[["Predictions_name","Counts"]]
        data = df.groupby("Predictions_name").sum()
        fig1 = px.pie(data, values='Counts', names= data.index, title='자세 유형 분포도')
        st.plotly_chart(fig1)

        # Save predictions to new CSV file
        save_path = st.text_input("If you want to save the file, Please write the file path down.:")
        if st.button("Save"):
            df.to_csv(f"{save_path}.csv", index=False)
            st.success("File saved successfully!")