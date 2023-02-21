import streamlit as st
import pandas as pd

#33개 Landmark = pose landmark 를 가진 컬럼 생성
def landmark():
    num_coords = 33 
    landmarks = []
    for val in range(1, num_coords+1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
    if 'landmark_df' in st.session_state:
        landmark_df = st.session_state.landmark_df
    else:
        landmark_df = pd.DataFrame(columns=landmarks)
    return landmark_df