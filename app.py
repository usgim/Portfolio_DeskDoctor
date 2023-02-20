#메인 함수

#사용자 모듈 정의
from app_home import app_home
from app_about import app_about
from app_manuel import app_manuel
from app_pose_tracking import app_pose_tracking
from app_model_attempt import app_model_attempt

from landmark import landmark

#외부 라이브러리
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

#css 적용
with open('C:/Users/user/DD/Page2/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    
    home = 'Home'
    about = 'DD를 소개합니다!'
    manuel = '사용법'
    pose_tracking_page = "앉은 자세 측정"
    model_attempt_page = "앉은 자세 평가"

    with st.sidebar:
        app_mode = option_menu("Desk Doctor!", [home,about,manuel, pose_tracking_page, model_attempt_page],
                         icons=['house','bi bi-chat-right-dots', 'bi bi-info-circle-fill', 'camera fill', 'kanban'],
                         menu_icon="person-workspace", 
                         default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#FAFAFA"},
        "icon": {"color": "orange", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02AB21"},
    }
    )

    st.subheader(app_mode)
    if app_mode == home:
        app_home()
    elif app_mode == about:
        app_about()
    elif app_mode == manuel:
        app_manuel()
    elif app_mode == pose_tracking_page:
        app_pose_tracking()
    elif app_mode == model_attempt_page:
        app_model_attempt()

if __name__ == "__main__":
    main()

