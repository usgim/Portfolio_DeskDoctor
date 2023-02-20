from streamlit_lottie import st_lottie
import streamlit as st
import requests

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def app_home():
    st.header("1조 pose_tracking_project")
    lottie_url = "https://assets4.lottiefiles.com/packages/lf20_3dw8ed6q.json"
    lottie_json = load_lottieurl(lottie_url)
    st.subheader("프로젝트 \"DD\" 메인화면입니다.")
    st_lottie(lottie_json)