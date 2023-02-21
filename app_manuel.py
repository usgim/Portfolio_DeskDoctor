import streamlit as st
from  PIL import Image

def app_manuel():
    def load_image(image_file,size):
        img = Image.open(image_file)
        img=img.resize(size)
        return img
    st.write("""
        DD에서는 아래의 방법을 통해 앉은 자세에 대한 기울어짐을 파악 할 수 있습니다.""")
    st.write(" ")
    st.write("""
    1.화면 왼쪽의 앉은 자세 측정 카테고리에 접속한다.""")
    col1, col2 = st.columns(2)
    with col1:
        st.image(load_image("./seat_menu.jpg",(305,352)))
    with col2:
        st.image(load_image("./human.jpg",(400,350)))
    
    
    st.write("2.카메라를 기준으로 정면으로 책상에 앉는다.")
    st.image(load_image("./start_button.jpg",(500,333)))
    st.write("3.START를 클릭하여 측정을 시작하고 업무 또는 공부를 시작한다.")
    st.write("*권장 측정시간은 30-40분으로 평소처럼 행동한다.")
    st.write("""
    4.측정 시간 후 화면 왼쪽의 자세 평가 카테고리에 접속한다.
\n적용시작을 눌러 나의 앉은 자세에 대한 기울어짐 결과를 확인한다.""")
