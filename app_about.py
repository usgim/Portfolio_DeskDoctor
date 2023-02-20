import streamlit as st
from io import BytesIO
import requests
from  PIL import Image

def app_about():
    def link_image(link,size):
        res = requests.get(link)
        image = Image.open(BytesIO(res.content))
        image=image.resize(size)
        st.image(image)
    st.write("어떤 모습으로 책상에 앉아 계시나요?")

    link_image('http://cdn.edujin.co.kr/news/photo/201902/23214_42778_4153.jpg',(500,333))
    st.write("혹시 지금도 이 자세로 화면을 보고 계신다면")
    link_image('http://i-leg.kr/images/e4_2.jpg',(500,333))
    st.write("당신의 허리척추 건강은 안녕하신가요?")
    col1, col2 = st.columns(2)
    with col1:
        link_image('https://cdn.imweb.me/thumbnail/20210317/43e8d57835d5f.jpg',(250,205))
    with col2:
        link_image('https://www.ortopedicka-ambulance.cz/images/upload/skolioza.jpg',(250,205))
    st.write("오랜시간 앉은 자세에서의 불균형은 척추건강을 위협합니다.")
    st.write("잘 앉는 것부터 시작하는게 어떨까요? 잘 앉기만 해도 허리 건강을 지킬수 있습니다.")
    st.write("DD와 함께 나의 앉은 자세를 확인해보고 허리건강을 지켜보세요.")
    link_image("https://www.flexispot.es/media/magefan_blog/LORRAINE_2_800_7_115_1_1.jpg",(500,333))