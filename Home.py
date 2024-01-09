import streamlit as st


st.set_page_config(page_title='Attendence System', layout='wide')

st.header('Attendance System Using Face Recognition')

with st.spinner('Loading Model And Connecting to Database'):
    import face_rec

st.success('Model loaded successfully')
st.success('Database Connected to successfully')
st.balloons()
