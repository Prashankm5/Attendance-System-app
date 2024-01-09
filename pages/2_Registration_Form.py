from Home import st, face_rec
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av


# st.set_page_config(page_title='Registration Form')
st.subheader('Registration Form')

#Step1. Collect person Name And Role

person_name = st.text_input(label='Name', placeholder='Full Name')
person_role = st.selectbox(label='Select Your role', options=('Student','Teacher'))


# init Registration Form
registration_form = face_rec.RegistrationForm()


# Collect Facial Embeddings of that person
def video_callback_func(frame):
    img = frame.to_ndarray(format='bgr24') # 3d array bgr
    reg_img, embedding = registration_form.get_embedding(img)
    # two step process
    # 1st step save data into local computer txt
    with open(file='face_embeddings.txt', mode='ab') as f:
        np.savetxt(f, embedding)
    
    
    return av.VideoFrame.from_ndarray(reg_img,format='bgr24')

webrtc_streamer(key='registration',video_frame_callback=video_callback_func,  rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })



# Save the data in redis Database
if st.button('Submit'):
    return_val = registration_form.save_data_in_redis_db(person_name,person_role)
    if return_val == True:
        st.success(f"{person_name} regidtered successfully!")

    elif return_val == 'name_false':
        st.error("Please Enter the name: Name can't be empty or space")
        
    elif return_val == 'file_false':
        st.error("face_embeddings.txt is not found. Please refresh the page and execute again.")

