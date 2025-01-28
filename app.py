import streamlit as st
from PIL import Image
from utils.util import *

st.title("Dress-to-Impress: AI Stylist")

st.write("## Upload an Image or Take a Photo")

photo = st.camera_input("Take a photo")
uploaded_photo = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_photo is None and photo is None:
    st.stop()

file = uploaded_photo if uploaded_photo is not None else photo if photo is not None else None

if file is not None:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Analyze Image
    with st.spinner("Analyzing image..."):
        face_analysis = analyze_image(image)
        face_shape = detect_face_shape(image)
        hair_color = detect_hair_color(image)
        
        st.write("### Facial Analysis")
        st.json(face_analysis)
        st.write("### Face Shape Analysis")
        st.write(face_shape)
        st.write("### Hair Color Detection")
        st.write(hair_color)
        
# User Measurements
st.write("## Enter Your Measurements")
height = st.number_input("Height (cm)", min_value=100, max_value=250)
waist = st.number_input("Waist (cm)", min_value=40, max_value=150)
chest = st.number_input("Chest (cm)", min_value=50, max_value=200)
torso = st.number_input("Torso Length (cm)", min_value=30, max_value=100)

user_measurements = {
    "Height": height,
    "Waist": waist,
    "Chest": chest,
    "Torso": torso
}

if st.button("Get Recommendations"):
    st.write("## Collected Data")
    st.write("### Face Analysis")
    st.json(face_analysis)
    st.write("### Face Shape")
    st.write(face_shape)
    st.write("### Hair Color")
    st.write(hair_color)
    st.write("### User Measurements")
    st.json(user_measurements)
    #with st.spinner("Generating AI recommendations..."):
        #recommendations = get_gpt_recommendations(face_analysis, user_measurements)
        #st.write("## AI Recommendations")
        #st.write(recommendations)

