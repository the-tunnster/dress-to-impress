import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image
from utils.util import *

st.title("Dress-to-Impress: AI Stylist")

with st.expander("Upload an Image or Take a Photo"):
    photo = st.camera_input("Take a photo")
    uploaded_photo = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_photo is None and photo is None:
    st.stop()

file = uploaded_photo if uploaded_photo is not None else photo if photo is not None else None
user_info = {}

if file is not None:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
        
user_info["body_measurements"] = {}

# User Preferences
user_info["hair_color"] = st.color_picker("Select your hair color", "#000000")
        
# User Measurements
st.write("## Enter Your Measurements")
col1, col2 = st.columns(2)
with col1:
    user_info["body_measurements"]["height"] = st.number_input("Height (cm)", min_value=100, max_value=250)
    user_info["body_measurements"]["waist"] = st.number_input("Waist (cm)", min_value=40, max_value=150)

with col2:
    user_info["body_measurements"]["chest"] = st.number_input("Chest (cm)", min_value=50, max_value=200)
    user_info["body_measurements"]["torso"] = st.number_input("Torso Length (cm)", min_value=30, max_value=100)

#Analyszing Image
user_info["face_analysis"] = analyze_image(image)
user_info["face_shape"], forehead_coords, temp = detect_face_shape(image)
user_info["skin_tone"] = detect_skin_tone(image, forehead_coords)
# = detect_hair_color(image, hairline_coords)
st.write(user_info["hair_color"])

if st.button("Get Recommendations"):
    with st.spinner("Generating AI recommendations..."):
        recommendations, keywords = get_gpt_recommendations(user_info)  # Get text + keywords
        image_urls = fetch_fashion_images(keywords)  # Search images

    st.write("## AI Recommendations ✨")
    st.markdown(recommendations)

    st.write("### Extracted Fashion Keywords 🔑")
    st.write(f"_{keywords}_")  # Display extracted keywords

    st.write("### Fashion Inspiration 📸")
    for url in image_urls:
        st.image(url, use_column_width=True)  # Show fetched images


        