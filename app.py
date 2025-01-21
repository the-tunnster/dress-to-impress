import streamlit as st
import cv2
import numpy as np
#import openai
from deepface import DeepFace
from PIL import Image
import mediapipe as mp

def analyze_image(image):
    """Analyze image for facial attributes and hair detection"""
    img_array = np.array(image)
    result = DeepFace.analyze(img_array, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
    return result

def detect_face_shape(image):
    """Detect facial landmarks to determine face shape."""
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    
    image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        # Key landmarks for face shape classification
        landmarks = results.multi_face_landmarks[0].landmark
        
        def landmark_to_array(landmark):
            return np.array([landmark.x, landmark.y])

        jaw_left = landmark_to_array(landmarks[0])   # Left jawline
        jaw_right = landmark_to_array(landmarks[16]) # Right jawline
        chin = landmark_to_array(landmarks[8])       # Chin
        forehead = landmark_to_array(landmarks[10])  # Forehead midpoint
        cheekbone_left = landmark_to_array(landmarks[3])  # Left cheekbone
        cheekbone_right = landmark_to_array(landmarks[13]) # Right cheekbone

        # Convert normalized coordinates to actual pixel positions
        h, w, _ = np.array(image).shape
        jaw_left *= [w, h]
        jaw_right *= [w, h]
        chin *= [w, h]
        forehead *= [w, h]
        cheekbone_left *= [w, h]
        cheekbone_right *= [w, h]
        
        # Calculate distances
        jaw_width = np.linalg.norm(jaw_right - jaw_left)  # Width of jawline
        cheekbone_width = np.linalg.norm(cheekbone_right - cheekbone_left)  # Width at cheekbones
        face_height = np.linalg.norm(chin - forehead)  # Face height

        # Ratios for classification
        jaw_to_face_ratio = jaw_width / face_height
        cheek_to_face_ratio = cheekbone_width / face_height
        jaw_to_cheek_ratio = jaw_width / cheekbone_width

        # Face Shape Classification
        if jaw_to_face_ratio > 1.45 and jaw_to_cheek_ratio > 0.9:
            return "Square"  # Strong jawline, nearly equal jaw & cheek width
        elif cheek_to_face_ratio > 0.9 and jaw_to_cheek_ratio < 0.75:
            return "Heart"  # Wide forehead, narrow jawline
        elif jaw_to_face_ratio < 1.2 and cheek_to_face_ratio > 0.85:
            return "Oval"  # Soft jawline, slightly wider cheeks
        elif cheek_to_face_ratio < 0.75:
            return "Diamond"  # Prominent cheekbones, narrow forehead & jaw
        elif jaw_to_face_ratio < 1.1 and cheek_to_face_ratio < 0.85:
            return "Round"  # Soft, circular features
        elif face_height > cheekbone_width * 1.4:
            return "Oblong"  # Longer face shape with straight sides
        else:
            return "Undefined"

    return "Unable to detect face shape"

def detect_skin_tone(image):
    """Determine the dominant skin tone from the image."""
    image_np = np.array(image)
    roi = image_np[100:200, 100:200]  # Sample a central skin region
    avg_color = np.mean(roi, axis=(0, 1))  # Get average RGB values
    
    return f"Skin Tone: {avg_color} (Placeholder)"  # Map to predefined skin tones

def detect_hair_color(image):
	"""Determine the dominant hair color from the top portion of the image."""
	image_np = np.array(image)
	top_region = image_np[: image_np.shape[0] // 2, :]
	hsv = cv2.cvtColor(top_region, cv2.COLOR_RGB2HSV)
	hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
	dominant_hue = np.argmax(hist)
	# Get an approximate BGR value for reference
	color_bgr = cv2.cvtColor(
		np.uint8([[[dominant_hue, 255, 255]]]),
		cv2.COLOR_HSV2BGR
	)[0][0]
	return f"Dominant Hue: {dominant_hue}, Approx. BGR: {color_bgr.tolist()}"

"""
def get_gpt_recommendations(face_data, user_measurements):
    Generate clothing and hairstyle recommendations using GPT
    prompt = f"Face details: {face_data}. User measurements: {user_measurements}. Recommend clothing and hairstyle."  
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']
"""

def main():
    st.title("Dress-to-Impress: AI Stylist")
    
    st.write("## Upload an Image or Take a Photo")
    photo = st.camera_input("Take a photo")

    # Upload Image
    uploaded_photo = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_photo is not None:
        file = uploaded_photo
    if photo is not None:
        file = photo
    
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

if __name__ == "__main__":
    main()
