import numpy as np
from deepface import DeepFace
import cv2
import mediapipe as mp
import openai
from openai import OpenAI
import os
import requests

# OPENAI Key & Client
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
client = OpenAI()

def analyze_image(image):
    """Analyze image for facial attributes and hair detection"""
    img_array = np.array(image)
    result = DeepFace.analyze(img_array, actions=['age', 'gender', 'race'], enforce_detection=False)
    return result

def detect_face_shape(image):
    """Detect facial landmarks to determine face shape and extract forehead & hairline coordinates."""
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    
    image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = face_mesh.process(image_rgb)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        def landmark_to_array(landmark):
            return np.array([landmark.x, landmark.y])

        jaw_left = landmark_to_array(landmarks[0])   # Left jawline
        jaw_right = landmark_to_array(landmarks[16]) # Right jawline
        chin = landmark_to_array(landmarks[8])       # Chin
        forehead = landmark_to_array(landmarks[10])  # Forehead midpoint
        hairline = landmark_to_array(landmarks[152]) # Hairline top point
        cheekbone_left = landmark_to_array(landmarks[3])  # Left cheekbone
        cheekbone_right = landmark_to_array(landmarks[13]) # Right cheekbone

        # Convert normalized coordinates to actual pixel positions
        h, w, _ = np.array(image).shape
        jaw_left *= [w, h]
        jaw_right *= [w, h]
        chin *= [w, h]
        forehead *= [w, h]
        hairline *= [w, h]
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
            face_shape = "Square"  # Strong jawline, nearly equal jaw & cheek width
        elif cheek_to_face_ratio > 0.9 and jaw_to_cheek_ratio < 0.75:
            face_shape = "Heart"  # Wide forehead, narrow jawline
        elif jaw_to_face_ratio < 1.2 and cheek_to_face_ratio > 0.85:
            face_shape = "Oval"  # Soft jawline, slightly wider cheeks
        elif cheek_to_face_ratio < 0.75:
            face_shape = "Diamond"  # Prominent cheekbones, narrow forehead & jaw
        elif jaw_to_face_ratio < 1.1 and cheek_to_face_ratio < 0.85:
            face_shape = "Round"  # Soft, circular features
        elif face_height > cheekbone_width * 1.4:
            face_shape = "Oblong"  # Longer face shape with straight sides
        else:
            face_shape = "Undefined"

        return face_shape, forehead.astype(int), hairline.astype(int)

    return "Unable to detect face shape", None, None

def detect_skin_tone(image, forehead_coords):
    """Determine the dominant skin tone from the forehead region."""
    if forehead_coords is None:
        return "Skin Tone: Unable to determine (No forehead detected)"

    image_np = np.array(image)
    x, y = forehead_coords

    # Define a small region around the forehead coordinates
    roi_size = 20  # Pixels
    roi = image_np[max(0, y - roi_size): min(image_np.shape[0], y + roi_size),
                   max(0, x - roi_size): min(image_np.shape[1], x + roi_size)]

    if roi.size == 0:
        return "Skin Tone: Unable to determine (ROI too small)"

    avg_color = np.mean(roi, axis=(0, 1))  # Get average RGB values
    
    return f"Skin Tone: {avg_color} (Placeholder)"  # Map to predefined skin tones

def detect_hair_color(image, hairline_coords):
    """Determine the dominant hair color from the hairline region."""
    if hairline_coords is None:
        return "Hair Color: Unable to determine (No hairline detected)"

    image_np = np.array(image)
    x, y = hairline_coords

    # Define a region around the hairline coordinates
    roi_size = 30  # Pixels
    hairline_region = image_np[max(0, y - roi_size): min(image_np.shape[0], y + roi_size),
                               max(0, x - roi_size): min(image_np.shape[1], x + roi_size)]

    if hairline_region.size == 0:
        return "Hair Color: Unable to determine (ROI too small)"

    # Convert to HSV and compute histogram
    hsv = cv2.cvtColor(hairline_region, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    dominant_hue = np.argmax(hist)

    # Get an approximate BGR value for reference
    color_bgr = cv2.cvtColor(
        np.uint8([[[dominant_hue, 255, 255]]]),
        cv2.COLOR_HSV2BGR
    )[0][0]
    
    return f"Dominant Hue: {dominant_hue}, Approx. BGR: {color_bgr.tolist()}"


def get_gpt_recommendations(user_info):
    """Generate clothing and hairstyle recommendations using OpenAI's GPT-4 and extract keywords directly."""
    
    prompt = f"""
    You are a professional fashion stylist. Based on the user's facial features, face shape, and hair color, provide personalized hairstyle and clothing recommendations.

    ### User Data:
    - **Face Analysis**: {user_info["face_analysis"]}
    - **Face Shape**: {user_info["face_shape"]}
    - **Hair Color**: {user_info["hair_color"]}
    - **Body Measurements**: {user_info["body_measurements"]}

    ### Instructions:
    1. Provide **personalized recommendations** in the following structured format:
       - **Hairstyle:** (best hairstyles for this face shape)
       - **Clothing Styles:** (best outfit choices based on proportions)
       - **Color Palette:** (colors that complement hair and skin tone)
       - **Extra Fashion Tips:** (optional additional advice)

    2. **Extract important fashion-related keywords from your recommendations** and return them as a **comma-separated list** at the end under:
       - **Keywords:** (list of keywords)

    Only return the formatted response with recommendations and the keyword list.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        recommendations = response.choices[0].message.content

        # Extract keywords from the structured response
        keywords_line = recommendations.split("**Keywords:**")[-1].strip() if "**Keywords:**" in recommendations else "No keywords found"

        return recommendations, keywords_line  # Return both recommendations and extracted keywords

    except Exception as e:
        return f"Error: {str(e)}", "No keywords extracted"



def fetch_fashion_images(keywords):
    """Search for fashion-related images using Unsplash API."""
    
    formatted_keywords = keywords.replace(", ", "+")  # Format for URL
    url = f"https://api.unsplash.com/search/photos?query={formatted_keywords}&per_page=5"

    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}  # Use headers for security

    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        results = response.json().get("results", [])
        if results:
            image_urls = [img["urls"]["regular"] for img in results]  # Extract image URLs
            return image_urls
        else:
            return ["No images found."]
    else:
        return [f"Error fetching images: {response.json()}"]


"""
def get_gpt_recommendations(face_data, user_measurements):
    ###Generate clothing and hairstyle recommendations using GPT
    prompt = f"Face details: {face_data}. User measurements: {user_measurements}. Recommend clothing and hairstyle."  
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']
"""