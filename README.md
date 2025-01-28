# Dress-to-Impress: AI-Powered Styling App

### To create the virtual environment
`python3 -m venv ./.venv`

### To activate the virtual environment and install dependancies
`source ./.venv/bin/activate`
`pip install -r requirements.txt`

## **1️⃣ Capture & Analyze User Features**
- Take a photo and extract key attributes using **AI & Computer Vision**:
  - **Gender Presentation:** Masculine / Feminine
  - **Hair Attributes:** Type, Color, Texture
  - **Skin Tone & Undertone**
  - **Facial Structure:** Jawline, cheekbones, face shape (Round, Oval, Square, etc.)
  - **Facial Hair Detection**
  - **Body Type Analysis:** Ectomorph, Mesomorph, Endomorph, or Fashion-Based Categories
  - **Neck & Shoulder Structure:** Broad/Narrow, Long/Short

## **2️⃣ User Preferences & Lifestyle Inputs**
- Ask users for their **style preferences**:
  - Corporate
  - Old-fashioned
  - Open to discovery
  - Edgy
- Gather **lifestyle data**:
  - Casual, Streetwear, High Fashion
  - Work attire (Corporate, Startup, Creative)
  - Special Events (Wedding, Interview, Date Night)
  - Cultural or Religious Preferences

## **3️⃣ Physical Measurements Input**
- Height
- Waist
- Chest
- Torso
- Shoulder Width

## **4️⃣ AI-Powered Style Recommendations**
### **👕 Clothing Suggestions**
- Best clothing styles for **body type & proportions**
- **Color matching** based on **skin tone & contrast level**
- **Layering advice** for optimal silhouette
- **Accessory & Footwear Pairing** (Watches, Glasses, Shoes, Jewelry)

### **💇‍♂️ Hairstyle Predictions**
- Best haircuts for **face shape & hair type**
- Facial hair styling for **jawline enhancement**

## **5️⃣ Image & Data Integration**
- Allow **multiple photos** (different angles & lighting)
- **Live Camera Mode** for Augmented Reality (AR) previews
- Integration with **Pinterest API** for a style/mood board
- Sync with shopping platforms (ASOS, Amazon, Nordstrom, etc.)

## **6️⃣ AI Personalization & Learning**
- **Adaptive Learning:** AI refines suggestions based on user choices
- **Explore Mode:** Safe + Experimental styles for discovery
- **Voting System:** "Do you like this look?" (Yes/No) to improve recommendations

## **7️⃣ Community & Social Features**
- **Community Voting:** Users share looks for feedback
- **Style Challenges:** "Try a new look this week" challenges
- **Celebrity & Influencer Style Matching**

## **8️⃣ Tech Stack Considerations**
- **Computer Vision API:** Google Vision, AWS Rekognition, OpenCV
- **ML Models:** GPT for text, TensorFlow/PyTorch for vision
- **Pinterest API** for inspiration
- **E-commerce API** for shopping integrations

## **💡 Bonus: "Virtual Stylist" Mode**
- Interactive AI chatbot explains style choices & lets users tweak recommendations.
