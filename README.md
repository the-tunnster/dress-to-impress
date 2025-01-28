# Dress-to-Impress 👗🤵 | AI-Powered Fashion Stylist
### Personalized Styling and Outfit Recommendations Using AI

## 📌 Overview
**Dress-to-Impress** is an AI-powered fashion recommendation system that provides **personalized styling advice** based on a user's **facial features, hair color, face shape, and body measurements**. Using **OpenAI’s GPT-4**, the app generates **custom outfit, hairstyle, and color palette suggestions** while integrating **Unsplash API** to provide **fashion inspiration images** tailored to the user’s profile.

- **AI-Powered Analysis** – Extracts key fashion traits from user images
- **Personalized Recommendations** – Suggests the best hairstyles, outfits, and colors
- **Visual Inspiration** – Fetches relevant fashion images for styling ideas
- **Seamless UX** – Built with **Streamlit** for an intuitive and interactive experience

---

## 🚀 Features
✅ **AI-Powered Styling Recommendations** – Uses **GPT-4** to generate hairstyle, clothing, and color palette suggestions  
✅ **Facial & Body Analysis** – Detects **face shape, hair color, and key features** to determine ideal fashion choices  
✅ **Keyword Extraction** – Identifies **fashion-related terms** to refine recommendations  
✅ **Image Search Integration** – Fetches **fashion inspiration images** using the **Unsplash API**  
✅ **Interactive User Interface** – Runs on a **Streamlit app**, allowing users to upload images and receive real-time analysis  

---

## 🛠️ Tech Stack
- **Frontend:** Streamlit (Python-based UI framework), React (Upcoming UI/UX update)  
- **AI Model:** OpenAI GPT-4 (for styling recommendations)  
- **Computer Vision:** DeepFace & MediaPipe (for face analysis)  
- **Image Processing:** OpenCV & PIL  
- **Image Search API:** Unsplash API (for fashion inspiration images), Pinterest API (Upcoming for better suggestions)  
- **Backend:** Python (Currently being updated to Flask for React integration)  

---

## 📦 Installation & Setup
### 🔹 1. Clone the Repository
```sh
git clone https://github.com/your-username/dress-to-impress.git
cd dress-to-impress
```

### 🔹 2. Create a Virtual Environment & Install Dependencies
```sh
python -m venv .venv
source .venv/bin/activate  # For macOS/Linux
.venv\Scripts\activate  # For Windows
pip install -r requirements.txt
```

### 🔹 3. Set Up Environment Variables
Create a `.env` file in the root directory and add:
```
OPENAI_API_KEY=your-openai-api-key-here
UNSPLASH_ACCESS_KEY=your-unsplash-api-key-here
```

### 🔹 4. Run the Application
```sh
streamlit run app.py
```

---

## 🛠️ How It Works
1. **User uploads an image** through the Streamlit interface.
2. **AI analyzes facial features, hair color, and body proportions**.
3. **OpenAI GPT-4 generates personalized styling recommendations**.
4. **Relevant fashion keywords are extracted** for further image retrieval.
5. **Unsplash API & Pinterest API fetch visual fashion inspiration** based on AI-extracted keywords.
6. **User receives a full styling guide** with text recommendations and curated images.

---

## 🛠️ Upcoming Features
- **Building a Privacy Policy** to ensure data protection and compliance.  
- **Integrating Pinterest API** for more relevant and personalized fashion inspiration.  
- **Creating a React front-end** for a smoother and more modern UI/UX experience.  
- **Updating the Python backend to use Flask** for better scalability and integration with the React frontend.  
- **Developing User Analytics** to track trends and improve recommendations based on user behavior.  

---

## 🤝 Contributing
We welcome contributions! To get started:
1. Fork the repository
2. Create a new branch (`feature-branch`)
3. Commit your changes
4. Push to your fork and submit a PR

