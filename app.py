import streamlit as st
from PIL import Image
import google.generativeai as genai
from api_key import api_key
import io

# Configure the Gemini API
genai.configure(api_key=api_key)

# Load Gemini model (keeping your original line)
model = genai.GenerativeModel('gemini-2.5-pro')

# Streamlit UI
st.set_page_config(page_title="🧠 AI Medical Image Diagnostic", layout="centered")
st.title("🧠 Ahmer Nanna Panna  Medical Image Diagnostic")
st.markdown("Upload a medical image (X-ray, MRI, CT, etc.) and get a detailed diagnostic report using Gemini AI.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload a medical image", type=["jpg", "jpeg", "png"])

# Optional patient info
patient_info = st.text_area("🧾 Patient Information (optional)", "")

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Build the prompt
    prompt = f"""
You are an advanced AI medical imaging specialist with expertise equivalent to a senior radiologist, trained to evaluate a wide range of diagnostic scans including X-rays, MRIs, CT scans, and medical photographs.  
Your task is to carefully analyze the uploaded medical image(s) and return a *comprehensive diagnostic report* that includes the following sections:

---

*1. Preliminary Overview:*  
- Describe what the image likely represents (e.g., body part, scan type).  
- Mention any quality issues (blurring, obstruction, etc.)

---

*2. Detailed Clinical Observations:*  
- Identify all visible abnormalities, anomalies, or patterns.  
- Use standard medical terms and explain when needed.  
- Point out location-specific findings and any urgent signs.

---

*3. Differential Diagnosis (with reasoning):*  
- List most likely diagnoses with justifications.  
- Mention if normal findings are present but need monitoring.

---

*4. Recommended Next Steps:*  
- Suggest additional scans, blood tests, biopsies, or referrals.  
- Mention what should be correlated with patient history.

---

*5. General Treatment Recommendations:*  
- Provide typical treatments for the likely condition.  
- Include a disclaimer that this is not medical advice.

---

*6. Risk Factors & Warnings:*  
- Highlight risk factors inferred from the image.  
- Mention any critical signs requiring immediate action.

---

*7. Notes for Human Physician Review:*  
- Mention ambiguities or areas needing expert interpretation.  
- State clearly if image quality is insufficient for confident analysis.

---

Patient Information: {patient_info}
"""

    # Analyze button
    if st.button("🩺 Analyze Image"):
        with st.spinner("Analyzing with Gemini..."):
            try:
                # Convert image to JPEG bytes to ensure compatibility
                img = Image.open(uploaded_file).convert("RGB")
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")
                image_bytes = buffer.getvalue()

                # Gemini expects image input this way
                gemini_image = {
                    "mime_type": "image/jpeg",
                    "data": image_bytes
                }

                # Generate content
                response = model.generate_content([prompt, gemini_image])
                st.subheader("📋 Diagnostic Report")
                st.markdown(response.text)

            except Exception as e:
                st.error(f"An error occurred while processing the image: {e}")