import os
import cv2
import csv
import re
import torch
import streamlit as st
from PIL import Image
from datetime import datetime
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering, AutoTokenizer, AutoModelForCausalLM
import google.generativeai as gen_ai
from dotenv import load_dotenv
import random
import time

# === Load environment variables ===
load_dotenv()
GOOGLE_API_KEY = "AIzaSyDMmcatKThhwSLVFlpXOO9YI7Wc3Rd5HX4"
gen_ai.configure(api_key=GOOGLE_API_KEY)

# interaction_log = "interactions_log.csv"

@st.cache_resource
def load_blip():
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-opt-2.7b")
    model.eval()
    return processor, model

@st.cache_resource
def load_biogpt():
    tokenizer = AutoTokenizer.from_pretrained("./biogpt_tokenizer")
    model = torch.load("./biogpt_model.pt", map_location="cpu")
    model.eval()
    return tokenizer, model

@st.cache_resource
def get_gemini_chat():
    model = gen_ai.GenerativeModel("models/gemini-1.5-pro")
    return model.start_chat(history=[])
    
def clean_output(text):
    refined = st.session_state.gemini_chat.send_message(
        f"Please refine the following biomedical output for readability and remove any tags or non-text characters:{text}. Dont remove any content"
    ).text
    return refined.strip()

def analyze_image(img_path, processor, model):
    image = Image.open(img_path)
    prompt = "Question: What injury is shown? Describe it briefly. Answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=50)
    full_response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    # Extract only the answer part if prefixed
    answer_match = re.search(r"Answer: (.*)", full_response, re.IGNORECASE)
    return answer_match.group(1).strip() if answer_match else full_response.strip()

def query_model(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
    return clean_output(tokenizer.decode(output[0], skip_special_tokens=True))

# def log_interaction(input_type, prompt, blip_response, gemini_response, biogpt_response, preferred_model, rating):
#     file_exists = os.path.exists(interaction_log)
#     with open(interaction_log, mode='a', newline='') as file:
#         writer = csv.writer(file)
#         if not file_exists:
#             writer.writerow(["Timestamp", "Input Type", "Prompt", "BLIP Response", "Gemini Response", "BioGPT Response", "Preferred Model", "Rating"])
#         writer.writerow([datetime.now(), input_type, prompt, blip_response, gemini_response, biogpt_response, preferred_model, rating])

def functionality(prompt, input_type="Text", blip_response=""):
    if "result" not in st.session_state or st.session_state.get("current_prompt") != prompt:
        biogpt_tok, biogpt_model = load_biogpt()
        gemini_response = st.session_state.gemini_chat.send_message(f'{prompt}. give a short description in 100 words').text
        biogpt_response = query_model(prompt, biogpt_tok, biogpt_model)
        outputs = [
            ("Response A", gemini_response, "Gemini"),
            ("Response B", biogpt_response, "BioGPT")
        ]
        random.shuffle(outputs)
        st.session_state.result = outputs
        st.session_state.gemini_response = gemini_response
        st.session_state.biogpt_response = biogpt_response
        st.session_state.current_prompt = prompt

    show_confidence = random.choice([True, False])

    cols = st.columns(2)
    st.session_state.label_mapping = {}
    for idx, (label, response, true_model) in enumerate(st.session_state.result):
        with cols[idx]:
            st.markdown(f"### {label}")
            st.write(response)

            # Get model confidence from Gemini for readability
            if show_confidence: 
                confidence_msg = st.session_state.gemini_chat.send_message(
                    f"Based on this response, how confident is the model in its medical accuracy (on a scale from 1 to 5)? in just 2 words: {response}"
                ).text
                st.caption(f"🔍 Confidence Estimate: {confidence_msg}")
                time.sleep(60)
            st.session_state.label_mapping[label] = true_model

    # st.selectbox("Which response did you prefer?", ["Response A", "Response B"], key="preferred_response")
    # st.slider("Rate the selected response", 1, 5, 3, key="rating_value")
    # st.feedback_text = st.text_area("💬 What made you prefer this response?", placeholder="Explain your reasoning or share any concerns...")

    if st.button("✅ Done"):
        
        for k in ["result", "gemini_response", "biogpt_response", "preferred_response", "rating_value", "current_prompt"]:
            st.session_state.pop(k, None)

st.title("AI Medical Assistant")

st.markdown("""
### 💡 Prompt Template Suggestions
| Goal         | Prompt Format                                |
|--------------|-----------------------------------------------|
| Definitions  | “XYZ is a condition characterized by…”        |
| Lists        | “Symptoms of XYZ include…”                    |
| Treatments   | “Treatment options for XYZ include…”          |
| Risks        | “Risk factors for XYZ include…”              |
| Causes       | “XYZ may be caused by…”                       |
""")

image_uploaded = st.radio("Do you want to upload a medical image?", ["Yes", "No"])

if "gemini_chat" not in st.session_state:
    st.session_state.gemini_chat = get_gemini_chat()

if image_uploaded == "Yes":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image")
        processor, blip_model = load_blip()
        img_path = os.path.join("temp_uploaded.jpg")
        with open(img_path, "wb") as f:
            f.write(uploaded_file.read())
        blip_response = analyze_image(img_path, processor, blip_model)
        st.success(f"BLIP-2: {blip_response}")

        user_followup_prompt = st.text_input("What would you like to know based on this finding?")
        if user_followup_prompt:
            functionality(user_followup_prompt, input_type="Image", blip_response=blip_response)
else:
    general_prompt = st.text_input("📝 Describe your medical concern or question:")
    if general_prompt:
        functionality(general_prompt, input_type="Text")
