import os
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering

# === Capture image from webcam ===
cam = cv2.VideoCapture(0)
cv2.namedWindow("📷 Press SPACE to capture, ESC to exit")

while True:
    ret, frame = cam.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    cv2.imshow("Press SPACE to capture", frame)
    k = cv2.waitKey(1)

    if k % 256 == 27:  # ESC to quit
        print("❌ Capture cancelled.")
        break
    elif k % 256 == 32:  # SPACE to capture
        img_name = os.path.join(os.getcwd(), "captured.jpg")
        saved = cv2.imwrite(img_name, frame)
        if saved:
            print(f"✅ Image saved as {img_name}")
        else:
            print("❌ Failed to save image.")
        break

cam.release()
cv2.destroyAllWindows()

# === Ask for question ===
prompt = input("📝 Enter your question (e.g., 'What happened to my arm?'): ").strip()
if not prompt.lower().startswith("question:"):
    prompt = f"Question: {prompt} Answer:"

# === Load BLIP-2 model and processor ===
if os.path.exists("blip2_model.pt") and os.path.exists("./blip2_processor"):
    print("🔁 Loading model from local cache...")
    model = torch.load("blip2_model.pt")
    processor = AutoProcessor.from_pretrained("./blip2_processor")
else:
    print("⬇️ Downloading model from HuggingFace...")
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-opt-2.7b")
    model.eval()
    torch.save(model, "blip2_model.pt")
    processor.save_pretrained("./blip2_processor")

# === Run VQA ===
image = Image.open("captured.jpg")
inputs = processor(images=image, text=prompt, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=50)

answer = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
print("\n🧠 Model answer:", answer)
