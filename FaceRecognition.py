import streamlit as st
import os
import cv2
import numpy as np
import pickle
from deepface import DeepFace
from ultralytics import YOLO
import datetime

# Define the function to find the best match using cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_best_match(face_representation, preloaded_faces, target_person, threshold=0.45):
    if target_person in preloaded_faces:
        representations = preloaded_faces[target_person]
        for representation in representations:
            similarity = cosine_similarity(np.array(representation), np.array(face_representation))
            if similarity >= threshold:
                return target_person, similarity
    return "Unknown", 0

# Environment variables for paths
model_path = os.getenv('MODEL_PATH', r'/app/model/best.pt')
representations_file = os.getenv('REPRESENTATIONS_FILE', r'/app/data/final_updated.pkl')
capture_folder = os.getenv('CAPTURE_FOLDER', r'/app/Captured_Picture')
result_folder = os.getenv('RESULT_FOLDER', r'/app/Result')

# Set up the YOLO model and load preloaded face embeddings
model = YOLO(model_path)
confidence_threshold = 0.60

with open(representations_file, 'rb') as f:
    preloaded_faces = pickle.load(f)

# Create directories if they don't exist
os.makedirs(capture_folder, exist_ok=True)
os.makedirs(result_folder, exist_ok=True)

# Define RTSP channels
rtsp_channels = [
    {"channel_number": 13, "channel_name": "AI/FO team", "url": "rtsp://ai:ai*12345@172.16.12.31:554/cam/realmonitor?channel=13&subtype=0"},
    {"channel_number": 10, "channel_name": "CRM", "url": "rtsp://ai:ai*12345@172.16.12.31:554/cam/realmonitor?channel=10&subtype=0"},
    {"channel_number": 17, "channel_name": "BA Team", "url": "rtsp://ai:ai*12345@172.16.12.31:554/cam/realmonitor?channel=17&subtype=0"},
    {"channel_number": 23, "channel_name": "BC technical", "url": "rtsp://ai:ai*12345@172.16.12.31:554/cam/realmonitor?channel=23&subtype=0"}
]

# Streamlit UI
st.title("Face Recognition System")
target_person = st.text_input("Enter the name of the person to recognize:")

if st.button("Start Recognition"):
    if target_person:
        captured_images = []
        for channel in rtsp_channels:
            st.write(f"Capturing image from channel: {channel['channel_name']} (Channel {channel['channel_number']})")
            cap = cv2.VideoCapture(channel["url"])
            
            ret, frame = cap.read()
            if not ret:
                st.warning(f"Failed to grab frame from {channel['channel_name']}.")
                cap.release()
                continue

            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            channel_capture_folder = os.path.join(capture_folder, f'channel_{channel["channel_number"]}_{channel["channel_name"]}')
            os.makedirs(channel_capture_folder, exist_ok=True)
            
            image_name = f'capture_{timestamp}.jpg'
            image_path = os.path.join(channel_capture_folder, image_name)
            cv2.imwrite(image_path, frame)
            captured_images.append((channel, image_path, frame))
            cap.release()

        recognized = False
        for channel, image_path, frame in captured_images:
            st.write(f"Performing face recognition on {channel['channel_name']} (Channel {channel['channel_number']})")
            small_frame = cv2.resize(frame, (1080, 1080))
            results = model.predict(small_frame)
            height_ratio = frame.shape[0] / small_frame.shape[0]
            width_ratio = frame.shape[1] / small_frame.shape[1]

            for result in results[0].boxes:
                confidence = result.conf.item()
                if confidence >= confidence_threshold:
                    x1, y1, x2, y2 = map(int, result.xyxy[0])
                    x1, y1, x2, y2 = int(x1 * width_ratio), int(y1 * height_ratio), int(x2 * width_ratio), int(y2 * height_ratio)
                    face = frame[y1:y2, x1:x2]
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    temp_face_path = 'temp_face.jpg'
                    cv2.imwrite(temp_face_path, face_rgb)
                    try:
                        face_representation = DeepFace.represent(img_path=temp_face_path, model_name='VGG-Face', enforce_detection=False)[0]["embedding"]
                        best_match, similarity = find_best_match(face_representation, preloaded_faces, target_person)
                        label_color = (0, 255, 0) if best_match == target_person else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), label_color, 2)
                        cv2.putText(frame, f"{best_match} ({similarity:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

                        if best_match == target_person:
                            recognized = True
                    except Exception as e:
                        st.error(f"Error processing face for {channel['channel_name']}: {e}")

            result_channel_folder = os.path.join(result_folder, f'channel_{channel["channel_number"]}_{channel["channel_name"]}')
            os.makedirs(result_channel_folder, exist_ok=True)
            
            result_image_name = f'result_{timestamp}.jpg'
            result_path = os.path.join(result_channel_folder, result_image_name)
            cv2.imwrite(result_path, frame)
            st.write(f"Result saved in {result_path}")

            # Display the result if the target person was recognized
            if recognized:
                st.image(result_path, caption=f"Recognition Result for {target_person}", use_column_width=True)
                break  # Stop after showing the first recognized image
        if not recognized:
            st.warning(f"{target_person} was not recognized in any channel.")
    else:
        st.warning("Please enter a valid name.")
