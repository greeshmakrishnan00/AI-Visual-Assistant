from ultralytics import YOLO
import cv2
import pyttsx3
import time
import os

# ==========================================
# AI VISUAL ASSISTANT PRO
# Fast Version + Voice + Screenshot Saving
# Exit with Q or q
# ==========================================

# Create folder for screenshots
os.makedirs("Detected_Objects", exist_ok=True)

# Load YOLO Model
model = YOLO("yolov8n.pt")

# ------------------------------------------
# TEXT TO SPEECH SETUP
# ------------------------------------------
engine = pyttsx3.init('sapi5')

voices = engine.getProperty('voices')

if len(voices) > 1:
    engine.setProperty('voice', voices[1].id)  # Female Voice
else:
    engine.setProperty('voice', voices[0].id)

engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# ------------------------------------------
# OBJECT DESCRIPTIONS
# ------------------------------------------
object_descriptions = {
    "person": "I can see a person in front of me.",
    "laptop": "I can see a laptop used for work and learning.",
    "cell phone": "I can see a mobile phone.",
    "bottle": "I can see a bottle used to store liquids.",
    "book": "I can see a book for reading and studying.",
    "chair": "I can see a chair used for sitting.",
    "keyboard": "I can see a computer keyboard.",
    "mouse": "I can see a computer mouse.",
    "tv": "I can see a television screen.",
    "cup": "I can see a drinking cup.",
    "car": "I can see a car.",
    "dog": "I can see a dog.",
    "cat": "I can see a cat.",
    "clock": "I can see a clock showing time.",
    "remote": "I can see a television remote control.",
    "backpack": "I can see a backpack.",
    "handbag": "I can see a handbag.",
    "potted plant": "I can see a potted plant."
}

# ------------------------------------------
# WEBCAM
# ------------------------------------------
cap = cv2.VideoCapture(0)

last_object = ""
display_text = ""

saved_objects = set()
spoken_objects = set()

frame_count = 0

print("=" * 60)
print("AI VISUAL ASSISTANT PRO STARTED")
print("Press Q or q to Exit")
print("=" * 60)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Faster processing
    frame = cv2.resize(frame, (416, 416))

    frame_count += 1

    # Process every 5th frame
    if frame_count % 5 != 0:

        cv2.imshow("AI Visual Assistant Pro", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in [ord('q'), ord('Q')]:
            print("Exiting Application...")
            break

        continue

    # YOLO Detection
    results = model(
        frame,
        imgsz=416,
        conf=0.30,
        verbose=False
    )

    detected_object = None

    for result in results:

        for box in result.boxes:

            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            label = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detected_object = label

            # Draw Bounding Box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw Label
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # ------------------------------------------
            # SAVE CROPPED OBJECT IMAGE ONCE
            # ------------------------------------------
            if label not in saved_objects:

                cropped = frame[y1:y2, x1:x2]

                if cropped.size > 0:

                    timestamp = time.strftime("%Y%m%d_%H%M%S")

                    filename = (
                        f"Detected_Objects/"
                        f"{label}_{timestamp}.jpg"
                    )

                    cv2.imwrite(filename, cropped)

                    print(f"Screenshot Saved: {filename}")

                    saved_objects.add(label)

            break

    # ------------------------------------------
    # SPEAK ONLY ON FIRST DETECTION
    # ------------------------------------------
    if detected_object:

        if detected_object not in spoken_objects:

            if detected_object in object_descriptions:

                display_text = (
                    f"Attention. I detected "
                    f"{detected_object}. "
                    f"{object_descriptions[detected_object]}"
                )

            else:

                display_text = (
                    f"Attention. I detected "
                    f"{detected_object}."
                )

            print("\nDetected:", detected_object)
            print("Assistant:", display_text)

            try:

                engine.stop()
                engine.say(display_text)
                engine.runAndWait()

            except Exception as e:

                print("Voice Error:", e)

            spoken_objects.add(detected_object)

        # Display output
        cv2.putText(
            frame,
            display_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

    else:
        display_text = ""

    cv2.imshow("AI Visual Assistant Pro", frame)

    key = cv2.waitKey(1) & 0xFF

    if key in [ord('q'), ord('Q')]:
        print("Exiting Application...")
        break

cap.release()
cv2.destroyAllWindows()

print("\nAI Visual Assistant Closed Successfully")