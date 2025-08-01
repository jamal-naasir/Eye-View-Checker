# Enhanced Eye Test with Redness, Sleep Prediction, and Report Export

import cv2
import mediapipe as mp
import os
import numpy as np
import random
import string
from collections import deque
from datetime import datetime
import time  # At top of file

CHART_DIR = "assets/charts"
CHART_LINES = [f"line_{i}.png" for i in range(1, 11)]  # Now shows 10 lines
VISION_SCALE = {
    1: "10%/100%",
    2: "20%/100%",
    3: "30%/100%",
    4: "40%/100%",
    5: "50%/100%",
    6: "60%/100%",
    7: "70%/100%",
    8: "80%/100%",
    9: "90%/100%",
    10: "100%/100%",
}

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

blink_history = []
blink_state = False
last_blink_time = 0
time_started = time.time()


EYE_LANDMARKS_LEFT = [33, 160, 158, 133, 153, 144]
EYE_LANDMARKS_RIGHT = [362, 385, 387, 263, 373, 380]
DARK_CIRCLE_POINTS = [145, 159]
SWELLING_POINTS = [(33, 159), (362, 386)]

report_data = {}

duration_minutes = max((time.time() - time_started) / 60, 1)
blink_count = sum(blink_history)
blinks_per_minute = blink_count / duration_minutes


def calculate_ear(eye_points):
    vertical1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    vertical2 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    horizontal = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def is_looking_center(landmarks):
    left_eye = landmarks.landmark[33]
    right_eye = landmarks.landmark[263]
    center_x = (left_eye.x + right_eye.x) / 2
    return 0.45 < center_x < 0.55

def detect_blink_with_ear(landmarks, image_w, image_h):
    global blink_state, last_blink_time
    left_eye = [(landmarks.landmark[p].x * image_w, landmarks.landmark[p].y * image_h) for p in EYE_LANDMARKS_LEFT]
    right_eye = [(landmarks.landmark[p].x * image_w, landmarks.landmark[p].y * image_h) for p in EYE_LANDMARKS_RIGHT]
    ear_left = calculate_ear(left_eye)
    ear_right = calculate_ear(right_eye)
    avg_ear = (ear_left + ear_right) / 2.0

    current_time = time.time()
    blink_detected = False

    if avg_ear < 0.21:
        if not blink_state and (current_time - last_blink_time) > 0.5:
            blink_state = True
            last_blink_time = current_time
            blink_detected = True
    else:
        blink_state = False

    return blink_detected
def detect_eye_color(eye_roi):
    if eye_roi.size == 0:
        return "Unknown"
    avg_color = cv2.mean(eye_roi)[:3]
    b, g, r = avg_color
    if r > 80 and g > 80 and b < 70:
        return "Hazel"
    elif b > r and b > g:
        return "Blue"
    elif r > g and r > b:
        return "Brown"
    else:
        return "Dark"

def estimate_dark_circles(frame, landmarks, w, h):
    try:
        # Define points for under-eye region
        left_eye_points = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_points = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Get coordinates for left eye region
        left_eye_region = np.array([(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) 
                                  for i in left_eye_points], np.int32)
        
        # Get coordinates for right eye region
        right_eye_region = np.array([(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) 
                                   for i in right_eye_points], np.int32)
        
        # Create masks for left and right eye regions
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [left_eye_region], 255)
        cv2.fillPoly(mask, [right_eye_region], 255)
        
        # Apply mask to get only the under-eye regions
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Convert to grayscale and apply thresholding
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return "None"
            
        # Get the largest contour area
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Calculate darkness ratio
        total_pixels = cv2.countNonZero(mask)
        if total_pixels == 0:
            return "Unknown"
            
        darkness_ratio = area / total_pixels
        
        # Determine darkness level based on ratio
        if darkness_ratio > 0.6:
            return "Severe"
        elif darkness_ratio > 0.4:
            return "Mild"
        elif darkness_ratio > 0.1:
            return "Slight"
        else:
            return "None"
            
    except Exception as e:
        print(f"Error in dark circle detection: {str(e)}")
        return "Unknown"
def estimate_swelling(frame, landmarks, w, h):
    try:
        top = landmarks.landmark[SWELLING_POINTS[0][1]]
        bottom = landmarks.landmark[SWELLING_POINTS[0][0]]
        v_distance = abs((top.y - bottom.y) * h)
        return "Yes" if v_distance > 30 else "No"
    except:
        return "Unknown"

def detect_redness(eye_roi):
    if eye_roi.size == 0:
        return "Unknown"
    hsv = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (160, 70, 50), (180, 255, 255))
    red_pixels = cv2.countNonZero(red_mask1 | red_mask2)
    total_pixels = eye_roi.shape[0] * eye_roi.shape[1]
    ratio = red_pixels / total_pixels
    return "High" if ratio > 0.1 else ("Low" if ratio > 0.03 else "None")

def predict_sleepiness(blinks_per_minute):
    return "Likely Sleepy" if blinks_per_minute > 12 else "Normal"
def estimate_tiredness(blink_rate, dark_circles, redness, swelling):
    tired_score = 0
    if blink_rate < 4 or blink_rate > 15:  # changed from 25 to 20
        tired_score += 1
    if dark_circles in ["Severe", "Mild"]:
        tired_score += 1
    if redness in ["High", "Low"]:
        tired_score += 1
    if swelling == "Yes":
        tired_score += 1
    return "Yes" if tired_score >= 2 else "No"

def export_report():
    os.makedirs("reports", exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"reports/eye_test_report_{now}.txt"
    with open(filename, "w") as f:
        for key, value in report_data.items():
            f.write(f"{key}: {value}\n")
    print(f"\n📄 Report saved to {filename}")
def generate_random_word(length):
    return ''.join(random.choices(string.ascii_uppercase, k=length))

def generate_chart_image(word, index):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    font_scale = 4.5 - (index * 0.5)
    font_scale = max(1.5, font_scale)
    text_size = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 5)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = (img.shape[0] + text_size[1]) // 2
    cv2.putText(img, word, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 5)
    return img


def run_test():
    cap = cv2.VideoCapture(0)
    score = 0
    user_text = ""
    left_eye_color = ""
    tiredness = "No"
    dark_circle_status = ""
    swelling_status = ""
    redness_status = ""
    sleep_status = ""
    time_started = time.time()
    CHART_WORDS = [generate_random_word(i + 1) for i in range(10)]
    CHART_IMAGES = [generate_chart_image(word, idx) for idx, word in enumerate(CHART_WORDS)]
    for idx in range(len(CHART_LINES)):
        chart_img = CHART_IMAGES[idx]
        expected_text = CHART_WORDS[idx]

        scale_factor = 0.6 - (idx * 0.05)
        scale_factor = max(0.1, scale_factor)
        width = int(chart_img.shape[1] * scale_factor)
        height = int(chart_img.shape[0] * scale_factor)
        chart_img = cv2.resize(chart_img, (width, height))
        chart_img = cv2.copyMakeBorder(chart_img,
            top=max(0, (480 - height) // 2),
            bottom=max(0, (480 - height + 1) // 2),
            left=max(0, (640 - width) // 2),
            right=max(0, (640 - width + 1) // 2),
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0])

        looking_center = False
        collecting_input = False
        user_text = ""

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            h, w, _ = frame.shape
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw alignment guides (crosshair)
                    h, w = frame.shape[:2]
                    center_x, center_y = w // 2, h // 2
                    
                    # Draw crosshair lines (thinner and semi-transparent)
                    color = (0, 255, 0)  # Green color for alignment guides
                    thickness = 1
                    length = 20  # Length of the crosshair lines
                    
                    # Horizontal line
                    cv2.line(frame, (center_x - length, center_y), (center_x + length, center_y), color, thickness)
                    # Vertical line
                    cv2.line(frame, (center_x, center_y - length), (center_x, center_y + length), color, thickness)
                    
                    # Draw a small circle at the center
                    cv2.circle(frame, (center_x, center_y), 3, color, -1)
                    
                    # Draw face mesh
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
                        connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
                    )
                    
                    # Draw a small circle at the nose tip (landmark 1) to help with alignment
                    nose_tip = face_landmarks.landmark[1]
                    nose_x, nose_y = int(nose_tip.x * w), int(nose_tip.y * h)
                    cv2.circle(frame, (nose_x, nose_y), 5, (0, 0, 255), -1)  # Red circle at nose tip
                    
                    looking_center = is_looking_center(face_landmarks)

                    if detect_blink_with_ear(face_landmarks, w, h):
                        blink_history.append(1)

                    if idx == 0:
                        x1 = int(face_landmarks.landmark[33].x * w)
                        x2 = int(face_landmarks.landmark[133].x * w)
                        y1 = int(face_landmarks.landmark[159].y * h) - 5
                        y2 = int(face_landmarks.landmark[145].y * h) + 5
                        eye_roi = frame[y1:y2, x1:x2]
                        left_eye_color = detect_eye_color(eye_roi)
                        dark_circle_status = estimate_dark_circles(frame, face_landmarks, w, h)
                        swelling_status = estimate_swelling(frame, face_landmarks, w, h)
                        redness_status = detect_redness(eye_roi)

            combined = cv2.hconcat([frame, chart_img])
            msg = f"Type your answer: {user_text}_" if collecting_input else "Press 'S' to start typing"
            blink_count = sum(blink_history)
            gaze_direction = "Center" if looking_center else "Away"
            duration_minutes = max((time.time() - time_started) / 60, 1)
            blink_rate = blink_count / duration_minutes
            tiredness = estimate_tiredness(blink_rate, dark_circle_status, redness_status, swelling_status)
            sleep_status = predict_sleepiness(blink_rate)

            if not looking_center:
                cv2.putText(combined, "Please look at the camera", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif collecting_input:
                # Draw a semi-transparent background for better visibility
                text_size = cv2.getTextSize(f"Type: {user_text}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                cv2.rectangle(combined, (40, 80), (60 + text_size[0], 130), (0, 0, 0), -1)
                cv2.rectangle(combined, (40, 80), (60 + text_size[0], 130), (255, 255, 255), 2)
                
                # Display the text
                cv2.putText(combined, f"Type: {user_text}", (50, 110), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Add a blinking cursor
                if int(time.time() * 2) % 2 == 0:  # Blink every 0.5 seconds
                    cursor_x = 50 + cv2.getTextSize(f"Type: {user_text}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]
                    cv2.line(combined, (cursor_x, 90), (cursor_x, 130), (255, 255, 255), 2)
            else:
                cv2.putText(combined, "Position detected - Ready to type", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(combined, f"Blink Count: {blink_count}", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            cv2.putText(combined, f"Eye Color: {left_eye_color}", (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(combined, f"Tired? {tiredness}", (400, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,100,255), 2)
            cv2.putText(combined, f"Dark Circles: {dark_circle_status}", (400, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150,50,250), 2)
            cv2.putText(combined, f"Swelling: {swelling_status}", (400, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,100,100), 2)
            cv2.putText(combined, f"Redness: {redness_status}", (400, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,50,200), 2)
            cv2.putText(combined, f"Sleep State: {sleep_status}", (400, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,100), 2)

            cv2.imshow("AI Eye Test - Type in Window", combined)
            key = cv2.waitKey(50) & 0xFF

            if key == 27:  # ESC key
                cap.release()
                cv2.destroyAllWindows()
                return

            # Automatically start collecting input when face is centered
            if looking_center and not collecting_input and not user_text:
                collecting_input = True
                # Small delay to allow user to see the instruction
                time.sleep(0.5)
                user_text = ""

            elif collecting_input:
                if key == 13:
                    break
                elif key == 8:
                    user_text = user_text[:-1]
                elif 32 <= key <= 126:
                    user_text += chr(key).upper()

        if not user_text.strip():
            print("❌ No answer provided. Ending test.")
            break

        if user_text.replace(" ", "").upper() == expected_text:
            score += 1
            print(f"✅ Line {idx+1} correct: {user_text}")
        else:
            print(f"❌ Incorrect input for line {idx+1}: {user_text} (Expected: {expected_text})")
            break

    cap.release()
    cv2.destroyAllWindows()
    report_data.update({
        "Read Line": str(score),
        "Estimated Vision": VISION_SCALE.get(score, '20/200+'),
        "Blink Count": str(sum(blink_history)),
        "Eye Color": left_eye_color,
        "Tiredness": tiredness,
        "Dark Circles": dark_circle_status,
        "Swelling": swelling_status,
        "Redness": redness_status,
        "Sleep Prediction": sleep_status
    })
    for k, v in report_data.items():
        print(f"{k}: {v}")
    export_report()

if __name__ == "__main__":
    run_test()
