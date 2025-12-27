"""
Qo'l bilan yurakcha shakli va yuz ifodasini aniqlash dasturi.
OpenCV va MediaPipe kutubxonalaridan foydalanadi.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time

# MediaPipe qo'l va yuz aniqlash
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Statistika
smile_count = 0
last_smile_time = 0

def draw_heart(img, center_x, center_y, size=50, color=(0, 0, 255)):
    """Yurak shakli chizish"""
    points = []
    for t in range(0, 360, 5):
        rad = math.radians(t)
        x = 16 * (math.sin(rad) ** 3)
        y = 13 * math.cos(rad) - 5 * math.cos(2*rad) - 2 * math.cos(3*rad) - math.cos(4*rad)
        px = int(center_x + x * size / 16)
        py = int(center_y - y * size / 16)
        points.append((px, py))
    
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], color)
    cv2.polylines(img, [pts], True, (255, 100, 100), 2)

def draw_emoji(img, emoji_type, x, y, size=60):
    """Emoji chizish"""
    # Yuz doirasi
    if emoji_type == "happy":
        face_color = (0, 255, 255)  # Sariq
    elif emoji_type == "sad":
        face_color = (255, 200, 100)  # Och ko'k
    elif emoji_type == "surprised":
        face_color = (0, 200, 255)  # To'q sariq
    else:
        face_color = (200, 200, 200)  # Kulrang
    
    cv2.circle(img, (x, y), size, face_color, -1)
    cv2.circle(img, (x, y), size, (0, 0, 0), 2)
    
    # Ko'zlar
    eye_y = y - size // 4
    left_eye_x = x - size // 3
    right_eye_x = x + size // 3
    
    if emoji_type == "happy":
        # Qisilgan ko'zlar (yoy shakli)
        cv2.ellipse(img, (left_eye_x, eye_y), (size//6, size//10), 0, 200, 340, (0, 0, 0), 2)
        cv2.ellipse(img, (right_eye_x, eye_y), (size//6, size//10), 0, 200, 340, (0, 0, 0), 2)
    elif emoji_type == "surprised":
        # Katta ko'zlar
        cv2.circle(img, (left_eye_x, eye_y), size//5, (255, 255, 255), -1)
        cv2.circle(img, (left_eye_x, eye_y), size//8, (0, 0, 0), -1)
        cv2.circle(img, (right_eye_x, eye_y), size//5, (255, 255, 255), -1)
        cv2.circle(img, (right_eye_x, eye_y), size//8, (0, 0, 0), -1)
    else:
        # Oddiy ko'zlar
        cv2.circle(img, (left_eye_x, eye_y), size//6, (0, 0, 0), -1)
        cv2.circle(img, (right_eye_x, eye_y), size//6, (0, 0, 0), -1)
    
    # Og'iz
    mouth_y = y + size // 3
    
    if emoji_type == "happy":
        # Kulgi - yoy shakli
        cv2.ellipse(img, (x, mouth_y - 5), (size//2, size//4), 0, 10, 170, (0, 0, 0), 3)
    elif emoji_type == "sad":
        # Xafa - teskari yoy
        cv2.ellipse(img, (x, mouth_y + 10), (size//3, size//5), 0, 190, 350, (0, 0, 0), 3)
    elif emoji_type == "surprised":
        # Hayron - doira og'iz
        cv2.circle(img, (x, mouth_y), size//4, (0, 0, 0), -1)
    else:
        # Neytral - to'g'ri chiziq
        cv2.line(img, (x - size//3, mouth_y), (x + size//3, mouth_y), (0, 0, 0), 2)

def calculate_distance(p1, p2):
    """Ikki nuqta orasidagi masofani hisoblash"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_landmark_pos(landmarks, idx, w, h):
    """Landmark pozitsiyasini olish"""
    lm = landmarks.landmark[idx]
    return (int(lm.x * w), int(lm.y * h))

def detect_emotion(face_landmarks, img_width, img_height):
    """
    Yuz ifodasini aniqlash
    Returns: emotion_type, mouth_ratio, eye_ratio
    """
    if face_landmarks is None:
        return "neutral", 0, 0
    
    # Og'iz nuqtalari (yuqori va pastki lab)
    # Yuqori lab: 13, Pastki lab: 14, Chap burchak: 61, O'ng burchak: 291
    upper_lip = get_landmark_pos(face_landmarks, 13, img_width, img_height)
    lower_lip = get_landmark_pos(face_landmarks, 14, img_width, img_height)
    left_mouth = get_landmark_pos(face_landmarks, 61, img_width, img_height)
    right_mouth = get_landmark_pos(face_landmarks, 291, img_width, img_height)
    
    # Og'iz ochiqlik darajasi
    mouth_height = calculate_distance(upper_lip, lower_lip)
    mouth_width = calculate_distance(left_mouth, right_mouth)
    mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
    
    # Ko'z nuqtalari
    # Chap ko'z: yuqori 159, pastki 145
    # O'ng ko'z: yuqori 386, pastki 374
    left_eye_top = get_landmark_pos(face_landmarks, 159, img_width, img_height)
    left_eye_bottom = get_landmark_pos(face_landmarks, 145, img_width, img_height)
    right_eye_top = get_landmark_pos(face_landmarks, 386, img_width, img_height)
    right_eye_bottom = get_landmark_pos(face_landmarks, 374, img_width, img_height)
    
    # Ko'z ochiqlik darajasi
    left_eye_height = calculate_distance(left_eye_top, left_eye_bottom)
    right_eye_height = calculate_distance(right_eye_top, right_eye_bottom)
    eye_ratio = (left_eye_height + right_eye_height) / 2
    
    # Lab burchaklari (kulayotganini aniqlash - burchaklar yuqoriga ko'tarilgan)
    left_corner = get_landmark_pos(face_landmarks, 61, img_width, img_height)
    right_corner = get_landmark_pos(face_landmarks, 291, img_width, img_height)
    mouth_center = get_landmark_pos(face_landmarks, 13, img_width, img_height)
    
    # Burchaklar markazdan yuqorida bo'lsa - kulgu
    corner_up = (left_corner[1] < mouth_center[1] - 5) and (right_corner[1] < mouth_center[1] - 5)
    
    # Kayfiyatni aniqlash
    if mouth_ratio > 0.4:  # Og'iz katta ochiq
        if eye_ratio > 12:  # Ko'zlar katta
            return "surprised", mouth_ratio, eye_ratio
        else:
            return "happy", mouth_ratio, eye_ratio
    elif mouth_ratio > 0.15 and (eye_ratio < 8 or corner_up):  # Og'iz sal ochiq, ko'z qisilgan
        return "happy", mouth_ratio, eye_ratio
    elif mouth_ratio < 0.1 and eye_ratio < 7:  # Og'iz yopiq, ko'z qisilgan
        return "sad", mouth_ratio, eye_ratio
    else:
        return "neutral", mouth_ratio, eye_ratio

def is_heart_gesture(landmarks_left, landmarks_right, img_width, img_height):
    """Yurakcha shakli tekshirish"""
    if landmarks_left is None or landmarks_right is None:
        return False, None
    
    left_thumb = landmarks_left.landmark[4]
    right_thumb = landmarks_right.landmark[4]
    left_index = landmarks_left.landmark[8]
    right_index = landmarks_right.landmark[8]
    
    left_thumb_pos = (int(left_thumb.x * img_width), int(left_thumb.y * img_height))
    right_thumb_pos = (int(right_thumb.x * img_width), int(right_thumb.y * img_height))
    left_index_pos = (int(left_index.x * img_width), int(left_index.y * img_height))
    right_index_pos = (int(right_index.x * img_width), int(right_index.y * img_height))
    
    thumb_distance = calculate_distance(left_thumb_pos, right_thumb_pos)
    index_distance = calculate_distance(left_index_pos, right_index_pos)
    
    threshold = 80
    
    if thumb_distance < threshold and index_distance < threshold:
        center_x = (left_thumb_pos[0] + right_thumb_pos[0] + left_index_pos[0] + right_index_pos[0]) // 4
        center_y = (left_thumb_pos[1] + right_thumb_pos[1] + left_index_pos[1] + right_index_pos[1]) // 4 - 30
        return True, (center_x, center_y)
    
    return False, None

def apply_mood_overlay(frame, emotion):
    """Kayfiyatga qarab ekran rangini o'zgartirish"""
    overlay = frame.copy()
    
    if emotion == "happy":
        # Yashil rang - xursand
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 100, 0), -1)
        alpha = 0.1
    elif emotion == "sad":
        # Ko'k rang - xafa
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (100, 0, 0), -1)
        alpha = 0.15
    elif emotion == "surprised":
        # Sariq rang - hayron
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 100, 100), -1)
        alpha = 0.1
    else:
        return frame
    
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

def main():
    global smile_count, last_smile_time
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Kamera ochilmadi!")
        return
    
    print("=" * 60)
    print("   YUZ IFODASI VA YURAKCHA GESTURE ANIQLASH DASTURI")
    print("=" * 60)
    print("\n Ko'rsatmalar:")
    print("  1. Yurakcha: Ikkala qo'l bilan yurakcha shakli yasang ❤️")
    print("  2. Kulgu:    Tabassum qiling 😊")
    print("  3. Xafa:     Xafa bo'ling 😢")
    print("  4. Hayron:   Og'zingizni katta oching 😮")
    print("\n  Chiqish uchun 'q' tugmasini bosing")
    print("=" * 60)
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands, mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        heart_detected = False
        heart_frames = 0
        heart_center = (0, 0)
        current_emotion = "neutral"
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Qo'llarni aniqlash
            hand_results = hands.process(rgb_frame)
            
            # Yuzni aniqlash
            face_results = face_mesh.process(rgb_frame)
            
            landmarks_left = None
            landmarks_right = None
            
            # Qo'l landmarklari
            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                    label = handedness.classification[0].label
                    
                    if label == "Left":
                        landmarks_right = hand_landmarks
                    else:
                        landmarks_left = hand_landmarks
                    
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )
            
            # Yuz ifodasini aniqlash
            face_landmarks = None
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                
                # Yuz konturini chizish (ixtiyoriy, yengil)
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(100, 200, 100), thickness=1)
                )
            
            # Kayfiyatni aniqlash
            emotion, mouth_ratio, eye_ratio = detect_emotion(face_landmarks, w, h)
            current_emotion = emotion
            
            # Kulgu sonini hisoblash
            if emotion == "happy":
                current_time = time.time()
                if current_time - last_smile_time > 2:  # Har 2 sekundda bir marta sanash
                    smile_count += 1
                    last_smile_time = current_time
            
            # Kayfiyatga qarab rang qo'shish
            frame = apply_mood_overlay(frame, emotion)
            
            # Yurakcha shakli tekshirish
            is_heart, center = is_heart_gesture(landmarks_left, landmarks_right, w, h)
            
            if is_heart:
                heart_detected = True
                heart_frames = 30
                heart_center = center
            
            # Yurak chizish
            if heart_detected and heart_frames > 0:
                draw_heart(frame, heart_center[0], heart_center[1], size=100, color=(0, 0, 255))
                
                text = "Jannatim"
                text2 = "onam"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = heart_center[0] - text_size[0] // 2
                text_y = heart_center[1] - 5
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
                
                text_size2 = cv2.getTextSize(text2, font, font_scale, thickness)[0]
                text_x2 = heart_center[0] - text_size2[0] // 2
                text_y2 = heart_center[1] + 20
                cv2.putText(frame, text2, (text_x2, text_y2), font, font_scale, (255, 255, 255), thickness)
                
                heart_frames -= 1
                if heart_frames == 0:
                    heart_detected = False
            
            # Emoji chizish
            draw_emoji(frame, emotion, w - 80, 80, 50)
            
            # Kayfiyat matni
            emotion_texts = {
                "happy": "Siz kulyapsiz! :)",
                "sad": "Siz nega xafasiz? :(",
                "surprised": "Voy! Hayron bo'ldingizmi? :O",
                "neutral": "Neytral holat"
            }
            
            emotion_colors = {
                "happy": (0, 255, 0),
                "sad": (255, 100, 100),
                "surprised": (0, 200, 255),
                "neutral": (200, 200, 200)
            }
            
            # Kayfiyat matnini ko'rsatish
            cv2.putText(frame, emotion_texts[emotion], (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, emotion_colors[emotion], 2)
            
            # Statistika
            cv2.putText(frame, f"Kulgu soni: {smile_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Qo'llar soni
            hands_count = len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0
            status_color = (0, 255, 0) if hands_count == 2 else (0, 165, 255)
            cv2.putText(frame, f"Qo'llar: {hands_count}/2", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
            
            # Chiqish ko'rsatmasi
            cv2.putText(frame, "Chiqish: 'q'", (10, h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            cv2.imshow('Yuz va Gesture Aniqlash', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 40)
    print(f"   Jami kulgu soni: {smile_count}")
    print("=" * 40)
    print("Dastur tugatildi!")

if __name__ == "__main__":
    main()
