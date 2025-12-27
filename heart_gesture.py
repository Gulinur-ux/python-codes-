"""
Qo'l bilan yurakcha shakli yasalganda yurak chizadigan dastur.
OpenCV va MediaPipe kutubxonalaridan foydalanadi.
"""

import cv2
import mediapipe as mp
import numpy as np
import math

# MediaPipe qo'l aniqlash
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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
    
    # Yurakni to'ldirish
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], color)
    cv2.polylines(img, [pts], True, (255, 100, 100), 2)

def calculate_distance(p1, p2):
    """Ikki nuqta orasidagi masofani hisoblash"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_heart_gesture(landmarks_left, landmarks_right, img_width, img_height):
    """
    Yurakcha shakli tekshirish:
    - Ikkala qo'lning bosh barmoqlari va ko'rsatkich barmoqlari bir-biriga yaqin bo'lishi kerak
    """
    if landmarks_left is None or landmarks_right is None:
        return False, None
    
    # Bosh barmoq uchlari (THUMB_TIP = 4)
    left_thumb = landmarks_left.landmark[4]
    right_thumb = landmarks_right.landmark[4]
    
    # Ko'rsatkich barmoq uchlari (INDEX_FINGER_TIP = 8)
    left_index = landmarks_left.landmark[8]
    right_index = landmarks_right.landmark[8]
    
    # Piksel koordinatalariga o'tkazish
    left_thumb_pos = (int(left_thumb.x * img_width), int(left_thumb.y * img_height))
    right_thumb_pos = (int(right_thumb.x * img_width), int(right_thumb.y * img_height))
    left_index_pos = (int(left_index.x * img_width), int(left_index.y * img_height))
    right_index_pos = (int(right_index.x * img_width), int(right_index.y * img_height))
    
    # Bosh barmoqlar bir-biriga yaqinmi?
    thumb_distance = calculate_distance(left_thumb_pos, right_thumb_pos)
    
    # Ko'rsatkich barmoqlar bir-biriga yaqinmi?
    index_distance = calculate_distance(left_index_pos, right_index_pos)
    
    # Threshold - barmoqlar qanchalik yaqin bo'lishi kerak
    threshold = 80
    
    # Yurakcha shakli: barmoqlar yaqin va to'g'ri joylashgan
    if thumb_distance < threshold and index_distance < threshold:
        # Yurak markazi
        center_x = (left_thumb_pos[0] + right_thumb_pos[0] + left_index_pos[0] + right_index_pos[0]) // 4
        center_y = (left_thumb_pos[1] + right_thumb_pos[1] + left_index_pos[1] + right_index_pos[1]) // 4 - 30
        return True, (center_x, center_y)
    
    return False, None

def main():
    # Kamerani ochish
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Kamera ochilmadi!")
        return
    
    print("=" * 50)
    print("YURAKCHA GESTURENI ANIQLASH DASTURI")
    print("=" * 50)
    print("\nKo'rsatmalar:")
    print("1. Ikkala qo'lingizni kamera oldida ko'rsating")
    print("2. Bosh barmoq va ko'rsatkich barmoqlaringiz bilan")
    print("   yurakcha shakli yasang ❤️")
    print("3. Chiqish uchun 'q' tugmasini bosing")
    print("=" * 50)
    
    # MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:
        
        heart_detected = False
        heart_frames = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Kadr o'qilmadi")
                continue
            
            # Oynani aks ettirish (selfie ko'rinishi)
            frame = cv2.flip(frame, 1)
            
            # O'lchamlarni olish
            h, w, _ = frame.shape
            
            # RGB formatiga o'tkazish
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Qo'llarni aniqlash
            results = hands.process(rgb_frame)
            
            landmarks_left = None
            landmarks_right = None
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Qo'l turini aniqlash
                    label = handedness.classification[0].label
                    
                    # Oyna aks ettirilganligi uchun chap va o'ng almashadi
                    if label == "Left":
                        landmarks_right = hand_landmarks
                    else:
                        landmarks_left = hand_landmarks
                    
                    # Qo'l nuqtalarini chizish
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )
            
            # Yurakcha shakli tekshirish
            is_heart, center = is_heart_gesture(landmarks_left, landmarks_right, w, h)
            
            if is_heart:
                heart_detected = True
                heart_frames = 30  # Yurak 30 kadr davomida ko'rinadi
                heart_center = center
            
            # Yurak chizish
            if heart_detected and heart_frames > 0:
                draw_heart(frame, heart_center[0], heart_center[1], size=80, color=(0, 0, 255))
                
                # Matn yozish
                cv2.putText(frame, "YURAK! <3", (heart_center[0] - 60, heart_center[1] - 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                heart_frames -= 1
                if heart_frames == 0:
                    heart_detected = False
            
            # Status matnlari
            cv2.putText(frame, "Yurakcha shakli yasang!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Aniqlangan qo'llar soni
            hands_count = 0
            if results.multi_hand_landmarks:
                hands_count = len(results.multi_hand_landmarks)
            
            status_color = (0, 255, 0) if hands_count == 2 else (0, 165, 255)
            cv2.putText(frame, f"Qo'llar: {hands_count}/2", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            cv2.putText(frame, "Chiqish: 'q' tugmasi", (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Oynani ko'rsatish
            cv2.imshow('Yurakcha Gesture', frame)
            
            # 'q' tugmasi bosilsa chiqish
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nDastur tugatildi!")

if __name__ == "__main__":
    main()
