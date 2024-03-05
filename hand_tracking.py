import cv2
import mediapipe as mp
import time
import pyttsx3

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

engine = pyttsx3.init()
engine.setProperty('rate', 150)

pTime = 0
cTime = 0 
hand_open = False
hand_close = False
hand_moving = False  # Flag to track hand movement status
notification_flag = False  # Flag to track if the notification has been triggered

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            finger_states = [0, 0, 0, 0, 0]  # Initialize finger states (0 for closed, 1 for open)
            
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                if id == 0:
                    cv2.circle(img, (cx, cy), 7, (255, 8, 8), cv2.FILLED)
                
                if id in [4, 8, 12, 16, 20]:  # Landmark indices for fingers
                    if cy < handLms.landmark[id - 2].y * h:  # Compare with the previous landmark
                        finger_states[(id - 4) // 4] = 1  # Finger is open
                elif id == 5:  # Landmark index for thumb
                    if cx > handLms.landmark[id - 1].x * w:  # Compare with the previous landmark
                        finger_states[4] = 1  # Thumb is open
            
            # Count open fingers and thumb
            open_finger_count = sum(finger_states[:4])
            open_thumb_count = finger_states[4]
            cv2.putText(img, f'Open Fingers: {open_finger_count}, Open Thumb: {open_thumb_count}', (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            
            # Check if all fingers or thumb are open or closed
            if open_finger_count == 4:
                hand_open = True
                hand_close = False
            elif open_finger_count == 0:
                hand_open = False
                hand_close = True
            else:
                hand_open = False
                hand_close = False
            
            # Display hand state
            if hand_open:
                cv2.circle(img, (50, 50), 30, (0, 255, 0), cv2.FILLED)
                cv2.putText(img, 'Hand Open', (70, 55), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                # Perform action when hand is open
                print("All fingers are open.")
                hand_moving = True
            else :
                cv2.circle(img, (50, 50), 30, (0, 0, 255), cv2.FILLED)
                cv2.putText(img, 'Hand Closed', (70, 55), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                # Perform action when hand is closed
                print("All fingers are closed.")
                hand_moving = True
            
           
            
                
    
    # Check if the hand is moving or not
    if hand_moving:
        if not notification_flag:
            engine.say("Hand is reacting")
            engine.runAndWait()
            notification_flag = True
    else:
        if notification_flag:
            engine.say("Hand stopped moving")
            engine.runAndWait()
            notification_flag = False
    
    hand_moving = False  # Reset hand moving flag
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 8, 255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
