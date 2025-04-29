import cv2

cap = cv2.VideoCapture("C:/Users/ntaru/Downloads/WhatsApp Video 2025-04-15 at 6.05.26 PM.mp4")

if(not cap.isOpened()):
    print("error in opening file")
    exit()

while cap.isOpened():
    ret,frame = cap.read()

    if not ret:
        print("error ino pening")
        break

    cv2.imshow("frame",frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    
