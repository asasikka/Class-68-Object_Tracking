import cv2

# 0 Represents Default Web Cam
# video=cv2.VideoCapture(0)
video=cv2.VideoCapture("bb2.mp4")
while True:
    check,image= video.read()
    print("What is check:", check) #boolean
    image=cv2.resize(image, (900, 500))
    cv2.imshow("Aarav's Camera", image)
    if cv2.waitKey(1) == 32:
        break

video.release()