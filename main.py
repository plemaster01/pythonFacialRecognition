import cv2 #pip install opencv-python
# get user values
imagePath = 'beatles2.jpg'
cascadePath = 'haarcascade_frontalface_default.xml'
# create the haar cascade
faceCascade = cv2.CascadeClassifier(cascadePath)
# read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect the faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=5,
    minSize=(30, 30)
)
print(f'Found {len(faces)} faces!')
# draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Found found', image)
cv2.waitKey(0)