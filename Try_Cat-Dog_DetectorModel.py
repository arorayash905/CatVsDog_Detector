import numpy as np
import cv2
from tensorflow.keras.models import load_model
model = load_model("Cat_Dog_Detector.model")

image = cv2.imread("cat1.jpg")
image0 = image
print(image)

image = cv2.resize(image, (150, 150))
image = np.reshape(image, [1, 150, 150, 3])
classes = model.predict(image)

print(classes[0])
if classes[0] > 0:
    image0 = cv2.putText(image0, "This is a Dog.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Image", image0)
    print("This is a dog.")

else:
    image0 = cv2.putText(image0, "This is a Cat.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Image", image0)
    print("This is a cat.")

cv2.waitKey(0)
cv2.destroyAllWindows()
