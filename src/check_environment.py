
import sys
from pathlib import Path

import cv2
import numpy as np

print("IV Fluid Monitoring Project")
print("Python version:", sys.version.split()[0])
print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)

output = Path("results/test_bottle.png")
output.parent.mkdir(parents=True, exist_ok=True)

image = np.zeros((400, 600, 3), dtype=np.uint8)
cv2.rectangle(image, (220, 60), (380, 350), (255, 255, 255), 3)
cv2.rectangle(image, (223, 200), (377, 347), (255, 180, 50), -1)

cv2.putText(
    image,
    "IV TEST IMAGE",
    (175, 385),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2,
)

if cv2.imwrite(str(output), image):
    print("Test image saved to:", output)
    print("Environment check completed successfully!")
else:
    print("Could not save test image.")
