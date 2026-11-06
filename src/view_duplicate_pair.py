
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

images = [
    (
        "100% label",
        ROOT / "datasets/dataset_1/Saline_bottle_original_size/sal_data_100/IMG20200208150929_01.jpg",
    ),
    (
        "Empty label",
        ROOT / "datasets/dataset_2/Saline_bottle_original_size/sal_data_empty/IMG20200208162558.jpg",
    ),
]

loaded = []

for label, path in images:
    image = cv2.imread(str(path))
    if image is None:
        print("Could not read:", path)
        continue

    image = cv2.resize(image, (500, 500))
    cv2.putText(
        image, label, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
    )
    loaded.append(image)

if len(loaded) == 2:
    cv2.imshow("Compare labelled images", np.hstack(loaded))
    print("Press any key on the image window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
