import numpy as np
import cv2


def find_contours(img):
    # finds contours of shapes, only returns the external contours of the shapes
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    return contours


def refine_mask(self, image, mask, mask_id):
    refine_length = self.refine_iterations_slider.value()
    refine_iterations = self.refine_iterations_slider.value()
    refine_mode = self.refine_mode.currentIndex()

    cell_mask = np.zeros(mask.shape, dtype=np.uint8)
    cell_mask[mask == mask_id] = 1

    wall_cnt = find_contours(cell_mask)[0]

    (ymax, xmax) = image.shape

    for i in range(refine_iterations):
        wall_contour = []
        img = image.copy()

        for j in range(len(wall_cnt)):
            x = wall_cnt[j, 0, 0]
            y = wall_cnt[j, 0, 1]

            y1, y2, x1, x2 = y - refine_length, y + refine_length, x - refine_length, x + refine_length

            if x1 < 0:
                x1 = 0
            if x2 > xmax:
                x2 = xmax
            if y1 < 0:
                y1 = 0
            if y2 > ymax:
                y2 = ymax

            box = [y1, y2, x1, x2]
            box_values = img[box[0]:box[1], box[2]:box[3]]

            if refine_mode == 0:
                box_index = np.array(np.unravel_index(box_values.argmax(), box_values.shape))
            else:
                box_index = np.array(np.unravel_index(box_values.argmin(), box_values.shape))

            wall_contour.append([box[2] + box_index[1], box[0] + box_index[0]])

        # creates cv contour, draws it
        wall_contour = np.array(wall_contour).reshape((-1, 1, 2)).astype(np.int32)
        wall_cnt = wall_contour.copy()

    # creates cv contour, draws it
    wall_contour = np.array(wall_contour).reshape((-1, 1, 2)).astype(np.int32)
    wall_mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.drawContours(wall_mask, [wall_contour], contourIdx=-1, color=(1, 1, 1), thickness=-1)

    return wall_mask
