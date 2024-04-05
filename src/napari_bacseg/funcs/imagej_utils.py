import cv2
import numpy as np
import tifffile
from napari.utils.notifications import show_info
from roifile import ImagejRoi, roiread


class _imagej_utils:

    def read_imagej_file(self, path, image, widget_notifications=True):

        contours = []
        mask = np.zeros(image.shape[-2:], dtype=np.uint16)

        # reads overlays sequentially and then converts them to openCV contours
        try:
            for roi in roiread(path):
                coordinates = roi.integer_coordinates

                top = roi.top
                left = roi.left

                coordinates[:, 1] = coordinates[:, 1] + top
                coordinates[:, 0] = coordinates[:, 0] + left

                cnt = np.array(coordinates).reshape((-1, 1, 2)).astype(np.int32)
                contours.append(cnt)

            for i in range(len(contours)):
                cnt = contours[i]
                cv2.drawContours(mask, [cnt], contourIdx=-1, color=(i + 1, i + 1, i + 1), thickness=-1, )
        except:
            if widget_notifications:
                show_info("Image does not contain ImageJ Overlays")

        return mask


    def export_imagej(self, image, contours, metadata, file_path):

        overlays = []

        for i in range(len(contours) - 1):
            try:
                cnt = contours[i]
                cnt = np.vstack(cnt).squeeze()
                roi = ImagejRoi.frompoints(cnt)
                roi = roi.tobytes()

                overlays.append(roi)

            except:
                pass

        tifffile.imwrite(file_path, image, imagej=True, metadata={"Overlays": overlays, "metadata": metadata}, )
