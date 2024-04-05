"""
Created on Wed Apr 27 10:32:45 2022

@author: turnerp
"""

import math
import os
import tempfile
import traceback

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import exposure
from napari.utils.notifications import show_info


class _stats_utils:
    @staticmethod
    def normalize99(X):
        """normalize image so 0.0 is 0.01st percentile and 1.0 is 99.99th percentile"""

        if np.max(X) > 0:
            X = X.copy()
            v_min, v_max = np.percentile(X[X != 0], (1, 99))
            X = exposure.rescale_intensity(X, in_range=(v_min, v_max))

        return X

    @staticmethod
    def find_contours(img):
        # finds contours of shapes, only returns the external contours of the shapes

        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        return contours

    @staticmethod
    def determine_overlap(cnt_num, contours, image):
        try:
            # gets current contour of interest
            cnt = contours[cnt_num]

            # number of pixels in contour
            cnt_pixels = len(cnt)

            # gets all other contours
            cnts = contours.copy()
            del cnts[cnt_num]

            # create mask of all contours, without contour of interest. Contours are filled
            cnts_mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.drawContours(cnts_mask, cnts, contourIdx=-1, color=(1, 1, 1), thickness=-1)

            # create mask of contour of interest. Only the contour outline is drawn.
            cnt_mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.drawContours(cnt_mask, [cnt], contourIdx=-1, color=(1, 1, 1), thickness=1)

            # dilate the contours mask. Neighbouring contours will now overlap.
            kernel = np.ones((3, 3), np.uint8)
            cnts_mask = cv2.dilate(cnts_mask, kernel, iterations=1)

            # get overlapping pixels
            overlap = cv2.bitwise_and(cnt_mask, cnts_mask)

            # count the number of overlapping pixels
            overlap_pixels = len(overlap[overlap == 1])

            # calculate the overlap percentage
            overlap_percentage = int((overlap_pixels / cnt_pixels) * 100)

        except:
            overlap_percentage = None

        return overlap_percentage

    @staticmethod
    def get_contour_statistics(cnt, image, pixel_size):
        # cell area
        try:
            area = cv2.contourArea(cnt) * pixel_size ** 2
        except:
            area = None

        # convex hull
        try:
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area
        except:
            solidity = None

        # perimiter
        try:
            perimeter = cv2.arcLength(cnt, True) * pixel_size
        except:
            perimeter = None

            # area/perimeter
        try:
            aOp = area / perimeter
        except:
            aOp = None

        # bounding rectangle
        try:
            x, y, w, h = cv2.boundingRect(cnt)
            rect_area = w * h
            # cell crop
            y1, y2, x1, x2 = y, (y + h), x, (x + w)
        except:
            y1, y2, x1, x2 = None, None, None, None

        # calculates moments, and centre of flake coordinates
        try:
            M = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cell_centre = [int(cx), int(cy)]
        except:
            cx = None
            cy = None
            cell_centre = [None, None]

        # cell length and width from PCA analysis
        try:
            cx, cy, lx, ly, wx, wy, data_pts = _stats_utils.pca(cnt)
            length, width, angle = _stats_utils.get_pca_points(image, cnt, cx, cy, lx, ly, wx, wy)
            radius = width / 2
            length = length * pixel_size
            width = width * pixel_size
            radius = radius * pixel_size

        except:
            length = None
            width = None
            radius = None

        # asepct ratio
        try:
            aspect_ratio = length / width
        except:
            aspect_ratio = None

        contour_statistics = dict(numpy_BBOX=[x1, x2, y1, y2], coco_BBOX=[x1, y1, h, w], pascal_BBOX=[x1, y1, x2,
                                                                                                      y2], cell_centre=cell_centre, cell_area=area, cell_length=length, cell_width=width, cell_radius=radius, aspect_ratio=aspect_ratio, circumference=perimeter, solidity=solidity, aOp=aOp, )

        return contour_statistics

    @staticmethod
    def angle_of_line(x1, y1, x2, y2):
        try:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        except Exception:
            angle = None

        return angle

    @staticmethod
    def euclidian_distance(x1, y1, x2, y2):
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

        return distance

    @staticmethod
    def pca(pts):
        # Construct a buffer used by the pca analysis
        sz = len(pts)
        data_pts = np.empty((sz, 2), dtype=np.float64)
        for i in range(data_pts.shape[0]):
            data_pts[i, 0] = pts[i, 0, 0]
            data_pts[i, 1] = pts[i, 0, 1]

        # #removes duplicate contour points
        arr, uniq_cnt = np.unique(data_pts, axis=0, return_counts=True)
        data_pts = arr[uniq_cnt == 1]

        # Perform PCA analysis
        mean = np.empty(0)
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

        # Store the center of the object
        cx, cy = (mean[0, 0], mean[0, 1])
        lx, ly = (cx + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cy + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0],)
        wx, wy = (cx - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cy - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0],)

        return cx, cy, lx, ly, wx, wy, data_pts

    @staticmethod
    def get_pca_points(img, cnt, cx, cy, lx, ly, wx, wy):
        if (lx - cx) == 0 or (wx - cx) == 0:
            pca_error = True
            length = 0
            width = 0
            pca_points = {"lx1": 0, "ly1": 0, "lx2": 0, "ly2": 0, "wx1": 0, "wy1": 0, "wx2": 0, "wy2": 0, }
        else:
            pca_error = False

            # get line slope and intercept
            length_slope = (ly - cy) / (lx - cx)
            length_intercept = cy - length_slope * cx
            width_slope = (wy - cy) / (wx - cx)
            width_intercept = cy - width_slope * cx

            lx1 = 0
            lx2 = max(img.shape)
            ly1 = length_slope * lx1 + length_intercept
            ly2 = length_slope * lx2 + length_intercept

            wx1 = 0
            wx2 = max(img.shape)
            wy1 = width_slope * wx1 + width_intercept
            wy2 = width_slope * wx2 + width_intercept

            contour_mask = np.zeros(img.shape, dtype=np.uint8)
            length_line_mask = np.zeros(img.shape, dtype=np.uint8)
            width_line_mask = np.zeros(img.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [cnt], contourIdx=-1, color=(255, 255, 255), thickness=-1, )
            cv2.line(length_line_mask, (int(lx1), int(ly1)), (int(lx2), int(ly2)), (255, 255, 255), 2, )
            cv2.line(width_line_mask, (int(wx1), int(wy1)), (int(wx2), int(wy2)), (255, 255, 255), 2, )

            Intersection = cv2.bitwise_and(contour_mask, length_line_mask)
            Intersection = np.array(np.where(Intersection.T == 255)).T
            [[lx1, ly1], [lx2, ly2]] = np.array([Intersection[0], Intersection[-1]])

            Intersection = cv2.bitwise_and(contour_mask, width_line_mask)
            Intersection = np.array(np.where(Intersection.T == 255)).T
            [[wx1, wy1], [wx2, wy2]] = np.array([Intersection[0], Intersection[-1]])

            pca_points = {"lx1": lx1, "ly1": ly1, "lx2": lx2, "ly2": ly2, "wx1": wx1, "wy1": wy1, "wx2": wx2, "wy2": wy2, }

            length = _stats_utils.euclidian_distance(lx1, ly1, lx2, ly2)
            width = _stats_utils.euclidian_distance(wx1, wy1, wx2, wy2)

            angle = _stats_utils.angle_of_line(lx1, ly1, lx2, ly2)

        return length, width, angle

    @staticmethod
    def rotate_contour(cnt, angle=90, units="DEGREES"):
        x = cnt[:, :, 1].copy()
        y = cnt[:, :, 0].copy()

        x_shift, y_shift = sum(x) / len(x), sum(y) / len(y)

        # Shift to origin (0,0)
        x = x - int(x_shift)
        y = y - int(y_shift)

        # Convert degrees to radians
        if units == "DEGREES":
            angle = math.radians(angle)

        # Rotation matrix multiplication to get rotated x & y
        xr = (x * math.cos(angle)) - (y * math.sin(angle)) + x_shift
        yr = (x * math.sin(angle)) + (y * math.cos(angle)) + y_shift

        cnt[:, :, 0] = yr
        cnt[:, :, 1] = xr

        shift_xy = [x_shift[0], y_shift[0]]

        return cnt, shift_xy

    @staticmethod
    def rotate_image(image, shift_xy, angle=90):
        x_shift, y_shift = shift_xy

        (h, w) = image.shape[:2]

        # Perform image rotation
        M = cv2.getRotationMatrix2D((y_shift, x_shift), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))

        return image, shift_xy

    @staticmethod
    def get_cell_images(self, image, mask, cell_mask, mask_id, layer_names, colicoords_dir):
        cell_image = image.copy()

        inverted_cell_mask = np.zeros(mask.shape, dtype=np.uint8)
        inverted_cell_mask[mask != 0] = 1
        inverted_cell_mask[mask == mask_id] = 0

        cnt = _stats_utils.find_contours(cell_mask)[0]

        x, y, w, h = cv2.boundingRect(cnt)

        if h > w:
            vertical = True
            cell_mask = np.zeros(mask.shape, dtype=np.uint8)
            cnt, shift_xy = _stats_utils.rotate_contour(cnt, angle=90)
            cell_image, shift_xy = _stats_utils.rotate_image(cell_image, shift_xy, angle=90)
            inverted_cell_mask, shift_xy = _stats_utils.rotate_image(inverted_cell_mask, shift_xy, angle=90)
            cv2.drawContours(cell_mask, [cnt], -1, 1, -1)
        else:
            vertical = False
            shift_xy = None

        x, y, w, h = cv2.boundingRect(cnt)
        y1, y2, x1, x2 = y, (y + h), x, (x + w)

        m = 5

        edge = False

        if y1 - 5 > 0:
            y1 = y1 - 5
        else:
            y1 = 0
            edge = True

        if y2 + 5 < cell_mask.shape[0]:
            y2 = y2 + 5
        else:
            y2 = cell_mask.shape[0]
            edge = True

        if x1 - 5 > 0:
            x1 = x1 - 5
        else:
            x1 = 0
            edge = True

        if x2 + 5 < cell_mask.shape[1]:
            x2 = x2 + 5
        else:
            x2 = cell_mask.shape[1]
            edge = True

        h, w = y2 - y1, x2 - x1

        inverted_cell_mask = inverted_cell_mask[y1:y2, x1:x2]
        cell_mask = cell_mask[y1:y2, x1:x2]
        cell_image = cell_image[:, y1:y2, x1:x2]

        for i in range(len(cell_image)):
            cell_img = cell_image[i].copy()
            cell_img[inverted_cell_mask == 1] = 0
            cell_img = _stats_utils.normalize99(cell_img)
            cell_image[i] = cell_img

        offset = [y1, x1]
        box = [y1, y2, x1, x2]

        cell_images = dict(cell_image=cell_image, cell_mask=cell_mask, channels=layer_names, offset=offset, shift_xy=shift_xy, box=box, edge=edge, vertical=vertical, mask_id=mask_id, contour=cnt, )

        if os.path.isdir(colicoords_dir) is False:
            os.mkdir(colicoords_dir)

        temp_path = tempfile.TemporaryFile(prefix="colicoords", suffix=".npy", dir=colicoords_dir).name

        np.save(temp_path, cell_images)

        return temp_path

    @staticmethod
    def get_layer_statistics(image, cell_mask, box, layer_names):
        layer_statistics = {}

        for i in range(len(image)):
            layer = layer_names[i]

            x1, x2, y1, y2 = box

            cell_image_crop = image[i][y1:y2, x1:x2].copy()
            cell_mask_crop = cell_mask[y1:y2, x1:x2].copy()

            try:
                cell_brightness = int(np.mean(cell_image_crop[cell_mask_crop != 0]))
                cell_background_brightness = int(np.mean(cell_image_crop[cell_mask_crop == 0]))
                cell_contrast = cell_brightness / cell_background_brightness
                cell_laplacian = int(cv2.Laplacian(cell_image_crop, cv2.CV_64F).var())
            except:
                cell_brightness = None
                cell_contrast = None
                cell_laplacian = None

            stats = {"cell_brightness[" + layer + "]": cell_brightness, "cell_contrast[" + layer + "]": cell_contrast, "cell_laplacian[" + layer + "]": cell_laplacian, }

            layer_statistics = {**layer_statistics, **stats}

        layer_statistics = {key: value for key, value in sorted(layer_statistics.items())}

        return layer_statistics

    def get_cell_statistics(self, mode, pixel_size, colicoords_dir, progress_callback=None):

        layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Nucleoid", "Classes", "center_lines", "Localisations"]]

        if mode == "active":
            dims = [self.viewer.dims.current_step[0]]
        else:
            dim_range = int(self.viewer.dims.range[0][1])
            dims = np.arange(0, dim_range)

        image_stack = []
        file_name_stack = []

        for i in dims:
            image = []
            file_names = []

            for layer in layer_names:
                image.append(self.viewer.layers[layer].data[i])
                file_names.append(self.viewer.layers[layer].metadata[i]["image_name"])

            image = np.stack(image, axis=0)
            image_stack.append(image)
            file_name_stack.append(file_names)

        image_stack = np.stack(image_stack, axis=0)

        mask_stack = self.segLayer.data.copy()
        meta_stack = self.segLayer.metadata.copy()
        label_stack = self.classLayer.data.copy()

        if mode == "active":
            current_step = self.viewer.dims.current_step[0]

            mask_stack = np.expand_dims(mask_stack[current_step], axis=0)
            label_stack = np.expand_dims(label_stack[current_step], axis=0)
            meta_stack = np.expand_dims(meta_stack[current_step], axis=0)

        cell_statistics = []

        cell_dict = {1: "Single", 2: "Dividing", 3: "Divided", 4: "Broken", 5: "Vertical"}

        for i in range(len(image_stack)):
            progress = int(((i + 1) / len(image_stack)) * 100)
            progress_callback.emit(progress)

            image = image_stack[i]
            mask = mask_stack[i]
            meta = meta_stack[i]
            label = label_stack[i]
            file_names = file_name_stack[i]

            image_stats = {}
            for j in range(len(image)):
                layer = layer_names[j]
                img = image[j]
                dat = {"image_brightness[" + layer + "]": int(np.mean(img)), "image_laplacian[" + layer + "]": int(cv2.Laplacian(img, cv2.CV_64F).var()), }
                image_stats = {**image_stats, **dat}

            contours = []
            contour_mask_ids = []
            cell_types = []
            mask_ids = np.unique(mask)

            for j in range(len(mask_ids)):
                mask_id = mask_ids[j]

                if mask_id != 0:
                    cell_mask = np.zeros(mask.shape, dtype=np.uint8)
                    cell_mask[mask == mask_id] = 1

                    cnt = _stats_utils.find_contours(cell_mask)[0]
                    contours.append(cnt)
                    contour_mask_ids.append(mask_id)

                    cell_label = np.unique(label[mask == mask_id])[0]

                    if cell_label in cell_dict.keys():
                        # print(cell_label)
                        cell_types.append(cell_dict[cell_label])

                        try:
                            background = np.zeros(mask.shape, dtype=np.uint8)
                            cv2.drawContours(background, contours, contourIdx=-1, color=(1, 1, 1), thickness=-1, )
                        except:
                            background = None

            for j in range(len(contours)):
                try:
                    cnt = contours[j]
                    mask_id = contour_mask_ids[j]
                    cell_type = cell_types[j]

                    cell_mask = np.zeros(mask.shape, dtype=np.uint8)
                    cv2.drawContours(cell_mask, [cnt], contourIdx=-1, color=(1, 1, 1), thickness=-1, )

                    overlap_percentage = _stats_utils.determine_overlap(j, contours, mask)

                    contour_statistics = _stats_utils.get_contour_statistics(cnt, mask, pixel_size)

                    box = contour_statistics["numpy_BBOX"]

                    cell_images_path = _stats_utils.get_cell_images(self, image, mask, cell_mask, mask_id, layer_names, colicoords_dir, )

                    layer_stats = _stats_utils.get_layer_statistics(image, cell_mask, box, layer_names)

                    morphology_stats = dict(
                        file_names=file_names,
                        colicoords=False,
                        cell_type=cell_type,
                        pixel_size_um=pixel_size,
                        length=contour_statistics["cell_length"],
                        width=contour_statistics["cell_width"],
                        radius=(contour_statistics["cell_radius"]),
                        area=contour_statistics["cell_area"],
                        circumference=contour_statistics["circumference"],
                        aspect_ratio=contour_statistics["aspect_ratio"],
                        solidity=contour_statistics["solidity"],
                        overlap_percentage=overlap_percentage,
                        box=box,
                        cell_images_path=cell_images_path, )

                    stats = {**morphology_stats, **image_stats, **layer_stats}

                    cell_statistics.append(stats)

                except:
                    pass

        return cell_statistics

    def process_cell_statistics(self, cell_statistics, path):

        def _event(viewer, cell_statistics=None):

            try:

                if type(cell_statistics) == dict:
                    ldist_data = cell_statistics["ldist_data"]
                    ldist_data = pd.DataFrame.from_dict(ldist_data).dropna(how="all")
                    cell_statistics = cell_statistics["cell_statistics"]
                else:
                    ldist_data = None

                export_path = os.path.join(path, "statistics.xlsx")

                drop_columns = ["cell_image", "cell_mask", "offset", "shift_xy", "edge", "vertical", "mask_id", "contour", "edge", "vertical", "mask_id", "cell", "refined_cnt", "oufti", "statistics",
                    "colicoords_channel", "channels", "cell_images_path", "ldist", ]

                cell_statistics = pd.DataFrame(cell_statistics)

                cell_statistics = cell_statistics.drop(columns=[col for col in cell_statistics if col in drop_columns])

                cell_statistics = cell_statistics.dropna(how="all")

                with pd.ExcelWriter(export_path) as writer:
                    cell_statistics.to_excel(writer, sheet_name="Cell Statistics", index=False, startrow=1, startcol=1, )
                    if isinstance(ldist_data, pd.DataFrame):
                        ldist_data.to_excel(writer, sheet_name="Length Distribution Data", index=False, startrow=1, startcol=1, )
            except:
                print(traceback.format_exc())

            return

        return _event(self.viewer, cell_statistics)

    @staticmethod
    def check_edge_cell(cell_mask, mask, buffer=1):

        edge = False

        try:

            cell_mask_bbox = cv2.boundingRect(cell_mask)
            [x, y, w, h] = cell_mask_bbox
            [x1, y1, x2, y2] = [x, y, x + w, y + h]
            bx1, by1, bx2, by2 = [x1 - buffer, y1 - buffer, x2 + buffer, y2 + buffer]

            if bx1 < 0:
                edge = True
            if by1 < 0:
                edge = True
            if bx2 > mask.shape[1]:
                edge = True
            if by2 > mask.shape[0]:
                edge = True

        except:
            print(traceback.format_exc())

        return edge

    @staticmethod
    def compute_cell_contrast(image, cell_mask, global_background, buffer=10, erode=True, contrast_mode="mean"):

        contrast = None

        try:

            cell_mask_bbox = cv2.boundingRect(cell_mask)
            [x, y, w, h] = cell_mask_bbox
            [x1, y1, x2, y2] = [x, y, x + w, y + h]
            bx1, by1, bx2, by2 = [x1 - buffer, y1 - buffer, x2 + buffer, y2 + buffer]

            if bx1 < 0:
                bx1 = 0
            if by1 < 0:
                by1 = 0
            if bx2 > image.shape[1]:
                bx2 = image.shape[1]
            if by2 > image.shape[0]:
                by2 = image.shape[0]

            local_background = np.zeros_like(image)
            local_background[by1:by2, bx1:bx2] = 1

            background_mask = local_background - cell_mask
            background_mask[global_background == 1] = 0

            cell_mask_size = len(cell_mask[cell_mask == 1])
            background_mask_size = len(background_mask[background_mask == 1])

            if cell_mask_size > 0 and background_mask_size > 0:

                if contrast_mode == "mean":
                    cell_mean = np.mean(image[cell_mask == 1])
                    background_mean = np.mean(image[background_mask == 1])
                elif contrast_mode == "median":
                    cell_mean = np.median(image[cell_mask == 1])
                    background_mean = np.median(image[background_mask == 1])
                elif contrast_mode == "max":
                    cell_mean = np.max(image[cell_mask == 1])
                    background_mean = np.max(image[background_mask == 1])
                elif contrast_mode == "min":
                    cell_mean = np.min(image[cell_mask == 1])
                    background_mean = np.min(image[background_mask == 1])

                contrast = cell_mean - background_mean

        except:
            print(traceback.format_exc())
            contrast = None
            pass

        return contrast

    def _filter_cells(self, remove=True, fov_mode ="", metric ="", criteria ="", threshold = "", ignore_edge=True):

        try:

            n_unfiltered = 0
            n_filtered = 0
            n_removed = 0
            n_fov = 0

            if metric == "":
                metric = self.filter_metric.currentText()
            if criteria == "":
                criteria = self.filter_criteria.currentText()
            if threshold == "":
                threshold = self.filter_threshold.value()
            if fov_mode == "":
                fov_mode = self.filter_mode.currentText()

            if "active " in fov_mode.lower():
                fov_list = [self.viewer.dims.current_step[0]]
            elif "all " in fov_mode.lower():
                N_fov = int(self.viewer.dims.range[0][1])
                fov_list = range(N_fov)
            else:
                fov_list = []

            contrast_mode = "mean"
            if "contrast" in metric.lower():
                if "mean" in metric.lower():
                    contrast_mode = "mean"
                elif "median" in metric.lower():
                    contrast_mode = "median"
                elif "min" in metric.lower():
                    contrast_mode = "min"
                elif "max" in metric.lower():
                    contrast_mode = "max"


            image_layers = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Nucleoid",
                                                                                             "Classes", "center_lines"]]
            selected_layer = self.viewer.layers.selection.active

            for fov in fov_list:

                try:

                    n_fov += 1

                    mask = self.segLayer.data[fov].copy()
                    class_mask = self.classLayer.data[fov].copy()
                    background_mask = np.zeros_like(mask)
                    background_mask[mask == 0] = 0
                    background_mask[mask != 0] = 1

                    filtered_mask = np.zeros_like(mask)

                    if selected_layer.name in image_layers:
                        image = selected_layer.data[fov].copy()
                    else:
                        image = None

                    mask_ids = np.unique(mask)

                    for mask_id in mask_ids:
                        if mask_id != 0:

                            cell_mask = np.zeros(mask.shape, dtype=np.uint8)
                            cell_mask[mask == mask_id] = 1

                            edge = _stats_utils.check_edge_cell(cell_mask, mask)

                            cnt, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                            value = None
                            if "area" in metric.lower():
                                value = cv2.contourArea(cnt[0])
                            if "contrast" in metric.lower():
                                if image is not None:
                                    contast = _stats_utils.compute_cell_contrast(image, cell_mask,
                                        background_mask, contrast_mode=contrast_mode)
                                    if contast is not None:
                                        value = contast

                            if ignore_edge == True and edge == True:
                                filtered_mask[mask == mask_id] = mask_id
                            if value != None:
                                if criteria == ">":
                                    if value < threshold:
                                        filtered_mask[mask == mask_id] = mask_id
                                if criteria == "<":
                                    if value > threshold:
                                        filtered_mask[mask == mask_id] = mask_id

                    n_unfiltered += (len(np.unique(mask)) - 1)
                    n_filtered += (len(np.unique(filtered_mask)) - 1)

                    if remove == True:
                        self.segLayer.data[fov] = filtered_mask.copy()
                        self.segLayer.refresh()

                        class_mask[filtered_mask == 0] = 0
                        self.classLayer.data[fov] = class_mask.copy()
                        self.classLayer.refresh()

                except:
                    print(traceback.format_exc())
                    pass

            n_removed = n_unfiltered - n_filtered

            if remove == True:
                show_info(f"Removed {n_removed} cells from {n_fov} FOV(s)")
            else:
                print_metric = metric.replace("(Selected Channel)", "")
                show_info(f"Found {n_removed} cells in {n_fov} FOV(s) with {print_metric} {criteria} {threshold}")

        except:
            print(traceback.format_exc())
            pass

    def _compute_simple_cell_stats(self):
        if self.unfolded == True:
            self.fold_images()

        current_fov = self.viewer.dims.current_step[0]

        mask = self.segLayer.data[current_fov]

        mask_ids = np.unique(mask)

        cell_area = []
        cell_solidity = []
        cell_aspect_ratio = []
        cell_centre = []
        cell_zoom = []
        cell_id = []

        for mask_id in mask_ids:
            if mask_id != 0:
                cnt_mask = np.zeros(mask.shape, dtype=np.uint8)
                cnt_mask[mask == mask_id] = 255

                cnt, _ = cv2.findContours(cnt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                x, y, w, h = cv2.boundingRect(cnt[0])
                y1, y2, x1, x2 = y, (y + h), x, (x + w)

                try:
                    area = cv2.contourArea(cnt[0])
                    hull = cv2.convexHull(cnt[0])
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area
                    (_, _), (width, height), _ = cv2.minAreaRect(cnt[0])
                    aspect_ratio = max(width, height) / min(width, height)

                except:
                    area = 0
                    solidity = 0
                    aspect_ratio = 0

                centre = (0, y1 + (y2 - y1) // 2, x1 + (x2 - x1) // 2)

                zoom = (min((mask.shape[0] / (y2 - y1)), (mask.shape[1] / (x2 - x1))) / 2)

                cell_area.append(area)
                cell_solidity.append(solidity)
                cell_aspect_ratio.append(aspect_ratio)
                cell_centre.append(centre)
                cell_zoom.append(zoom)
                cell_id.append(mask_id)

        cell_stats = {"cell_area": cell_area, "cell_solidity": cell_solidity, "cell_aspect_ratio": cell_aspect_ratio, "cell_centre": cell_centre, "cell_zoom": cell_zoom, "mask_id": cell_id, }

        layer_names = [layer.name for layer in self.viewer.layers if layer.name]

        for layer in layer_names:
            meta = self.viewer.layers[layer].metadata[current_fov]
            meta["simple_cell_stats"] = cell_stats
            self.viewer.layers[layer].metadata[current_fov] = meta
