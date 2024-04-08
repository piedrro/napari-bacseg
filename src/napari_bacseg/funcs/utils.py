import copy
import hashlib
import json
import os
import tempfile
import traceback
import warnings
import cv2
import mat4py
import numpy as np
import pandas as pd
import scipy
import tifffile

from astropy.io import fits
from glob2 import glob
from napari.utils.notifications import show_info
from skimage import exposure
from skimage.registration import phase_cross_correlation


class _utils:

    def normalize99(self, X):
        """normalize image so 0.0 is 0.01st percentile and 1.0 is 99.99th percentile"""
    
        if np.max(X) > 0:
            X = X.copy()
            v_min, v_max = np.percentile(X[X != 0], (0.1, 99.9))
            X = exposure.rescale_intensity(X, in_range=(v_min, v_max))
    
        return X

    def rescale01(self, x):
        """normalize image from 0 to 1"""
    
        if np.max(x) > 0:
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
    
        return x

    def import_imagej(self, progress_callback, paths):

        if isinstance(paths, list) == False:
            paths = [paths]
    
        if len(paths) == 1:
            paths = os.path.abspath(paths[0])
    
            if os.path.isfile(paths) == True:
                file_paths = [paths]
    
            else:
                file_paths = glob(paths + r"*\**\*.tif", recursive=True)
        else:
            file_paths = paths
    
        file_paths = [file for file in file_paths if file.split(".")[-1] == "tif"]
    
        images = []
        masks = []
        nmasks = []
        metadata = {}
        imported_images = {}
    
        img_index = 0
    
        for i in range(len(file_paths)):
            progress = int(((i + 1) / len(file_paths)) * 100)
            try:
                progress_callback.emit(progress)
            except:
                pass
    
            if self.widget_notifications:
                show_info("loading image " + str(i + 1) + " of " + str(len(file_paths)))
    
            paths = file_paths[i]
            paths = os.path.abspath(paths)
    
            import_precision = self.import_precision.currentText()
            multiframe_mode = self.import_multiframe_mode.currentIndex()
            crop_mode = self.import_crop_mode.currentIndex()
    
            image_list, meta = self.read_image_file(paths, import_precision, multiframe_mode, crop_mode)
    
            akseg_hash = self.get_hash(img_path=paths)

            file_name = os.path.basename(paths)

            for index, frame in enumerate(image_list):
                contrast_limit = np.percentile(frame, (1, 99))
                contrast_limit = [int(contrast_limit[0] * 0.5), int(contrast_limit[1] * 2), ]

                mask = self.read_imagej_file(paths, frame)

                self.active_import_mode = "imagej"

                if len(image_list) > 1:
                    frame_name = (file_name.replace(".", "_") + "_" + str(index) + ".tif")
                else:
                    frame_name = copy.deepcopy(file_name)

                frame_meta = copy.deepcopy(meta)

                frame_meta["akseg_hash"] = akseg_hash
                frame_meta["image_name"] = frame_name
                frame_meta["image_path"] = paths
                frame_meta["mask_name"] = frame_name
                frame_meta["mask_path"] = paths
                frame_meta["label_name"] = None
                frame_meta["label_path"] = None
                frame_meta["import_mode"] = "Dataset"
                frame_meta["contrast_limit"] = contrast_limit
                frame_meta["contrast_alpha"] = 0
                frame_meta["contrast_beta"] = 0
                frame_meta["contrast_gamma"] = 0
                frame_meta["dims"] = [frame.shape[-1], frame.shape[-2]]
                frame_meta["crop"] = [0, frame.shape[-2], 0, frame.shape[-1]]

                images.append(frame)
                masks.append(mask)
                metadata[img_index] = frame_meta

                if imported_images == {}:
                    imported_images["Image"] = dict(images=[frame], masks=[mask], nmasks=[], classes=[], metadata={img_index: frame_meta}, )
                else:
                    imported_images["Image"]["images"].append(frame)
                    imported_images["Image"]["masks"].append(mask)
                    imported_images["Image"]["metadata"][img_index] = frame_meta

                img_index += 1

        imported_data = dict(imported_images=imported_images)

        return imported_data

    def get_folder(self, files):

        folder = ""
        parent_folder = ""

        paths = files["path"].tolist()

        paths = [os.path.normpath(path) for path in paths]

        if len(paths) > 1:
            paths = np.array([path.split(os.sep) for path in paths]).T

            for i in range(len(paths)):
                if len(set(paths[i])) != 1:
                    folder = str(paths[i - 1][0])
                    parent_folder = str(paths[i - 2][0])

                    break

        else:
            folder = paths[0].split(os.sep)[-2]
            parent_folder = paths[0].split(os.sep)[-3]

        return folder, parent_folder

    def read_image_file(self, path, precision="native", multiframe_mode=0, crop_mode=0):

        path = os.path.abspath(path)
        path = os.path.normpath(path)

        image_name = os.path.basename(path)

        if os.path.splitext(image_name)[1] == ".fits":
            with fits.open(path, ignore_missing_simple=True) as hdul:
                image = hdul[0].data
                try:
                    metadata = dict(hdul[0].header)

                    unserializable_keys = [key for key, value in metadata.items() if type(value) not in [bool, int, float, str]]

                    for key in unserializable_keys:
                        metadata.pop(key)

                except:
                    metadata = {}
        else:
            with tifffile.TiffFile(path) as tif:
                try:
                    metadata = tif.pages[0].tags["ImageDescription"].value
                    metadata = json.loads(metadata)
                except:
                    metadata = {}

            image = tifffile.imread(path)

        image = self.crop_image(image, crop_mode)

        image = self.get_frame(image, multiframe_mode)

        image = self.rescale_image(image, precision=precision)

        folder = os.path.abspath(path).split(os.sep)[-2]
        parent_folder = os.path.abspath(path).split(os.sep)[-3]

        if "image_name" not in metadata.keys():
            metadata["image_name"] = image_name
            metadata["channel"] = None
            metadata["modality"] = None
            metadata["stain"] = None
            metadata["stain_target"] = None
            metadata["light_source"] = None
            metadata["segmentation_file"] = None
            metadata["segmentation_channel"] = None
            metadata["image_path"] = path
            metadata["mask_name"] = None
            metadata["mask_path"] = None
            metadata["label_name"] = None
            metadata["label_path"] = None
            metadata["crop_mode"] = crop_mode
            metadata["multiframe_mode"] = multiframe_mode
            metadata["folder"] = folder
            metadata["parent_folder"] = parent_folder
            metadata["dims"] = [image[0].shape[-1], image[0].shape[-2]]
            metadata["crop"] = [0, image[0].shape[-2], 0, image[0].shape[-1]]

        return image, metadata

    def get_frame(self, img, multiframe_mode):

        if len(img.shape) > 2:
            if multiframe_mode == 0:
                img = img[0, :, :]

            elif multiframe_mode == 1:
                img = np.max(img, axis=0)

            elif multiframe_mode == 2:
                img = np.mean(img, axis=0).astype(np.uint16)

            elif multiframe_mode == 3:
                img = np.sum(img, axis=0)

            elif multiframe_mode == 4:
                img = [im for im in img]

        if type(img) != list:
            img = [img]

        return img

    def crop_image(self, img, crop_mode=0):

        if crop_mode != 0:
            if len(img.shape) > 2:
                imgL = img[:, :, : img.shape[-1] // 2]
                imgR = img[:, :, img.shape[-1] // 2:]
            else:
                imgL = img[:, : img.shape[-1] // 2]
                imgR = img[:, img.shape[-1] // 2:]

            if crop_mode == 1:
                img = imgL
            if crop_mode == 2:
                img = imgR

            if crop_mode == 3:
                if np.mean(imgL) > np.mean(imgR):
                    img = imgL
                else:
                    img = imgR

        return img

    def rescale_image(self, image, precision="int16"):

        precision_dict = {"int8": np.uint8, "int16": np.uint16, "int32": np.uint32, "native": image[0].dtype, }

        dtype = precision_dict[precision]

        if "int" in str(dtype):
            max_value = np.iinfo(dtype).max - 1
        else:
            max_value = np.finfo(dtype).max - 1

        if precision != "native":
            for i, img in enumerate(image):
                img = ((img - np.min(img)) / np.max(img)) * max_value
                img = img.astype(dtype)
                image[i] = img

        return image

    def get_brightest_fov(self, image):

        imageL = image[0, :, : image.shape[2] // 2]
        imageR = image[0, :, image.shape[2] // 2:]

        if np.mean(imageL) > np.mean(imageR):
            image = image[:, :, : image.shape[2] // 2]
        else:
            image = image[:, :, : image.shape[2] // 2]

        return image

    def imadjust(self, img):

        v_min, v_max = np.percentile(img, (1, 99))
        img = exposure.rescale_intensity(img, in_range=(v_min, v_max))

        return img

    def get_channel(self, img, multiframe_mode):

        if len(img.shape) > 2:
            if multiframe_mode == 0:
                img = img[0, :, :]

            elif multiframe_mode == 1:
                img = np.max(img, axis=0)

            elif multiframe_mode == 2:
                img = np.mean(img, axis=0).astype(np.uint16)

        return img

    def get_fov(self, img, channel_mode):

        imgL = img[:, : img.shape[1] // 2]
        imgR = img[:, img.shape[1] // 2:]

        if channel_mode == 0:
            if np.mean(imgL) > np.mean(imgR):
                img = imgL
            else:
                img = imgR
        if channel_mode == 1:
            img = imgL
        if channel_mode == 2:
            img = imgR

        return img

    def process_image(self, image, multiframe_mode, channel_mode):

        image = self.get_channel(image, multiframe_mode)

        image = self.get_fov(image, channel_mode)

        # if len(image.shape) < 3:
        #
        #     image = np.expand_dims(image, axis=0)

        return image

    def stack_images(self, images, metadata=None):

        if len(images) != 0:
            dims = []

            for img in images:
                dims.append([img.shape[0], img.shape[1]])

            dims = np.array(dims)

            stack_dim = max(dims[:, 0]), max(dims[:, 1])

            image_stack = []

            for i in range(len(images)):
                img = images[i]

                img_temp = np.zeros(stack_dim, dtype=img.dtype)
                # # img_temp[:] = np.nan

                y_centre = (img_temp.shape[0]) // 2
                x_centre = (img_temp.shape[1]) // 2

                if (img.shape[0] % 2) == 0:
                    y1 = y_centre - img.shape[0] // 2
                    y2 = y1 + img.shape[0]
                else:
                    y1 = int(y_centre - img.shape[0] / 2 + 0.5)
                    y2 = y1 + img.shape[0]

                if (img.shape[1] % 2) == 0:
                    x1 = x_centre - img.shape[1] // 2
                    x2 = x1 + img.shape[1]
                else:
                    x1 = int(x_centre - img.shape[1] / 2 + 0.5)
                    x2 = x1 + img.shape[1]

                img_temp[y1:y2, x1:x2] = img

                image_stack.append(img_temp)

                if metadata:
                    try:
                        metadata[i]["crop"] = [y1, y2, x1, x2]
                    except:
                        pass

            image_stack = np.stack(image_stack, axis=0)

        else:
            image_stack = images
            metadata = metadata

        return image_stack, metadata

    def unstack_images(self, stack, axis=0):

        images = [np.squeeze(e, axis) for e in np.split(stack, stack.shape[axis], axis=axis)]

        return images

    def append_image_stacks(self, current_metadata, new_metadata, current_image_stack, new_image_stack):

        current_image_stack = self.unstack_images(current_image_stack)

        new_image_stack = self.unstack_images(new_image_stack)

        appended_image_stack = current_image_stack + new_image_stack

        appended_metadata = current_metadata

        for key, value in new_metadata.items():
            new_key = max(appended_metadata.keys()) + 1

            appended_metadata[new_key] = value

        appended_image_stack, appended_metadata = self.stack_images(appended_image_stack, appended_metadata)

        return appended_image_stack, appended_metadata

    def append_metadata(self, current_metadata, new_metadata):

        appended_metadata = current_metadata

        for key, value in new_metadata.items():
            new_key = max(appended_metadata.keys()) + 1

            appended_metadata[new_key] = value

            return appended_metadata

    def read_ak_metadata(self):

        meta_path = os.path.join(self.database_path, "Metadata", "AKSEG Metadata.xlsx")

        ak_meta = pd.read_excel(meta_path)

        user_initials = list(ak_meta["User Initials"].dropna())
        microscope = list(ak_meta["Microscope"].dropna())
        modality = list(ak_meta["Image Modality"].dropna())

        ak_meta = dict(user_initials=user_initials, microscope=microscope, modality=modality)

        return ak_meta

    def get_hash(self, img_path=None, img=None):

        if img is not None:
            img_path = tempfile.TemporaryFile(suffix=".tif").name
            tifffile.imwrite(img_path, img)

            with open(img_path, "rb") as f:
                bytes = f.read()  # read entire file as bytes
                hash_code = hashlib.sha256(bytes).hexdigest()

            os.remove(img_path)

        else:
            with open(img_path, "rb") as f:
                bytes = f.read()  # read entire file as bytes
                hash_code = hashlib.sha256(bytes).hexdigest()

        return hash_code

    def align_image_channels(self):

        layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Nucleoid", "Classes", "center_lines", "Localisations"]]

        if self.import_align.isChecked() and len(layer_names) > 1:
            primary_image = layer_names[-1]

            layer_names.remove(primary_image)

            dim_range = int(self.viewer.dims.range[0][1])

            for i in range(dim_range):
                img = self.viewer.layers[primary_image].data[i, :, :]

                for layer in layer_names:
                    shifted_img = self.viewer.layers[layer].data[i, :, :]

                    try:
                        shift, error, diffphase = phase_cross_correlation(img, shifted_img, upsample_factor=100)
                        shifted_img = scipy.ndimage.shift(shifted_img, shift)

                    except:
                        pass

                    self.viewer.layers[layer].data[i, :, :] = shifted_img

    def get_histogram(self, image, bins):
        """calculates and returns histogram"""
    
        # array with size of bins, set to zeros
        histogram = np.zeros(bins)
    
        # loop through pixels and sum up counts of pixels
    
        for pixel in image:
            try:
                histogram[pixel] += 1
            except:
                pass
    
        return histogram

    def cumsum(self, a):
        """cumulative sum function"""
    
        a = iter(a)
        b = [next(a)]
        for i in a:
            b.append(b[-1] + i)
        return np.array(b)

    def autocontrast_values(self, image, clip_hist_percent=0.001):
        # calculate histogram
        hist, bin_edges = np.histogram(image, bins=(2 ** 16) - 1)
        hist_size = len(hist)
    
        # calculate cumulative distribution from the histogram
        accumulator = self.cumsum(hist)
    
        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= maximum / 100.0
        clip_hist_percent /= 2.0
    
        # Locate left cut
        minimum_gray = 0
        try:
            while accumulator[minimum_gray] < clip_hist_percent:
                minimum_gray += 1
        except:
            pass
    
        # Locate right cut
        maximum_gray = hist_size - 1
        try:
            while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
                maximum_gray -= 1
        except:
            pass
    
        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha
    
        # calculate gamma value
        img = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        mid = 0.5
        mean = np.mean(img)
        gamma = np.log(mid * 255) / np.log(mean)
    
        if gamma > 2:
            gamma = 2
        if gamma < 0.2:
            gamma = 0.2
    
        if maximum_gray > minimum_gray:
            contrast_limit = [minimum_gray, maximum_gray]
        else:
            contrast_limit = [np.min(image), np.max(image)]
    
        return contrast_limit, alpha, beta, gamma

    def get_contours_from_mask(self, mask, label, export_labels):
    
        export_mask = np.zeros(mask.shape, dtype=np.uint16)
        export_label = np.zeros(mask.shape, dtype=np.uint16)
    
        contours = []
    
        mask_ids = np.unique(mask)
    
        export_labels = [int(label) for label in export_labels]
    
        new_mask = np.zeros(mask.shape, dtype=np.uint16)
    
        new_mask_id = 1
    
        for mask_id in mask_ids:
            try:
                if mask_id != 0:
                    cnt_mask = np.zeros(mask.shape, dtype=np.uint8)
    
                    cnt_mask[mask == mask_id] = 255
                    label_id = int(np.unique(label[cnt_mask == 255])[0])
    
                    if label_id in export_labels:
    
                        export_mask[cnt_mask == 255] = new_mask_id
                        export_label[cnt_mask == 255] = label_id
    
                        cnt, _ = cv2.findContours(cnt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, )
    
                        contours.append(cnt[0])
    
                        new_mask_id += 1
    
            except:
                pass
    
        return contours, export_mask, export_label

    def automatic_brightness_and_contrast(self, image, clip_hist_percent=0.1):
        if np.max(image) > 0:
            # Calculate grayscale histogram
            hist = cv2.calcHist([image], [0], None, [2 ** 16], [0, 2 ** 16])
            hist_size = len(hist)
    
            # Calculate cumulative distribution from the histogram
            accumulator = []
            accumulator.append(float(hist[0]))
            for index in range(1, hist_size):
                accumulator.append(accumulator[index - 1] + float(hist[index]))
    
            # Locate points to clip
            maximum = accumulator[-1]
            clip_hist_percent *= maximum / 100.0
            clip_hist_percent /= 2.0
    
            # Locate left cut
            minimum_gray = 0
            while accumulator[minimum_gray] < clip_hist_percent:
                minimum_gray += 1
    
            # Locate right cut
            maximum_gray = hist_size - 1
            while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
                maximum_gray -= 1
    
            # Calculate alpha and beta values
            alpha = 255 / (maximum_gray - minimum_gray)
            beta = -minimum_gray * alpha
    
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
        return image

    def add_scale_bar(self, image,
            pixel_resolution=100, pixel_resolution_units="nm",
            scalebar_size=20, scalebar_size_units="um", scalebar_colour="white", scalebar_thickness=10, scalebar_margin=10, ):
        try:
            if float(pixel_resolution) > 0 and float(scalebar_size) > 0:
                h, w = image.shape
    
                pixel_resolution = float(pixel_resolution)
                scalebar_size = float(scalebar_size)
    
                scalebar_margin = int(w / 100 * scalebar_margin)
    
                if pixel_resolution_units != "nm":
                    rescaled_pixel_resolution = pixel_resolution * 1000
                else:
                    rescaled_pixel_resolution = pixel_resolution
    
                if scalebar_size_units != "nm":
                    rescaled_scalebar_size = scalebar_size * 1000
                else:
                    rescaled_scalebar_size = scalebar_size
    
                scalebar_len = int(rescaled_scalebar_size / rescaled_pixel_resolution)
    
                if scalebar_len > 0 and scalebar_len < w:
                    if scalebar_colour == "White":
                        bit_depth = str(image.dtype)
                        bit_depth = int(bit_depth.replace("uint", ""))
                        colour = (2 ** bit_depth) - 1
                    else:
                        colour = 0
    
                    scalebar_pos = (w - scalebar_margin - scalebar_len, h - scalebar_margin - int(scalebar_thickness),)  # Position of the scale bar in the image (in pixels)
    
                    image = cv2.rectangle(image, scalebar_pos, (scalebar_pos[0] + scalebar_len, scalebar_pos[1] + int(scalebar_thickness),), colour, -1, )
    
                else:
                    show_info(f"{int(scalebar_size)} ({scalebar_size_units}) Scale bar is too large for the {(rescaled_pixel_resolution / 1000) * w}x{(rescaled_pixel_resolution / 1000) * h} (um) image")
    
        except:
            print(traceback.format_exc())
    
        return image

    def _manualImport(self):
        try:
            if (self.viewer.layers.index("Segmentations") != len(self.viewer.layers) - 1):
                # reshapes masks to be same shape as active image
                self.active_layer = self.viewer.layers[-1]
    
                if self.active_layer.metadata == {}:
                    active_image = self.active_layer.data
    
                    if len(active_image.shape) < 3:
                        active_image = np.expand_dims(active_image, axis=0)
                        self.active_layer.data = active_image
    
                    if self.classLayer.data.shape != self.active_layer.data.shape:
                        self.classLayer.data = np.zeros(active_image.shape, np.uint16)
    
                    if self.segLayer.data.shape != self.active_layer.data.shape:
                        self.segLayer.data = np.zeros(active_image.shape, np.uint16)
    
                    image_name = str(self.viewer.layers[-1]) + ".tif"
    
                    meta = {}
                    for i in range(active_image.shape[0]):
                        img = active_image[i, :, :]
    
                        contrast_limit, alpha, beta, gamma = self.autocontrast_values(img, clip_hist_percent=1)
    
                        img_meta = dict(image_name=image_name, image_path="Unknown", mask_name=None, mask_path=None, label_name=None, label_path=None, folder=None, parent_folder=None, contrast_limit=contrast_limit, contrast_alpha=alpha, contrast_beta=beta, contrast_gamma=gamma, akseg_hash=None, import_mode="manual", dims=[
                            img.shape[1], img.shape[0]], crop=[0, img.shape[0], 0, img.shape[1]], frame=i, frames=active_image.shape[0], )
    
                        meta[i] = img_meta
    
                    self.active_layer.metadata = meta
                    self.segLayer.metadata = meta
                    self.classLayer.metadata = meta
    
                    self._updateFileName()
                    self._updateSegmentationCombo()
    
                    self.viewer.reset_view()
                    self._autoContrast()
                    self._autoClassify()
    
        except:
            pass
