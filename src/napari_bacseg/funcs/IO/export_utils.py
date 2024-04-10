import os
import traceback
import warnings
import cv2
import numpy as np
import pandas as pd
import tifffile
from napari.utils.notifications import show_info


class _export_utils:

    def get_export_data(self, mask_stack, label_stack, meta_stack):
        export_labels = []

        if self.export_single.isChecked():
            export_labels.append(1)
        if self.export_dividing.isChecked():
            export_labels.append(2)
        if self.export_divided.isChecked():
            export_labels.append(3)
        if self.export_vertical.isChecked():
            export_labels.append(4)
        if self.export_broken.isChecked():
            export_labels.append(5)
        if self.export_edge.isChecked():
            export_labels.append(6)

        export_mask_stack = np.zeros(mask_stack.shape, dtype=np.uint16)
        export_label_stack = np.zeros(label_stack.shape, dtype=np.uint16)
        export_contours = {}

        for i in range(len(mask_stack)):
            meta = meta_stack[i]
            y1, y2, x1, x2 = meta["crop"]

            mask = mask_stack[i, :, :][y1:y2, x1:x2]
            label = label_stack[i, :, :][y1:y2, x1:x2]

            export_mask = np.zeros(mask.shape, dtype=np.uint16)
            export_label = np.zeros(mask.shape, dtype=np.uint16)
            contours = []

            mask_ids = np.unique(mask)

            for mask_id in mask_ids:
                if mask_id != 0:
                    cnt_mask = np.zeros(mask.shape, dtype=np.uint8)

                    cnt_mask[mask == mask_id] = 255
                    label_id = np.unique(label[cnt_mask == 255])[0]

                    if label_id in export_labels:
                        new_mask_id = np.max(np.unique(export_mask)) + 1
                        export_mask[cnt_mask == 255] = new_mask_id
                        export_label[cnt_mask == 255] = label_id

                        cnt, _ = cv2.findContours(cnt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, )

                        contours.append(cnt[0])

            export_mask_stack[i, :, :][y1:y2, x1:x2] = export_mask
            export_label_stack[i, :, :][y1:y2, x1:x2] = export_label
            export_contours[i] = contours

        return export_mask_stack, export_label_stack, export_contours

    def get_export_labels(self):
        export_labels = []

        if self.export_single.isChecked():
            export_labels.append(1)
        if self.export_dividing.isChecked():
            export_labels.append(2)
        if self.export_divided.isChecked():
            export_labels.append(3)
        if self.export_vertical.isChecked():
            export_labels.append(4)
        if self.export_broken.isChecked():
            export_labels.append(5)
        if self.export_edge.isChecked():
            export_labels.append(6)

        return export_labels

    def generate_export_image(self, export_channel, dim, normalize=False, invert=False,
            autocontrast=False, scalebar=False, cropzoom=False, mask_background=False, ):

        layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Nucleoid", "Classes", "center_lines", "Localisations"]]

        layer_names.reverse()

        if export_channel == "Multi Channel":

            multi_channel_mode = self.export_multi_channel_mode.currentText()

            if multi_channel_mode == "All Channels (Stack)":
                mode = "stack"
            if multi_channel_mode == "All Channels (Horizontal Stack)":
                mode = "hstack"
            elif multi_channel_mode == "All Channels (Vertical Stack)":
                mode = "vstack"
            elif multi_channel_mode == "First Three Channels (RGB)":
                mode = "rgb"
                layer_names = layer_names[:3]
        else:
            mode = "single"
            layer_names = [export_channel]

        mask = self.segLayer.data
        nmask = self.nucLayer.data
        label = self.classLayer.data
        metadata = self.viewer.layers[layer_names[0]].metadata

        mask = mask[dim]
        nmask = nmask[dim]
        label = label[dim]

        try:
            metadata = metadata[dim[0]]
        except:
            metadata = {}

        if cropzoom:
            layer = self.viewer.layers[layer_names[0]]
            crop = layer.corner_pixels.T
            y_range = crop[-2]
            x_range = crop[-1]
            mask = mask[y_range[0]: y_range[1], x_range[0]: x_range[1]]
            nmask = nmask[y_range[0]: y_range[1], x_range[0]: x_range[1]]
            label = label[y_range[0]: y_range[1], x_range[0]: x_range[1]]

        image = []

        for layer in layer_names:
            img = self.viewer.layers[layer].data.copy()

            img = img[dim]

            if cropzoom:
                img = img[y_range[0]: y_range[1], x_range[0]: x_range[1]]

            if mask_background:
                img[mask == 0] = 0

            if invert:
                img = cv2.bitwise_not(img)

            if normalize:
                img = self.normalize99(img)

            if autocontrast:
                img = self.automatic_brightness_and_contrast(img)

            if scalebar:
                pixel_resolution = self.export_scalebar_resolution.text()
                pixel_resolution_units = (self.export_scalebar_resolution_units.currentText())
                scalebar_size = self.export_scalebar_size.text()
                scalebar_size_units = self.export_scalebar_size_units.currentText()
                scalebar_colour = self.export_scalebar_colour.currentText()
                scalebar_thickness = self.export_scalebar_thickness.currentText()

                img = self.add_scale_bar(img, pixel_resolution=pixel_resolution, pixel_resolution_units=pixel_resolution_units, scalebar_size=scalebar_size, scalebar_size_units=scalebar_size_units, scalebar_colour=scalebar_colour, scalebar_thickness=scalebar_thickness, )

            image.append(img)

        if mode == "rgb":
            while len(image) < 3:
                blank = np.zeros(img.shape, dtype=img.dtype)

                if scalebar:
                    blank = self.add_scale_bar(blank, pixel_resolution=pixel_resolution, pixel_resolution_units=pixel_resolution_units, scalebar_size=scalebar_size, scalebar_size_units=scalebar_size_units, scalebar_colour=scalebar_colour, scalebar_thickness=scalebar_thickness, )

                image.append(blank)

        if mode == "rgb":
            image = np.stack(image, axis=-1)

            image = self.rescale01(image)
            image = image * (2 ** 16 - 1)
            image = image.astype(np.uint16)

        elif mode == "stack":
            image = np.stack(image, axis=0)
        elif mode == "hstack":
            image = np.hstack(image)
        elif mode == "vstack":
            image = np.vstack(image)
        else:
            image = image[0]

        return image, mask, nmask, label, metadata, mode

    def export_stacks(self, progress_callback, mode):
        try:
            layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Nucleoid", "Classes", "center_lines", "Localisations"]]

            export_stack_channel = self.export_stack_channel.currentText()
            export_stack_mode = self.export_stack_mode.currentText()
            export_stack_modifier = self.export_stack_modifier.text()

            if mode == "active":
                export_channels = [self.export_stack_channel.currentText()]
            else:
                export_channels = [layer.name for layer in self.viewer.layers]

            export_channels = [channel for channel in export_channels if channel not in ["Segmentations", "Nucleoid", "Classes", "center_lines", "Localisations"]]

            if len(export_channels) == 0:
                show_info("No image layers to export")
            else:
                overwrite = self.export_stack_overwrite_setting.isChecked()
                export_images = self.export_stack_image_setting.isChecked()

                normalise = self.export_normalise.isChecked()
                invert = self.export_invert.isChecked()
                autocontrast = self.export_autocontrast.isChecked()
                scalebar = self.export_scalebar.isChecked()
                cropzoom = self.export_cropzoom.isChecked()
                mask_background = self.export_mask_background.isChecked()

                for channel in export_channels:
                    image_stack = []
                    mask_stack = []

                    dims = self.viewer.layers[channel].data.shape[0]

                    for dim in range(dims):
                        progress_callback.emit(int((dim / (dims - 1)) * 100))

                        image, mask, nmask, label, meta, mode = self.generate_export_image(channel, (dim,), normalise, invert, autocontrast, scalebar, cropzoom, mask_background, )

                        if len(image.shape) > 2:
                            image = image[0]
                            mask = mask[0]

                        image_stack.append(image)
                        mask_stack.append(mask)

                    image_stack = np.stack(image_stack, axis=0)
                    mask_stack = np.stack(mask_stack, axis=0)

                    file_name = meta["image_name"]

                    if "image_path" in meta.keys():
                        image_path = meta["image_path"]
                    if "path" in meta.keys():
                        image_path = meta["path"]

                    image_path = os.path.normpath(image_path)
                    file_name, file_extension = os.path.splitext(file_name)

                    file_name = file_name + export_stack_modifier + ".tif"
                    image_path = image_path.replace(image_path.split(os.sep)[-1], file_name)

                    if (self.export_stack_location.currentText() == "Import Directory" and file_name != None and image_path != None):
                        export_path = os.path.abspath(image_path.replace(file_name, ""))

                    elif (self.export_stack_location.currentText() == "Select Directory"):
                        export_path = os.path.abspath(self.export_directory)

                    else:
                        export_path = None

                    if os.path.isdir(export_path) != True:
                        if self.widget_notifications:
                            show_info("Directory does not exist, try selecting a directory instead!")

                    else:
                        y1, y2, x1, x2 = meta["crop"]

                        if len(image.shape) > 2:
                            image = image[:, y1:y2, x1:x2]
                            mask = mask[:, y1:y2, x1:x2]
                        else:
                            image = image[y1:y2, x1:x2]
                            mask = mask[y1:y2, x1:x2]

                        if os.path.isdir(export_path) == False:
                            os.makedirs(file_path)

                        file_path = export_path + os.sep + file_name

                        if os.path.isfile(file_path) == True and overwrite == False:
                            if self.widget_notifications:
                                show_info(file_name + " already exists, BacSeg will not overwrite files!")

                        else:
                            if export_stack_mode == "Export .tif Images":
                                tifffile.imwrite(file_path, image_stack, metadata=meta)

                            if export_stack_mode == "Export .tif Masks":
                                tifffile.imwrite(file_path, mask_stack, metadata=meta)

        except:
            print(traceback.format_exc())

    def export_files(self, progress_callback, mode):
        desktop = os.path.expanduser("~/Desktop")

        overwrite = self.export_overwrite_setting.isChecked()
        export_images = self.export_image_setting.isChecked()

        normalise = self.export_normalise.isChecked()
        invert = self.export_invert.isChecked()
        autocontrast = self.export_autocontrast.isChecked()
        scalebar = self.export_scalebar.isChecked()
        cropzoom = self.export_cropzoom.isChecked()
        mask_background = self.export_mask_background.isChecked()

        export_channel = self.export_channel.currentText()
        export_modifier = self.export_modifier.text()
        export_labels = self.get_export_labels()

        data_shape = self.viewer.layers[0].data.shape

        viewer_dims = np.array(self.viewer.dims.range[:-2]).astype(int)

        if mode == "active":
            current_dim = self.viewer.dims.current_step[:-2]
            if len(viewer_dims) == 2:
                dim_list = [current_dim]
            else:
                dim_list = [current_dim]

        else:
            dim_list = []
            for image_index in range(*viewer_dims[0]):
                if len(viewer_dims) == 2:
                    for tile_index in range(*viewer_dims[1]):
                        dim_list.append((image_index, tile_index))
                else:
                    dim_list.append((image_index,))

        for i, dim in enumerate(dim_list):

            image, mask, nmask, label, meta, mode = self.generate_export_image(export_channel, dim,
                normalise, invert, autocontrast, scalebar, cropzoom, mask_background, )

            contours, mask, label = self.get_contours_from_mask(mask, label, export_labels)

            if "midlines" in meta.keys():
                midlines = meta["midlines"].copy()
            else:
                midlines = None

            if "shape" in meta.keys():
                meta.pop("shape")

            file_name = meta["image_name"]

            if "image_path" in meta.keys():
                image_path = meta["image_path"]
            if "path" in meta.keys():
                image_path = meta["path"]

            image_path = os.path.normpath(image_path)

            file_name, file_extension = os.path.splitext(file_name)

            if len(dim) == 2:
                file_name = file_name + f"_{dim}"

            file_name = file_name + export_modifier + ".tif"
            image_path = image_path.replace(image_path.split(os.sep)[-1], file_name)

            if (self.export_location.currentText() == "Import Directory" and file_name != None and image_path != None):
                export_path = os.path.abspath(image_path.replace(file_name, ""))
                export_path = os.path.normpath(export_path)

            elif self.export_location.currentText() == "Select Directory":
                export_path = os.path.abspath(self.export_directory)

            else:
                export_path = None

            if os.path.isdir(export_path) != True:
                if self.widget_notifications:
                    show_info("Directory does not exist, try selecting a directory instead!")

            else:

                if export_channel != "Multi Channel":

                    y1, y2, x1, x2 = meta["crop"]

                    if len(image.shape) > 2:
                        image = image[:, y1:y2, x1:x2]
                    else:
                        image = image[y1:y2, x1:x2]

                    mask = mask[y1:y2, x1:x2]
                    label = label[y1:y2, x1:x2]

                if os.path.isdir(export_path) == False:
                    os.makedirs(file_path)

                file_path = export_path + os.sep + file_name

                if os.path.isfile(file_path) == True and overwrite == False:
                    if self.widget_notifications:
                        show_info(file_name + " already exists, BacSeg will not overwrite files!")

                else:

                    if self.export_mode.currentText() == "Export .tif Images":
                        tifffile.imwrite(file_path, image, metadata=meta)

                    if self.export_mode.currentText() == "Export .tif Masks":
                        tifffile.imwrite(file_path, mask, metadata=meta)

                    if (self.export_mode.currentText() == "Export .tif Images and Masks"):
                        image_path = os.path.abspath(export_path + "\\images")
                        mask_path = os.path.abspath(export_path + "\\masks")

                        if not os.path.exists(image_path):
                            os.makedirs(image_path)

                        if not os.path.exists(mask_path):
                            os.makedirs(mask_path)

                        image_path = os.path.abspath(image_path + os.sep + file_name)
                        mask_path = os.path.abspath(mask_path + os.sep + file_name)

                        tifffile.imwrite(image_path, image, metadata=meta)
                        tifffile.imwrite(mask_path, mask, metadata=meta)

                    if self.export_mode.currentText() == "Export Cellpose":

                        self.export_cellpose(file_path, image, mask)

                        if export_images:
                            tifffile.imwrite(file_path, image, metadata=meta)

                    if self.export_mode.currentText() == "Export Oufti":
                        try:
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore")

                                oufti_data = self.get_oufti_data(image, mask, midlines)

                                if "midlines" in meta.keys():
                                    meta.pop("midlines")

                                self.export_oufti(image, oufti_data, file_path)

                                if export_images:
                                    tifffile.imwrite(file_path, image, metadata=meta)

                        except:
                            raise Exception("BacSeg can't load Cellpose and OUFTI dependencies simultaneously. Restart BacSeg, reload images/masks, then export Oufti")

                    if self.export_mode.currentText() == "Export ImageJ":

                        if mode == "rgb":
                            if self.widget_notifications:
                                show_info("ImageJ can't handle RGB images with annotations, export as image stack instead...")

                        self.export_imagej(image, contours, meta, file_path)

                    if self.export_mode.currentText() == "Export JSON":

                        from napari_bacseg.funcs.IO.json_utils import export_coco_json

                        export_coco_json(file_name, image, mask, nmask, label, file_path)

                        if export_images:
                            tifffile.imwrite(file_path, image, metadata=meta)

                    if self.export_mode.currentText() == "Export CSV":
                        self.export_csv(image, contours, meta, file_path)

                        if export_images:
                            tifffile.imwrite(file_path, image, metadata=meta)

            progress = int(((i + 1) / len(dim_list)) * 100)
            try:
                progress_callback.emit(progress)
            except:
                pass

    def export_csv(self, image, contours, meta, file_path):
        processed_contours = []

        if len(contours) > 0:
            contour_shapes = []
            for cnt in contours:
                try:
                    contour_shapes.append(cnt.shape[0])
                except:
                    pass

            max_length = np.max(contour_shapes)

            for i in range(len(contours)):
                try:
                    cnt = contours[i]
                    cnt = np.vstack(cnt).squeeze().astype(str)

                    if len(cnt.shape) < max_length:
                        cnt = np.pad(cnt, ((0, max_length - cnt.shape[0]), (0, 0)), "constant", constant_values="", )

                    processed_contours.append(cnt)

                except:
                    pass

            try:
                file_extension = file_path.split(".")[-1]
                file_path = file_path.replace(file_extension, "csv")

                processed_contours = np.hstack(processed_contours)
                headers = np.array([[f"x[{str(x)}]", f"y[{str((x))}]"] for x in range(processed_contours.shape[-1] // 2)]).flatten()

                pd.DataFrame(processed_contours, columns=headers).to_csv(file_path, index=False, header=True)

            except:
                print(traceback.format_exc())
