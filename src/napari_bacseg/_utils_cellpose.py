import os
import traceback
import warnings

import cv2
import numpy as np
from napari.utils.notifications import show_info
from PyQt5.QtWidgets import QFileDialog


def export_cellpose(file_path, image, mask):
    flow = np.zeros(mask.shape, dtype=np.uint16)
    outlines = np.zeros(mask.shape, dtype=np.uint16)
    mask_ids = np.unique(mask)

    colours = []
    ismanual = []

    for i in range(1, len(mask_ids)):
        try:
            cell_mask = np.zeros(mask.shape, dtype=np.uint8)
            cell_mask[mask == i] = 255

            contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt = contours[0]

            outlines = cv2.drawContours(outlines, cnt, -1, i, 1)

            colour = np.random.randint(0, 255, (3), dtype=np.uint16)
            colours.append(colour)
            ismanual.append(True)

            base = os.path.splitext(file_path)[0]

        except:
            pass

    np.save(base + "_seg.npy", {"img": image.astype(np.uint32), "colors": colours, "outlines": outlines.astype(np.uint16) if outlines.max() < 2 ** 16 - 1 else outlines.astype(np.uint32), "masks": mask.astype(np.uint16) if mask.max() < 2 ** 16 - 1 else mask.astype(np.uint32), "chan_choose": [
        0, 0], "ismanual": ismanual, "filename": file_path, "flows": flow, "est_diam": 15, }, )


def _postpocess_cellpose(self, mask):
    try:
        min_seg_size = int(self.cellpose_min_seg_size.currentText())

        post_processed_mask = np.zeros(mask.shape, dtype=np.uint16)

        mask_ids = sorted(np.unique(mask))

        for i in range(1, len(mask_ids)):
            cell_mask = np.zeros(mask.shape, dtype=np.uint8)
            cell_mask[mask == i] = 255

            contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            cnt = contours[0]

            area = cv2.contourArea(cnt)

            if area > min_seg_size:
                post_processed_mask[mask == i] = i
    except:
        post_processed_mask = mask

    return post_processed_mask


def _run_cellpose(self, progress_callback, images):
    mask_stack = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        flow_threshold = float(self.cellpose_flowthresh_label.text())
        mask_threshold = float(self.cellpose_maskthresh_label.text())
        min_size = int(self.cellpose_minsize_label.text())
        diameter = int(self.cellpose_diameter_label.text())

        model_type = self.cellpose_segmodel.currentText()
        model_path = self.cellpose_custom_model_path

        model, gpu, labels_to_flows = _initialise_cellpose_model(self, model_type, model_path, diameter)

        if model != None:
            masks = []
            if self.widget_notifications:
                show_info(f"Segmenting {len(images)} images")

            for i in range(len(images)):
                progress = int(((i + 1) / len(images)) * 100)

                try:
                    progress_callback.emit(progress)
                except:
                    pass

                image = images[i]

                if self.cellpose_invert_images.isChecked():
                    image = cv2.bitwise_not(image)
                    print(True)

                mask, flow, diam = model.eval(
                    image,
                    diameter=diameter,
                    channels=[0, 0],
                    flow_threshold=flow_threshold,
                    cellprob_threshold=mask_threshold,
                    min_size=min_size, batch_size=3, )

                mask = _postpocess_cellpose(self, mask)

                masks.append(mask)

            mask_stack = np.stack(masks, axis=0)

        if gpu:
            import torch

            torch.cuda.empty_cache()

        return mask_stack


def _process_cellpose(self, segmentation_data):
    if len(segmentation_data) > 0:
        masks = segmentation_data

        if self.cellpose_seg_mode.currentIndex() == 0:
            output_layer = self.segLayer
        else:
            output_layer = self.nucLayer

        if output_layer.data.shape != masks.shape:
            current_fov = self.viewer.dims.current_step[0]
            output_layer.data[current_fov, :, :] = masks

        else:
            output_layer.data = masks

        output_layer.contour = 1
        output_layer.opacity = 1

        self.cellpose_segmentation = True
        self.cellpose_progressbar.setValue(0)

        if self.cellpose_seg_mode.currentIndex() == 0:
            self._autoClassify(reset=True)

        self._autoContrast()

        if self.cellpose_resetimage.isChecked() == True:
            self.viewer.reset_view()

        self._reorderLayers()

        # layer_names = [  #     layer.name  #     for layer in self.viewer.layers  #     if layer.name not in ["Segmentations", "Classes", "center_lines"]  # ]  #  # # ensures segmentation and classes is in correct order in the viewer  # for layer in layer_names:  #     layer_index = self.viewer.layers.index(layer)  #     self.viewer.layers.move(layer_index, 0)


def load_cellpose_dependencies(self, omni=False):
    if self.widget_notifications:
        show_info("Loading Cellpose dependencies")

    if omni == True:
        from omnipose.core import labels_to_flows
    else:
        from cellpose.dynamics import labels_to_flows

    import torch
    from cellpose import models

    model = None
    gpu = False

    if torch.cuda.is_available() and self.cellpose_usegpu.isChecked():
        if self.widget_notifications:
            show_info("Cellpose Using GPU")
        gpu = True
        torch.cuda.empty_cache()

    return gpu, models, labels_to_flows


def _initialise_cellpose_model(self, model_type="custom", model_path=None, diameter=15, mode="eval"):
    model = None
    gpu = False
    omnipose_model = False
    labels_to_flows = None

    if model_type == "custom":
        if model_path not in ["", None] and os.path.isfile(model_path) == True:
            model_name = os.path.basename(model_path)

            if "omnipose_" in model_name and mode == "eval":
                load_model = True
                omnipose_model = True
                if self.widget_notifications:
                    show_info(f"Loading Cellpose Model: {os.path.basename(model_path)}")
            elif "cellpose_" in model_name:
                load_model = True
                omnipose_model = False
                if self.widget_notifications:
                    show_info(f"Loading Cellpose Model: {os.path.basename(model_path)}")
            else:
                if self.widget_notifications:
                    show_info(f"BacSeg Cannot train Omnipose models")
                load_model = False

            if load_model == True:
                gpu, models, labels_to_flows = load_cellpose_dependencies(self, omnipose_model)

                model = models.CellposeModel(pretrained_model=model_path, diam_mean=diameter, model_type=None, gpu=gpu, net_avg=False, )

            else:
                model, gpu, omnipose_model, labels_to_flows = (None, None, True, None,)

        else:
            if self.widget_notifications:
                show_info("Please select valid Cellpose Model")

    else:
        if "_omni" in model_type and mode == "eval":
            load_model = True
            omnipose_model = True
            if self.widget_notifications:
                show_info(f"Loading Omnipose Model: {model_type}")
        elif "omni" not in model_type:
            load_model = True
            omnipose_model = False
            if self.widget_notifications:
                show_info(f"Loading Cellpose Model: {model_type}")
        else:
            load_model = False

        if load_model == True:
            gpu, models, labels_to_flows = load_cellpose_dependencies(self, omnipose_model)

            model = models.CellposeModel(diam_mean=diameter, model_type=model_type, gpu=gpu, net_avg=False, )

            self.cellpose_model = model

        else:
            if self.widget_notifications:
                show_info(f"BacSeg Cannot train Omnipose models")

            model, gpu, omnipose_model, labels_to_flows = (None, None, True, None,)

    return model, gpu, labels_to_flows


def unstack_images(stack, axis=0):
    images = [np.squeeze(e, axis) for e in np.split(stack, stack.shape[axis], axis=axis)]

    return images


def train_cellpose_model(self, progress_callback=0):
    try:
        channel = self.cellpose_trainchannel.currentText()

        images = self.viewer.layers[channel].data
        images = unstack_images(images, axis=0)

        if self.cellpose_seg_mode.currentIndex() == 0:
            masks = self.segLayer.data
            masks = unstack_images(masks, axis=0)
        else:
            masks = self.nucleiLayer.data
            masks = unstack_images(masks, axis=0)

        nepochs = int(self.cellpose_nepochs.currentText())
        batchsize = int(self.cellpose_batchsize.currentText())
        model_type = self.cellpose_trainmodel.currentText()
        model_load_path = self.cellpose_custom_model_path
        model_save_path = self.cellpose_train_model_path
        diameter = int(self.cellpose_diameter_label.text())

        model, gpu, labels_to_flows = _initialise_cellpose_model(self, model_type, model_load_path, diameter, mode="train")

        if model != None:
            from cellpose.io import logger_setup

            logger, log_file = logger_setup()

            progress_callback = log_file

            self.cellpose_log_file = log_file

            file_path = self.cellpose_train_model_path
            if self.widget_notifications:
                show_info("Generating Cellpose training dataset...")

            flows = labels_to_flows(masks, use_gpu=False, device=None, redo_flows=False)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)

                if self.widget_notifications:
                    show_info("Training Cellpose Model...")

                pretrained_model = model.train(images, flows, channels=[0, 0], batch_size=batchsize, n_epochs=nepochs, save_path=file_path, save_every=1, )
                if self.widget_notifications:
                    show_info(f"Training Complete. Cellpose model path: {pretrained_model}")

    except:
        print(traceback.format_exc())

    if gpu:
        import torch

        torch.cuda.empty_cache()


def _select_cellpose_save_path(self):
    desktop = os.path.expanduser("~/Desktop")
    path = QFileDialog.getExistingDirectory(self, "Select Save Directory", desktop)

    if os.path.isdir(path):
        self.cellpose_train_model_path = path


def _select_cellpose_save_directory(self):
    if self.database_path != "":
        file_path = os.path.join(self.database_path, "Models")
    else:
        file_path = os.path.expanduser("~/Desktop")

    path = QFileDialog.getExistingDirectory(self, "Select Cellpose Save Directory", file_path)

    if path != "":
        if os.path.isdir(path):
            self.cellpose_train_model_path = os.path.abspath(path)

            if self.widget_notifications:
                show_info(f"Cellpose model save path: {self.cellpose_train_model_path}")


def _select_custom_cellpose_model(self, path=None):
    cellpose_model_names = ["bact_phase_cp", "bact_fluor_cp", "plant_cp", "worm_cp", "cyto2_omni", "bact_phase_omni", "bact_fluor_omni", "plant_omni", "worm_omni", "worm_bact_omni",
        "worm_high_res_omni", "cyto", "cyto2", "nuclei", ]

    if self.database_path != "":
        file_path = os.path.join(self.database_path, "Models")
    else:
        file_path = os.path.expanduser("~/Desktop")

    if type(path) != str:
        path, _ = QFileDialog.getOpenFileName(self, "Open File", file_path, "Cellpose Models (*)")

    if path != "":
        model_name = os.path.basename(path)

        if "torch" in model_name:
            model_name = model_name.split("torch")[0]

        if ("cellpose_" in model_name or "omnipose_" in model_name or model_name in cellpose_model_names):
            if os.path.isfile(path):
                self.cellpose_custom_model_path = path
                self.cellpose_segmodel.setCurrentIndex(6)
                self.cellpose_trainmodel.setCurrentIndex(6)

                if "_omni" in model_name or "omnipose_" in model_name:
                    if self.widget_notifications:
                        show_info(f"Selected Omnipose model: {model_name}")
                else:
                    if self.widget_notifications:
                        show_info(f"Selected Cellpose model: {model_name}")

            else:
                if self.widget_notifications:
                    show_info("Custom Cellpose model path is invalid")

        else:
            if self.widget_notifications:
                show_info("Custom Cellpose model path is invalid")

    else:
        if self.widget_notifications:
            show_info("Custom Cellpose model path is invalid")
