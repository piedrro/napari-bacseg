import itertools
import os
import random
import shutil
import traceback
import unittest
import warnings

import numpy as np
import pandas as pd
import tifffile
from glob2 import glob


class test_bacseg(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)

            from napari import Viewer

            from napari_bacseg._widget import BacSeg

            viewer = Viewer()

            cls.BacSeg = BacSeg(viewer)
            cls.BacSeg.widget_notifications = False

            cls.export_setting_dict = {"mode": ["masks", "images", "images + masks", "oufti", "cellpose", "imagej", "json", "csv", ], "overwrite": [True, False], "export_images": [True,
                                                                                                                                                                                    False], "normalise": [
                True, False], "invert": [True, False], "autocontrast": [True, False], }

            cls.import_images_setting_dict = {"mode": ["images", "scanr", "nim"], "filemode": ["directory", "file"], "import_limit": ["None", "1", "5", "100"], "import_precision": [
                "int16"], "multiframe_mode_index": [0, 1, 2, 3], "crop_mode_index": [0, 1, 2, 3], }

            cls.import_masks_setting_dict = {"mode": ["masks", "imagej", "json", "csv"], "filemode": ["directory", "file"], "import_limit": ["None", "100", "1000"], "import_precision": ["int8",
                                                                                                                                                                                          "int16",
                                                                                                                                                                                          "int32",
                                                                                                                                                                                          "native"], "multiframe_mode_index": [
                0, 1, 2, 3], "crop_mode_index": [0, 1, 2, 3], }

    def perumuation_test(self, function, setting_dict={}, limit=10, shuffle=True):
        modes = setting_dict.pop("mode")

        for mode in modes:
            settings = setting_dict.values()

            possible_settings_values = list(itertools.product(*settings))

            if shuffle == True:
                random.shuffle(possible_settings_values)

            possible_settings_values = possible_settings_values[:limit]

            for setting in possible_settings_values:
                active_settings = dict(zip(setting_dict.keys(), setting))

                with self.subTest(msg="getAll"):
                    function(mode=mode, **active_settings)

    def import_images(self, mode="images", filemode="directory", import_limit="None", import_precision="int16", multiframe_mode_index=0, crop_mode_index=0, ):
        if filemode == "directory":
            self.BacSeg.import_filemode.setCurrentText("Import Directory")
        elif filemode == "file":
            self.BacSeg.import_filemode.setCurrentText("Import File")

        self.BacSeg.import_limit.setCurrentText(import_limit)

        if import_precision in ["int8", "int16", "int32", "native"]:
            self.BacSeg.import_precision.setCurrentText(import_precision)

        if multiframe_mode_index in [0, 1, 2, 3]:
            self.BacSeg.import_multiframe_mode.setCurrentIndex(multiframe_mode_index)

        if crop_mode_index in [0, 1, 2, 3]:
            self.BacSeg.import_crop_mode.setCurrentIndex(crop_mode_index)

        if mode.lower() == "images":
            self.BacSeg.import_mode.setCurrentText("Images")
            paths = glob("test_data/images/Images/*.tif")

            assert len(paths) > 0

            target_num_images = len(paths)
            target_channels = ["Image"]

            from napari_bacseg.funcs.utils import import_images

            self.BacSeg.import_images = self.BacSeg.wrapper(import_images)

            data = self.BacSeg.import_images(file_paths=paths, progress_callback=None)

        if mode.lower() == "scanr":
            self.BacSeg.import_mode.setCurrentText("ScanR Data")
            paths = glob("test_data/images/ScanR Images/**/*.tif")

            assert len(paths) > 0

            from napari_bacseg.funcs.utils import (read_scanr_directory, read_scanr_images, )

            self.BacSeg.read_scanr_directory = self.BacSeg.wrapper(read_scanr_directory)
            self.BacSeg.read_scanr_images = self.BacSeg.wrapper(read_scanr_images)

            (measurements, file_paths, channels,) = self.BacSeg.read_scanr_directory(paths)
            target_num_images = len(measurements)
            target_channels = channels

            data = self.BacSeg.read_scanr_images(measurements=measurements, channels=channels, progress_callback=None, )

        if mode.lower() == "nim":
            self.BacSeg.import_mode.setCurrentText("NanoImager Data")
            paths = glob("test_data/images/NIM Images/**/*.tif")

            assert len(paths) > 0

            from napari_bacseg.funcs.utils import (read_nim_directory, read_nim_images, )

            self.BacSeg.read_nim_directory = self.BacSeg.wrapper(read_nim_directory)
            self.BacSeg.read_nim_images = self.BacSeg.wrapper(read_nim_images)

            (measurements, file_paths, channels,) = self.BacSeg.read_nim_directory(paths)
            target_num_images = len(measurements)
            target_channels = channels

            data = self.BacSeg.read_nim_images(measurements=measurements, channels=channels, progress_callback=None, )

        self.BacSeg._process_import(data)

        layer_names = [layer.name for layer in self.BacSeg.viewer.layers if layer.name not in ["Segmentations", "Nucleoid", "Classes", "center_lines"]]

        loaded_images = self.BacSeg.viewer.layers[layer_names[0]].data

        if import_limit != "None":
            if target_num_images > int(import_limit):
                target_num_images = int(import_limit)

        assert len(loaded_images) == target_num_images
        assert len(layer_names) == len(target_channels)

    def read_mask_files(self, paths):
        mask_paths = glob("test_data/images + masks/masks/*.tif")

        file_names = [os.path.basename(path.split(path.split(".")[-1])[0]) + "tif" for path in paths]

        paths = [path for path in mask_paths if os.path.basename(path) in file_names]

        mask_dict = {}

        for path in paths:
            mask = tifffile.imread(path)
            num_masks = len(np.unique(mask))

            mask_dict[os.path.basename(path).strip()] = num_masks

        return mask_dict

    def import_masks(self, mode="masks", randomise=False):
        self.BacSeg.import_mode.setCurrentText("Images")
        image_paths = glob("test_data/images + masks/images/*.tif")

        assert len(image_paths) > 0

        if randomise:
            image_paths = random.sample(image_paths, random.randint(1, len(image_paths)))

        from napari_bacseg.funcs.utils import (get_hash, import_imagej, import_images, import_masks, )

        self.BacSeg.import_masks = self.BacSeg.wrapper(import_masks)
        self.BacSeg.import_images = self.BacSeg.wrapper(import_images)
        self.BacSeg.import_imagej = self.BacSeg.wrapper(import_imagej)
        self.BacSeg.get_hash = self.BacSeg.wrapper(get_hash)

        data = self.BacSeg.import_images(file_paths=image_paths, progress_callback=None)

        self.BacSeg._process_import(data)

        if mode == "masks":
            self.BacSeg.import_mode.setCurrentText("Mask (.tif) Segmentation(s)")

            mask_paths = glob("test_data/images + masks/masks/*.tif")
            assert len(mask_paths) > 0

            self.BacSeg.import_masks(mask_paths, file_extension=".tif")
            self.BacSeg._autoClassify()

        if mode == "oufti":
            self.BacSeg.import_mode.setCurrentText("Oufti (.mat) Segmentation(s)")

            mask_paths = glob("test_data/images + masks/oufti/*.mat")
            assert len(mask_paths) > 0

            self.BacSeg.import_masks(mask_paths, file_extension=".mat")
            self.BacSeg._autoClassify()

        if mode == "cellpose":
            self.BacSeg.import_mode.setCurrentText("Cellpose (.npy) Segmentation(s)")

            mask_paths = glob("test_data/images + masks/cellpose/*.npy")
            assert len(mask_paths) > 0

            self.BacSeg.import_masks(mask_paths, file_extension=".npy")
            self.BacSeg._autoClassify()

        if mode == "imagej":
            self.BacSeg.import_mode.setCurrentText("ImageJ files(s)")

            mask_paths = glob("test_data/images + masks/imagej/*.tif")

            assert len(mask_paths) > 0

            data = self.BacSeg.import_imagej(paths=mask_paths, progress_callback=None)

            self.BacSeg._process_import(data)
            self.BacSeg._autoClassify()

        if mode == "json":
            self.BacSeg.import_mode.setCurrentText("JSON (.txt) Segmentation(s)")

            self.BacSeg.import_imagej = self.BacSeg.wrapper(import_imagej)

            mask_paths = glob("test_data/images + masks/json/*.txt")
            assert len(mask_paths) > 0

            data = self.BacSeg.import_imagej(mask_paths)

            self.BacSeg._process_import(data)

        mask_dict = self.read_mask_files(mask_paths)

        layer_names = [layer.name for layer in self.BacSeg.viewer.layers if layer.name not in ["Segmentations", "Nucleoid", "Classes", "center_lines"]]
        for layer in layer_names:
            metadata = self.BacSeg.viewer.layers[layer].metadata

            for index, meta in metadata.items():
                image_name = meta["image_name"].strip()
                mask = self.BacSeg.segLayer.data[index]

                if image_name in mask_dict.keys():
                    num_cells = len(np.unique(mask))
                    target_num_cells = mask_dict[image_name]

                    assert num_cells >= target_num_cells - 5

    def export_data(self, mode="masks", overwrite=False, export_images=False, normalise=False, invert=False, autocontrast=False, export_modifier="_BacSeg", ):
        if self.BacSeg.segLayer.data.shape == (1, 100, 100):
            self.import_masks(mode="masks")

        self.BacSeg.export_directory = "test_data/exported_data"

        layer_names = [layer.name for layer in self.BacSeg.viewer.layers if layer.name not in ["Segmentations", "Nucleoid", "Classes", "center_lines"]]

        if os.path.isdir(self.BacSeg.export_directory) != True:
            os.mkdir(self.BacSeg.export_directory)
        else:
            shutil.rmtree(self.BacSeg.export_directory)
            os.mkdir(self.BacSeg.export_directory)

        if mode == "masks":
            self.BacSeg.export_mode.setCurrentText("Export .tif Masks")
        if mode == "images":
            self.BacSeg.export_mode.setCurrentText("Export .tif Images")
        if mode == "images + masks":
            self.BacSeg.export_mode.setCurrentText("Export .tif Images and Masks")
        if mode == "oufti":
            self.BacSeg.export_mode.setCurrentText("Export Oufti")
        if mode == "cellpose":
            self.BacSeg.export_mode.setCurrentText("Export Cellpose")
        if mode == "imagej":
            self.BacSeg.export_mode.setCurrentText("Export ImageJ")
        if mode == "json":
            self.BacSeg.export_mode.setCurrentText("Export JSON")
        if mode == "csv":
            self.BacSeg.export_mode.setCurrentText("Export CSV")

        self.BacSeg.export_overwrite_setting.setChecked(overwrite)
        self.BacSeg.export_image_setting.setChecked(export_images)
        self.BacSeg.export_normalise.setChecked(normalise)
        self.BacSeg.export_invert.setChecked(invert)
        self.BacSeg.export_autocontrast.setChecked(autocontrast)
        self.BacSeg.export_location.setCurrentText("Select Directory")

        self.BacSeg.export_channel.setCurrentText(layer_names[0])
        self.BacSeg.export_modifier.setText(export_modifier)

        from napari_bacseg.funcs.utils import export_files

        self.BacSeg.export_files = self.BacSeg.wrapper(export_files)

        self.BacSeg.export_files(mode="All", progress_callback=None)

        num_exported_files = len(glob(self.BacSeg.export_directory + "/*"))

        layer_names = [layer.name for layer in self.BacSeg.viewer.layers if layer.name not in ["Segmentations", "Nucleoid", "Classes", "center_lines"]]

        num_images = self.BacSeg.viewer.layers[layer_names[0]].data.shape[0]

        if export_images == True and mode in ["oufti", "cellpose", "json", "csv", ]:
            targeted_image_num = num_images * 2
        elif mode == "images + masks":
            num_exported_files = len(glob(self.BacSeg.export_directory + "/*/*"))
            targeted_image_num = num_images * 2
        else:
            targeted_image_num = num_images

        assert num_exported_files == targeted_image_num

    def load_custom_cellpose_model(self):
        from napari_bacseg.funcs.cellpose_utils import _select_custom_cellpose_model

        self.BacSeg._select_custom_cellpose_model = self.BacSeg.wrapper(_select_custom_cellpose_model)

        model_path = "test_data/cellpose_model/cellpose_residual_on_style_on_concatenation_off_Image_2022_08_01_17_24_23.500813"

        self.BacSeg._select_custom_cellpose_model(path=model_path)

        self.BacSeg.cellpose_segmodel.setCurrentIndex(6)

    def train_cellpose(self, mode="active", model="cyto", import_limit=1):
        cellpose_models = ["cyto", "nuclei", "tissuenet", "livecell", "cyto2", "general", "custom", ]

        if model == "custom":
            self.load_custom_cellpose_model()
        else:
            model_index = cellpose_models.index(model)
            self.BacSeg.cellpose_segmodel.setCurrentIndex(model_index)

        self.import_images(mode="images", import_limit=str(import_limit))

        from napari_bacseg.funcs.utils import unstack_images
        from napari_bacseg.funcs.cellpose_utils import (_process_cellpose, _run_cellpose, )

        self.BacSeg._run_cellpose = self.BacSeg.wrapper(_run_cellpose)
        self.BacSeg._process_cellpose = self.BacSeg.wrapper(_process_cellpose)

        channel = self.BacSeg.cellpose_segchannel.currentText()
        current_fov = self.BacSeg.viewer.dims.current_step[0]

        images = self.BacSeg.viewer.layers[channel].data

        if mode == "active":
            images = [images[current_fov, :, :]]
        else:
            images = unstack_images(images)

        data = self.BacSeg._run_cellpose(images=images, progress_callback=None)
        self.BacSeg._process_cellpose(data)

        masks = self.BacSeg.viewer.layers["Segmentations"].data

        mask_num = len([mask for mask in [len(np.unique(mask)) for mask in masks] if mask > 1])

        assert mask_num == len(images)
        assert len(data) == len(images)

    def load_database(self, path=None):
        if type(path) == type(None):
            test_database = r"test_data/database/BacSeg_Database"
            if os.path.exists(test_database) == True:
                path = test_database
            else:
                path = r"\\physics\dfs\DAQ\CondensedMatterGroups\AKGroup\Piers\AKSEG"

        from napari_bacseg.funcs.database_utils import _load_bacseg_database

        self.BacSeg._load_bacseg_database = self.BacSeg.wrapper(_load_bacseg_database)

        self.BacSeg._load_bacseg_database(path=path)

        self.BacSeg.database_path = path

        return path

    def create_database(self, path=None):
        if type(path) == type(None):
            path = r"test_data/database"

        if os.path.exists(path) == False:
            os.mkdir(path)
        else:
            shutil.rmtree(path)
            os.mkdir(path)

        from napari_bacseg.funcs.database_utils import _create_bacseg_database

        self.BacSeg._create_bacseg_database = self.BacSeg.wrapper(_create_bacseg_database)

        database_directory = self.BacSeg._create_bacseg_database(path=path)

        self.BacSeg.database_path = database_directory

        return database_directory

    def get_database_metadata(self, user_initial="all"):
        database_directory = self.BacSeg.database_path

        if user_initial != "all":
            meta_path = os.path.join(self.BacSeg.database_path, "images", user_initial, f"{user_initial}_file_metadata.txt", )

            if os.path.exists(meta_path) == True:
                metadata = pd.read_csv(meta_path, sep=",", low_memory=False)
            else:
                metadata = pd.DataFrame()

        else:
            meta_files = glob(os.path.join(database_directory, "Images") + "/**/*file_metadata.txt")

            meta_list = []

            for user_metadata_path in meta_files:
                if os.path.exists(user_metadata_path) == True:
                    user_metadata = pd.read_csv(user_metadata_path, sep=",", low_memory=False)
                    meta_list.append(user_metadata)
                else:
                    meta_list.append(pd.DataFrame())

            metadata = pd.concat(meta_list, axis=0, ignore_index=True).reset_index(drop=True)

        metadata = metadata.loc[:, ~metadata.columns.duplicated()].copy()

        return metadata

    def update_database_metadata(self, user_initial):
        control_dict = {"abxconcentration": "upload_abxconcentration", "antibiotic": "upload_antibiotic", "content": "upload_content", "microscope": "upload_microscope", "modality": "label_modality", "mount": "upload_mount", "protocol": "upload_protocol", "source": "label_light_source", "stain": "label_stain", "stain_target": "label_stain_target", "treatment_time": "upload_treatmenttime", "user_initial": "upload_initial", "meta1": "upload_usermeta1", "meta2": "upload_usermeta2", "meta3": "upload_usermeta3", }

        dbmeta = self.get_database_metadata(user_initial=user_initial)

        upload_control_dict = {}

        for meta_key, meta_values in dbmeta.items():
            if meta_key in control_dict.keys():
                control_name = control_dict[meta_key]

                try:
                    combo_box = getattr(self.BacSeg, control_name)

                    upload_control_dict[control_name] = combo_box.currentText()

                except:
                    print(traceback.format_exc())

        from napari_bacseg.funcs.database_utils import update_database_metadata

        self.BacSeg.update_database_metadata = self.BacSeg.wrapper(update_database_metadata)

        self.BacSeg.update_database_metadata(control=None)

        for control_name, control_text in upload_control_dict.items():
            combo_box = getattr(self.BacSeg, control_name)
            combo_box.setCurrentText(control_text)

    # def upload_database(self, path=None, mode="active", user_initial="AK", content="BacSeg", microscope="Nikon", modalilty="Fluorescence", source="LED", stain="None", stain_target="None", overwrite_images=False, overwrite_masks=False, overwrite_all_metadata=False, overwrite_selected_metadata=False, ):  #     with warnings.catch_warnings():  #         warnings.filterwarnings("ignore", category=ResourceWarning)

    #         if type(path) == str:  #             self.create_database(path=path)

    #         pre_upload_metadata = len(self.get_database_metadata(user_initial=user_initial))

    #         self.BacSeg.upload_initial.setCurrentText(user_initial)  #         self.BacSeg.upload_content.setCurrentText(content)  #         self.BacSeg.upload_microscope.setCurrentText(microscope)  #         self.BacSeg.label_modality.setCurrentText(modalilty)  #         self.BacSeg.label_light_source.setCurrentText(source)  #         self.BacSeg.label_stain.setCurrentText(stain)  #         self.BacSeg.label_stain_target.setCurrentText(stain_target)  #         self.BacSeg.upload_overwrite_images.setChecked(overwrite_images)  #         self.BacSeg.upload_overwrite_masks.setChecked(overwrite_masks)  #         self.BacSeg.overwrite_all_metadata.setChecked(overwrite_all_metadata)  #         self.BacSeg.overwrite_selected_metadata.setChecked(overwrite_selected_metadata)

    #         from napari_bacseg._utils_database_IO import (_upload_bacseg_database, )

    #         self.BacSeg._upload_bacseg_database = self.BacSeg.wrapper(_upload_bacseg_database)

    #         self.BacSeg._upload_bacseg_database(mode=mode, progress_callback=None)

    #         post_upload_metadata = len(self.get_database_metadata(user_initial=user_initial))

    #         database_length_increase = (post_upload_metadata - pre_upload_metadata)

    #         self.update_database_metadata(user_initial)

    #     return database_length_increase

    # def test_database_upload(self, mode = "all"):  #  #     self.create_database()  #     self.import_masks(mode="imagej")  #  #     database_length_increase = self.upload_database(mode = mode)  #  #     database_directory = self.BacSeg.database_path  #     user_initial = self.BacSeg.upload_initial.currentText()  #     image_directory = os.path.join(database_directory, "Images", user_initial, "images")  #  #     num_uploads = len(glob(image_directory + "/*/*.tif"))  #  #     if mode == "active":  #         target_num_uploads = 1  #     else:  #         active_layer = self.BacSeg.viewer.layers.selection.active.name  #         target_num_uploads = self.BacSeg.viewer.layers[active_layer].data.shape[0]  #  #     assert target_num_uploads == num_uploads  #     assert target_num_uploads == database_length_increase  #

    # def test_database_download(self, import_limit=1):  #     meta_keys = ["user_initial", "content", "microscope", "antibiotic", "treatment time (mins)", "source", "modality", "stain", "antibiotic concentration", "mounting method", "protocol",  #         "segmented", "segmentation_curated", "labelled", "label_curated", "user_meta1", "user_meta2", "user_meta3", ]

    #     self.load_database()

    #     metadata = self.get_database_metadata()

    #     metadata = metadata[meta_keys]

    #     metadata = metadata.sample(n=import_limit, random_state=0).dropna(axis=1, how="all")

    #     meta_options = {}  #     for column in metadata.columns.values:  #         value = metadata[column].values[0]

    #         if value != "None":  #             meta_options[column] = value

    #     user_initial = meta_options.pop("user_initial")

    #     keys = random.sample(list(meta_options), 3)  #     meta_options = {key: meta_options[key] for key in keys}

    #     self.BacSeg.upload_initial.setCurrentText(user_initial)

    #     # if "content" in meta_options:  #     #     self.BacSeg.upload_content.setCurrentText(meta_options["content"])  #     # if "microscope" in meta_options:  #     #     self.BacSeg.upload_microscope.setCurrentText(meta_options["microscope"])  #     # if "antibiotic" in meta_options:  #     #     self.BacSeg.upload_antibiotic.setCurrentText(meta_options["antibiotic"])  #     # if "treatment time (mins)" in meta_options:  #     #     self.BacSeg.upload_treatment_time.setCurrentText(meta_options["treatment time (mins)"])  #     # if "antibiotic concentration" in meta_options:  #     #     self.BacSeg.upload_abxconcentration.setCurrentText(meta_options["antibiotic concentration"])  #     # if "mounting method" in meta_options:  #     #     self.BacSeg.upload_mount.setCurrentText(meta_options["mounting method"])  #     # if "protocol" in meta_options:  #     #     self.BacSeg.upload_protocol.setCurrentText(meta_options["protocol"])  #     #  #     # if "segmentation_curated" in meta_options:  #     #     if meta_options["segmentation_curated"] == True:  #     #         self.BacSeg.upload_segmentation_combo.setCurrentIndex(3)  #     #     else:  #     #         self.BacSeg.upload_segmentation_combo.setCurrentIndex(2)  #     # elif "segmented" in meta_options:  #     #     if meta_options["segmented"] == True:  #     #         self.BacSeg.upload_segmentation_combo.setCurrentIndex(2)  #     #     else:  #     #         self.BacSeg.upload_segmentation_combo.setCurrentIndex(1)  #     # else:  #     #     self.BacSeg.upload_segmentation_combo.setCurrentIndex(0)  #     #  #     # if "label_curated" in meta_options:  #     #     if meta_options["label_curated"] == True:  #     #         self.BacSeg.upload_label_combo.setCurrentIndex(3)  #     #     else:  #     #         self.BacSeg.upload_label_combo.setCurrentIndex(2)  #     # elif "labelled" in meta_options:  #     #     if meta_options["labelled"] == True:  #     #         self.BacSeg.upload_label_combo.setCurrentIndex(2)  #     #     else:  #     #         self.BacSeg.upload_label_combo.setCurrentIndex(1)  #     # else:  #     #     self.BacSeg.upload_label_combo.setCurrentIndex(0)

    #     from napari_bacseg._utils_database_IO import (get_filtered_database_metadata, read_bacseg_images, )

    #     self.BacSeg.get_filtered_database_metadata = self.BacSeg.wrapper(get_filtered_database_metadata)  #     self.BacSeg.read_bacseg_images = self.BacSeg.wrapper(read_bacseg_images)

    #     self.BacSeg.active_import_mode = "BacSeg"

    #     (measurements, file_paths, channels,) = self.BacSeg.get_filtered_database_metadata()

    #     data = self.BacSeg.read_bacseg_images(measurements=measurements, channels=channels, progress_callback=None, )  #     self.BacSeg._process_import(data)

    #     layer_names = [layer.name for layer in self.BacSeg.viewer.layers if layer.name not in ["Segmentations", "Nucleoid", "Classes", "center_lines"]]

    #     loaded_images = len(self.BacSeg.viewer.layers[layer_names[0]].data)

    #     assert loaded_images == import_limit

    # def test_import(self):  #     self.perumuation_test(self.import_images, setting_dict=self.import_images_setting_dict)  # def test_export(self):  #     self.perumuation_test(self.export_data, setting_dict=self.export_setting_dict)


if __name__ == "__main__":
    unittest.main()
