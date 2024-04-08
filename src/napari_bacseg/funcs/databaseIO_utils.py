import datetime
import json
import os
import shutil
import tempfile
import traceback
from csv import Sniffer
from multiprocessing import Pool
import cv2
import numpy as np
import pandas as pd
import tifffile
from napari.utils.notifications import show_info
from ast import literal_eval
from napari_bacseg.funcs.IO.json_utils import import_coco_json


class _databaseIO:
    def check_metadata_format(self, metadata, expected_columns):
        
        missing_columns = list(set(expected_columns) - set(metadata.columns.tolist()))
        extra_columns = list(set(metadata.columns.tolist()) - set(expected_columns))
    
        all_columns = list(set(expected_columns + extra_columns))
    
        metadata[missing_columns] = pd.DataFrame([[None] * len(missing_columns)], index=metadata.index)
    
        date = datetime.datetime.now()
    
        metadata.loc[metadata["date_uploaded"].isin(["None", None, np.nan, 0]), ["date_uploaded", "date_created", "date_modified"],] = str(date)
    
        metadata = metadata[all_columns]
    
        metadata = metadata.astype({"segmented": bool, "labelled": bool, "segmentation_curated": bool, "label_curated": bool, })
    
        return metadata, all_columns

    @staticmethod
    def get_meta_value(meta, value):

        if value in meta.keys():
            data = meta[value]
        else:
            data = None

        return data

    @staticmethod
    def list_from_string(string):
        
        string_list = string.strip("[]")
        string_list = string_list.split(",")
        string_list = [x.strip() for x in string_list]
    
        return string_list
    
    def read_bacseg_images(self, progress_callback, measurements, channels):
    
        database_path = self.database_path
    
        imported_images = {}
        iter = 1
    
        import_limit = self.database_download_limit.currentText()
    
        if import_limit == "All":
            import_limit = len(measurements)
        else:
            if int(import_limit) > len(measurements):
                import_limit = len(measurements)
    
        show_info(f"Downloading {len(measurements)} files.")
    
        if import_limit == str(1):
            try:
                measurement = measurements.get_group(list(measurements.groups)[0])
                imported_data = [self.download_bacseg_files(measurement, channels, database_path)]
                progress_callback.emit(100)
            except:
                pass
        else:
            measurements = [measurements.get_group(i) for i in measurements.groups][: int(import_limit)]
            with Pool(4) as pool:
                iter = []
    
                def callback(*args):
                    iter.append(1)
                    progress = (len(iter) / int(import_limit)) * 100
    
                    if progress_callback != None:
                        try:
                            progress_callback.emit(progress)
                        except:
                            pass
    
                    return
    
                results = [pool.apply_async(self.download_bacseg_files,
                    args=(measurement, channels, database_path), callback=callback, ) for measurement in measurements]
    
                try:
                    results[-1].get()
                except:
                    print(traceback.format_exc())
                else:
                    results = [r.get() for r in results]
                    imported_data = [dat for dat in results if dat != {}]
                    pool.close()
                    pool.join()
    
        try:
            imported_images = {}
    
            for dat in imported_data:
                for channel, channel_data in dat.items():
    
                    image = channel_data["images"]
                    mask = channel_data["masks"]
                    nmask = channel_data["nmasks"]
                    label = channel_data["classes"]
                    meta = channel_data["metadata"]
    
                    if channel not in imported_images.keys():
                        imported_images[channel] = dict(images=[image], masks=[mask], nmasks=[nmask], classes=[label], metadata={0: meta}, )
                    else:
                        image_index = len(imported_images[channel]["images"])
    
                        imported_images[channel]["images"].append(image)
                        imported_images[channel]["masks"].append(mask)
                        imported_images[channel]["nmasks"].append(nmask)
                        imported_images[channel]["classes"].append(label)
                        imported_images[channel]["metadata"][image_index] = meta
    
        except:
            print(traceback.format_exc())
    
        imported_data = dict(imported_images=imported_images)
    
        return imported_data
    
    def generate_multichannel_stack(self):
    
        segChannel = self.cellpose_segchannel.currentText()
        user_initial = self.upload_initial.currentText()
        content = self.upload_content.currentText()
        microscope = self.upload_microscope.currentText()
        antibiotic = self.upload_antibiotic.currentText()
        abxconcentration = self.upload_abxconcentration.currentText()
        treatmenttime = self.upload_treatmenttime.currentText()
        mount = self.upload_mount.currentText()
        protocol = self.upload_protocol.currentText()
        overwrite_all_metadata = self.overwrite_all_metadata.isChecked()
        overwrite_selected_metadata = self.overwrite_selected_metadata.isChecked()
        date_uploaded = str(datetime.datetime.now())
        strain = self.upload_strain.currentText()
        phenotype = self.upload_phenotype.currentText()
    
        segmented = False
        segmented_curated = False
        labelled = False
        labelled_curated = False
    
        if self.upload_segmentation_combo.currentIndex() == 2:
            segmented = True
            segmented_curated = False
        if self.upload_segmentation_combo.currentIndex() == 3:
            segmented = True
            segmented_curated = True
        if self.upload_label_combo.currentIndex() == 2:
            labelled = True
            labelled_curated = False
        if self.upload_label_combo.currentIndex() == 3:
            labelled = True
            labelled_curated = True
    
        metadata = dict(user_initial=user_initial, content=content,
            microscope=microscope, antibiotic=antibiotic,
            abxconcentration=abxconcentration,
            treatmenttime=treatmenttime,
            mount=mount, protocol=protocol,
            strain=strain, phenotype=phenotype, )
    
        num_user_keys = self.user_metadata_keys
    
        for key in range(1, num_user_keys + 1):
            control_name = f"upload_usermeta{key}"
            combo_box = getattr(self, control_name)
            combo_box_value = combo_box.currentText()
            metadata[f"user_meta{key}"] = combo_box_value
    
        layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Nucleoid", "Classes", "center_lines", "Localisations"]]
    
        layer_names.reverse()
    
        # put segmentation channel as first channel in stack
        segChannel = self.cellpose_segchannel.currentText()
        layer_names.remove(segChannel)
        layer_names.insert(0, segChannel)
    
        dim_range = int(self.viewer.dims.range[0][1])
    
        multi_image_stack = []
        multi_meta_stack = {}
    
        for i in range(dim_range):
            rgb_images = []
            rgb_meta = {}
            file_list = []
            layer_list = []
    
            try:
                for j in range(len(layer_names)):
                    segmentation_file = self.viewer.layers[segChannel].metadata[i]["image_name"]
    
                    layer = str(layer_names[j])
    
                    img = self.viewer.layers[layer].data[i]
                    meta = self.viewer.layers[layer].metadata[i]
    
                    if meta["image_name"] != "missing image channel":
                        file_list.append(meta["image_name"])
                        layer_list.append(layer)
    
                        meta["user_initial"] = user_initial
    
                        if (meta["import_mode"] != "BacSeg" or overwrite_all_metadata is True):
                            meta["microscope"] = microscope
                            meta["image_content"] = content
                            meta["antibiotic"] = antibiotic
                            meta["strain"] = strain
                            meta["phenotype"] = phenotype
                            meta["treatmenttime"] = treatmenttime
                            meta["abxconcentration"] = abxconcentration
                            meta["mount"] = mount
                            meta["protocol"] = protocol
                            meta["channel"] = layer
                            meta["segmentation_channel"] = segChannel
                            meta["file_list"] = []
                            meta["layer_list"] = []
                            meta["segmentation_file"] = segmentation_file
    
                            for key in range(1, num_user_keys + 1):
                                if f"user_meta{key}" in metadata.keys():
                                    meta[f"user_meta{key}"] = metadata[f"user_meta{key}"]
                                else:
                                    meta[f"user_meta{key}"] = ""
    
                        if (meta["import_mode"] == "BacSeg" and overwrite_all_metadata is True):
                            metadata = {key: val for key, val in metadata.items() if val != "Required for upload"}
    
                            for key, value in metadata.items():
                                meta[key] = value
    
                        if overwrite_selected_metadata is True:
                            metadata = {key: val for key, val in metadata.items() if val not in ["", "Required for upload"]}
    
                            for key, value in metadata.items():
                                meta[key] = value
    
                        meta["segmented"] = segmented
                        meta["labelled"] = labelled
                        meta["segmentations_curated"] = segmented_curated
                        meta["labels_curated"] = labelled_curated
    
                        if self.cellpose_segmentation == True:
                            meta["cellpose_segmentation"] = self.cellpose_segmentation
                            meta["flow_threshold"] = float(self.cellpose_flowthresh_label.text())
                            meta["mask_threshold"] = float(self.cellpose_maskthresh_label.text())
                            meta["min_size"] = int(self.cellpose_minsize_label.text())
                            meta["diameter"] = int(self.cellpose_diameter_label.text())
                            meta["cellpose_model"] = self.cellpose_segmodel.currentText()
                            meta["custom_model"] = os.path.abspath(self.cellpose_custom_model_path)
    
                        rgb_images.append(img)
                        rgb_meta[layer] = meta
    
                for layer in layer_names:
                    if layer in rgb_meta.keys():
                        rgb_meta[layer]["file_list"] = file_list
                        rgb_meta[layer]["channel_list"] = layer_list
                        rgb_meta["channel_list"] = layer_list
    
                rgb_images = np.stack(rgb_images, axis=0)
    
                multi_image_stack.append(rgb_images)
                multi_meta_stack[i] = rgb_meta
    
            except:
                print(traceback.format_exc())
    
        return multi_image_stack, multi_meta_stack, layer_names

    @staticmethod
    def upload_bacseg_files(path, widget_notifications=True, num_user_keys=6):
        file_metadata_list = []

        try:
            dat = np.load(path, allow_pickle=True).item()

            user_metadata = dat["user_metadata"]
            image = dat["image"]
            image_meta = dat["image_meta"]
            mask = dat["mask"]
            nmask = dat["nmask"]
            class_mask = dat["class_mask"]
            save_dir = dat["save_dir"]
            overwrite_images = dat["overwrite_images"]
            overwrite_masks = dat["overwrite_masks"]
            overwrite_metadata = dat["overwrite_metadata"]
            overwrite_selected_metadata = dat["overwrite_selected_metadata"]
            overwrite_all_metadata = dat["overwrite_all_metadata"]
            upload_images = dat["upload_images"]
            upload_segmentations = dat["upload_segmentations"]
            upload_metadata = dat["upload_metadata"]
            image_dir = dat["image_dir"]
            mask_dir = dat["mask_dir"]
            class_dir = dat["class_dir"]
            json_dir = dat["json_dir"]

            metadata_file_names = user_metadata["file_name"].tolist()
            metadata_akseg_hash = user_metadata["akseg_hash"].tolist()

            channel_list = image_meta["channel_list"]

            file_metadata_list = []

            for j, layer in enumerate(channel_list):
                img = image[j, :, :]
                meta = image_meta[layer]

                file_name = _databaseIO.get_meta_value(meta, "image_name")
                folder = _databaseIO.get_meta_value(meta, "folder")
                akseg_hash = _databaseIO.get_meta_value(meta, "akseg_hash")

                import_mode = meta["import_mode"]

                if "posX" in meta.keys():
                    posX = meta["posX"]
                    posY = meta["posX"]
                    posZ = meta["posX"]
                elif "StagePos_um" in meta.keys():
                    posX, posY, posZ = meta["StagePos_um"]
                else:
                    posX, posY, posZ = 0, 0, 0

                if file_name in metadata_file_names:
                    try:
                        date_uploaded = user_metadata[(user_metadata["file_name"] == file_name) & (user_metadata["folder"] == folder)]["date_uploaded"].item()
                    except:
                        date_uploaded = str(datetime.datetime.now())
                else:
                    date_uploaded = str(datetime.datetime.now())

                if "date_created" in meta.keys():
                    date_created = meta["date_created"]
                elif file_name in metadata_file_names:
                    try:
                        date_created = user_metadata[(user_metadata["file_name"] == file_name) & (user_metadata["folder"] == folder)]["date_created"].item()
                    except:
                        date_created = str(datetime.datetime.now())
                else:
                    date_created = str(datetime.datetime.now())

                date_modified = str(datetime.datetime.now())

                # stops user from overwriting BacSeg files, unless they have opened them from BacSeg for curation
                if (akseg_hash in metadata_akseg_hash and import_mode != "BacSeg" and overwrite_images == False and overwrite_masks == False and overwrite_metadata is False):
                    if widget_notifications:
                        show_info("file already exists  in BacSeg Database:   " + file_name)

                    file_metadata = None

                else:
                    if import_mode == "BacSeg":
                        if overwrite_selected_metadata is True:
                            if widget_notifications:
                                show_info("Overwriting selected metadata on BacSeg Database:   " + file_name)

                        elif overwrite_all_metadata is True:
                            if widget_notifications:
                                show_info("Overwriting all metadata on BacSeg Database:   " + file_name)

                        else:
                            if widget_notifications:
                                show_info("Editing file on BacSeg Database:   " + file_name)

                    elif overwrite_images is True and overwrite_masks is True:
                        if widget_notifications:
                            show_info("Overwriting image + mask/label on BacSeg Database:   " + file_name)

                    elif overwrite_images is True:
                        if widget_notifications:
                            show_info("Overwriting image on BacSeg Database:   " + file_name)

                    elif overwrite_masks is True:
                        if widget_notifications:
                            show_info("Overwriting mask/label on BacSeg Database:   " + file_name)

                    else:
                        if widget_notifications:
                            show_info("Uploading file to BacSeg Database:   " + file_name)

                    y1, y2, x1, x2 = meta["crop"]

                    if len(img.shape) > 2:
                        img = img[:, y1:y2, x1:x2]
                    else:
                        img = img[y1:y2, x1:x2]

                    mask = mask[y1:y2, x1:x2]
                    nmask = nmask[y1:y2, x1:x2]
                    class_mask = class_mask[y1:y2, x1:x2]

                    unique_segmentations = np.unique(mask)
                    unique_segmentations = np.delete(unique_segmentations, np.where(unique_segmentations == 0))

                    num_segmentations = len(unique_segmentations)
                    image_laplacian = cv2.Laplacian(img, cv2.CV_64F).var()

                    meta.pop("shape", None)

                    file_name = os.path.splitext(meta["image_name"])[0] + ".tif"

                    image_path = os.path.join(image_dir, file_name)
                    mask_path = os.path.join(mask_dir, file_name)
                    json_path = os.path.join(json_dir, file_name.replace(".tif", ".txt"))
                    class_path = os.path.join(class_dir, file_name)

                    segmentation_file = _databaseIO.get_meta_value(meta, "segmentation_file")

                    if segmentation_file is None:
                        segmentation_channel = channel_list[0]
                        segmentation_file = image_meta[segmentation_channel]["image_name"]
                        meta["segmentation_file"] = segmentation_file
                        meta["segmentation_channel"] = segmentation_channel

                    if upload_images is True:
                        if (os.path.isfile(image_path) is False or import_mode == "BacSeg" or overwrite_images is True or overwrite_metadata is True):
                            tifffile.imwrite(os.path.abspath(image_path), img, metadata=meta)

                    if upload_segmentations is True:
                        if (os.path.isfile(mask_path) is False or import_mode == "BacSeg" or overwrite_masks is True or overwrite_metadata is True):
                            if file_name == segmentation_file:

                                from napari_bacseg.funcs.IO.json_utils import export_coco_json

                                export_coco_json(file_name, img, mask, nmask, class_mask, json_path, )

                    if "mask_path" not in meta.keys():
                        meta["mask_path"] = None
                    if "label_path" not in meta.keys():
                        meta["label_path"] = None

                    file_metadata = {"date_uploaded": date_uploaded,
                                     "date_created": date_created,
                                     "date_modified": date_modified,
                                     "file_name": file_name,
                                     "channel": _databaseIO.get_meta_value(meta, "channel"),
                                     "file_list": _databaseIO.get_meta_value(meta, "file_list"),
                                     "channel_list": _databaseIO.get_meta_value(meta, "channel_list"),
                                     "segmentation_file": _databaseIO.get_meta_value(meta, "segmentation_file"),
                                     "segmentation_channel": _databaseIO.get_meta_value(meta, "segmentation_channel"),
                                     "strain": _databaseIO.get_meta_value(meta, "strain"),
                                     "phenotype": _databaseIO.get_meta_value(meta, "phenotype"),
                                     "akseg_hash": _databaseIO.get_meta_value(meta, "akseg_hash"),
                                     "user_initial": _databaseIO.get_meta_value(meta, "user_initial"),
                                     "content": _databaseIO.get_meta_value(meta, "image_content"),
                                     "microscope": _databaseIO.get_meta_value(meta, "microscope"),
                                     "modality": _databaseIO.get_meta_value(meta, "modality"),
                                     "source": _databaseIO.get_meta_value(meta, "light_source"),
                                     "stain": _databaseIO.get_meta_value(meta, "stain"),
                                     "stain_target": _databaseIO.get_meta_value(meta, "stain_target"),
                                     "antibiotic": _databaseIO.get_meta_value(meta, "antibiotic"),
                                     "treatment time (mins)": _databaseIO.get_meta_value(meta, "treatmenttime"),
                                     "antibiotic concentration": _databaseIO.get_meta_value(meta, "abxconcentration"),
                                     "mounting method": _databaseIO.get_meta_value(meta, "mount"),
                                     "protocol": _databaseIO.get_meta_value(meta, "protocol"),
                                     "folder": _databaseIO.get_meta_value(meta, "folder"),
                                     "parent_folder": _databaseIO.get_meta_value(meta, "parent_folder"),
                                     "num_segmentations": num_segmentations, "image_laplacian": image_laplacian,
                                     "image_focus": _databaseIO.get_meta_value(meta, "image_focus"),
                                     "image_debris": _databaseIO.get_meta_value(meta, "image_debris"),
                                     "segmented": _databaseIO.get_meta_value(meta, "segmented"),
                                     "labelled": _databaseIO.get_meta_value(meta, "labelled"),
                                     "segmentation_curated": _databaseIO.get_meta_value(meta, "segmentations_curated"),
                                     "label_curated": _databaseIO.get_meta_value(meta, "labels_curated"),
                                     "posX": posX, "posY": posY, "posZ": posZ,
                                     "image_load_path": _databaseIO.get_meta_value(meta, "image_path"),
                                     "image_save_path": image_path,
                                     "mask_load_path": _databaseIO.get_meta_value(meta, "mask_path"),
                                     "mask_save_path": mask_path,
                                     "label_load_path": _databaseIO.get_meta_value(meta, "label_path"),
                                     "label_save_path": class_path, }

                    for key in range(1, num_user_keys + 1):
                        file_metadata[f"user_meta{key}"] = _databaseIO.get_meta_value(meta, f"user_meta{key}")

                    file_metadata_list.append(file_metadata)

        except:
            file_metadata_list = []
            print(traceback.format_exc())

        return file_metadata_list
    
    @staticmethod
    def download_bacseg_files(measurement, channels, database_path):
        imported_images = {}
    
        try:
            segmentation_channel = measurement["segmentation_channel"].unique()[0]
    
            if segmentation_channel not in ["", " ", None, np.nan]:
    
                segmentation_file = measurement[measurement["channel"] == segmentation_channel]["file_name"].item()
                user_initial = measurement[measurement["channel"] == segmentation_channel]["user_initial"].item()
                folder = measurement[measurement["channel"] == segmentation_channel]["folder"].item()
    
                json_path = os.path.join(database_path, "Images", user_initial, "json", folder, segmentation_file, )
    
                mask, nmask, label = import_coco_json(json_path)
    
                for j in range(len(channels)):
                    channel = channels[j]
    
                    measurement_channels = measurement["channel"].unique()
    
                    if channel in measurement_channels:
                        dat = measurement[measurement["channel"] == channel]
    
                        file_name = dat["file_name"].item()
                        user_initial = dat["user_initial"].item()
                        folder = dat["folder"].item()
    
                        image_path = os.path.join(database_path, "Images", user_initial, "images", folder, file_name, )
                        image_path = os.path.abspath(image_path)
    
                        image = tifffile.imread(image_path)
    
                        with tifffile.TiffFile(image_path) as tif:
                            try:
                                meta = tif.pages[0].tags["ImageDescription"].value
                                meta = json.loads(meta)
                            except:
                                meta = {}
    
                        meta["import_mode"] = "BacSeg"
    
                        for key, value in dat.to_dict("records")[0].items():
                            meta[key] = value
    
                        if "segmentation_file" in meta.keys():
                            if meta["segmentation_file"] in [None, "None"]:
                                meta["segmentation_file"] = _databaseIO.list_from_string(measurement["file_list"].iloc[0])[0]
                                meta["segmentation_channel"] = _databaseIO.list_from_string(measurement["channel_list"].iloc[0])[0]
    
                    else:
                        image = np.zeros((100, 100), dtype=np.uint16)
                        mask = np.zeros((100, 100), dtype=np.uint16)
                        nmask = np.zeros((100, 100), dtype=np.uint16)
                        label = np.zeros((100, 100), dtype=np.uint16)
    
                        meta = {}
    
                        meta["image_name"] = "missing image channel"
                        meta["image_path"] = "missing image channel"
                        meta["folder"] = (None,)
                        meta["parent_folder"] = (None,)
                        meta["akseg_hash"] = None
                        meta["fov_mode"] = None
                        meta["import_mode"] = "BacSeg"
                        meta["contrast_limit"] = None
                        meta["contrast_alpha"] = None
                        meta["contrast_beta"] = None
                        meta["contrast_gamma"] = None
                        meta["dims"] = [image.shape[-1], image.shape[-2]]
                        meta["crop"] = [0, image.shape[-2], 0, image.shape[-1]]
                        meta["light_source"] = channel
    
                    imported_images[channel] = dict(images=image, masks=mask, nmasks=nmask, classes=label, metadata=meta, )
    
        except:
            print(traceback.format_exc())
    
        return imported_images

    def generate_upload_tempfiles(self, user_metadata, image_stack, meta_stack, mask_stack,
            nmask_stack, class_stack, save_dir, overwrite_images, overwrite_masks,
            overwrite_metadata, overwrite_selected_metadata, overwrite_all_metadata,
            upload_images, upload_segmentations, upload_metadata, ):
    
        upload_tempfiles = []
    
        upload_dir = os.path.join(tempfile.gettempdir(), "BacSeg")
    
        if os.path.isdir(upload_dir) != True:
            os.mkdir(upload_dir)
        else:
            shutil.rmtree(upload_dir)
            os.mkdir(upload_dir)
    
        for i in range(len(image_stack)):
            try:
                image = image_stack[i]
                image_meta = meta_stack[i]
                mask = mask_stack[i]
                nmask = nmask_stack[i]
                class_mask = class_stack[i]
    
                meta = image_meta[image_meta["channel_list"][0]]
    
                folder = meta["folder"]
    
                image_dir = os.path.join(save_dir, "images", folder)
                mask_dir = os.path.join(save_dir, "masks", folder)
                class_dir = os.path.join(save_dir, "labels", folder)
                json_dir = os.path.join(save_dir, "json", folder)
    
                if os.path.exists(image_dir) == False:
                    os.makedirs(image_dir)
    
                if os.path.exists(mask_dir) == False:
                    os.makedirs(mask_dir)
    
                if os.path.exists(json_dir) == False:
                    os.makedirs(json_dir)
    
                if os.path.exists(class_dir) == False:
                    os.makedirs(class_dir)
    
                upload_data = dict(user_metadata=user_metadata, image=image,
                    image_meta=image_meta, mask=mask, nmask=nmask, class_mask=class_mask,
                    save_dir=save_dir, overwrite_images=overwrite_images, overwrite_masks=overwrite_masks,
                    overwrite_metadata=overwrite_metadata, overwrite_selected_metadata=overwrite_selected_metadata,
                    overwrite_all_metadata=overwrite_all_metadata, image_dir=image_dir, mask_dir=mask_dir,
                    json_dir=json_dir, class_dir=class_dir, upload_images=upload_images,
                    upload_segmentations=upload_segmentations, upload_metadata=upload_metadata, )
    
                if os.path.isdir(upload_dir) is False:
                    os.mkdir(upload_dir)
    
                temp_path = tempfile.TemporaryFile(prefix="BacSeg", suffix=".npy", dir=upload_dir).name
    
                np.save(temp_path, upload_data)
    
                upload_tempfiles.append(temp_path)
    
            except:
                print(traceback.format_exc())
    
        return upload_tempfiles
    
    
    def _upload_bacseg_database(self, progress_callback, mode):
        try:
            database_path = self.database_path
    
            if os.path.exists(database_path) == False:
                if self.widget_notifications:
                    show_info("Could not find BacSeg Database")
    
            else:
                user_initial = self.upload_initial.currentText()
                content = self.upload_content.currentText()
                microscope = self.upload_microscope.currentText()
                modalilty = self.img_modality.currentText()
                source = self.img_light_source.currentText()
                stain = self.img_stain.currentText()
                stain_target = self.img_stain_target.currentText()
                date_modified = str(datetime.datetime.now())
                overwrite_images = self.upload_overwrite_images.isChecked()
                overwrite_masks = self.upload_overwrite_masks.isChecked()
                overwrite_all_metadata = self.overwrite_all_metadata.isChecked()
                overwrite_selected_metadata = (self.overwrite_selected_metadata.isChecked())
                upload_images = self.upload_images_setting.isChecked()
                upload_segmentations = (self.upload_segmentations_setting.isChecked())
                upload_metadata = self.upload_metadata_setting.isChecked()
                strain = self.upload_strain.currentText()
                phenotype = self.upload_phenotype.currentText()
    
                num_user_keys = self.user_metadata_keys
    
                save_dir = os.path.join(database_path, "Images", user_initial)
    
                if (overwrite_all_metadata is True or overwrite_selected_metadata is True):
                    overwrite_metadata = True
                else:
                    overwrite_metadata = False
    
                user_metadata_path = os.path.join(database_path, "Images", user_initial, f"{user_initial}_file_metadata.txt", )
    
                if os.path.exists(user_metadata_path):
                    user_metadata, expected_columns = self.read_user_metadata(user_metadata_path=user_metadata_path)
    
                    metadata_file_names = user_metadata["file_name"].tolist()
                    metadata_akseg_hash = user_metadata["akseg_hash"].tolist()
    
                else:
                    expected_columns = self.metadata_columns
    
                    metadata_file_names = []
                    metadata_akseg_hash = []
    
                    user_metadata = pd.DataFrame(columns=self.metadata_columns)
    
                channel_labels = ["modality", "light_source", "stain", "stain_target", ]
                channel_metadata = [layer.metadata[0] for layer in self.viewer.layers if layer.name not in ["Segmentations", "Nucleoid", "Classes", "center_lines", "Localisations"]]
    
                metalabels = []
    
                missing_metalabel_layers = []
                for layer in self.viewer.layers:
                    if layer.name not in ["Segmentations", "Nucleoid", "Classes", "center_lines", ]:
                        layer_metadata = layer.metadata[0]
                        for label in channel_labels:
                            if label in layer_metadata.keys():
                                metalabels.append(layer.metadata[0][label])
                            else:
                                metalabels.append("Required for upload")
                                missing_metalabel_layers.append(layer.name)
    
                missing_metalabel_layers = list(set(missing_metalabel_layers))
    
                metalabels = metalabels + [user_initial, content, microscope]
    
                if ("Required for upload" in metalabels and self.active_import_mode != "BacSeg"):
                    if self.widget_notifications:
                        show_info("Please fill out channel (all channels) and image metadata before uploading files")
                        show_info(f"Missing metadata for {missing_metalabel_layers}")
    
                else:
                    segChannel = self.cellpose_segchannel.currentText()
                    channel_list = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Nucleoid", "Classes", "center_lines", "Localisations"]]
    
                    if segChannel == "":
                        if self.widget_notifications:
                            show_info("Please pick an image channel to upload")
    
                    else:
    
                        (image_stack, meta_stack, channel_list,) = self.generate_multichannel_stack()
                        mask_stack = self.segLayer.data
                        nmask_stack = self.nucLayer.data
                        class_stack = self.classLayer.data
    
                        if len(image_stack) >= 1:
                            if mode == "active":
                                current_step = self.viewer.dims.current_step[0]
    
                                image_stack = np.expand_dims(image_stack[current_step], axis=0)
                                mask_stack = np.expand_dims(mask_stack[current_step], axis=0)
                                nmask_stack = np.expand_dims(nmask_stack[current_step], axis=0)
                                class_stack = np.expand_dims(class_stack[current_step], axis=0)
                                meta_stack = np.expand_dims(meta_stack[current_step], axis=0)
    
                        upload_tempfiles = self.generate_upload_tempfiles(user_metadata, image_stack, meta_stack,
                            mask_stack, nmask_stack, class_stack, save_dir, overwrite_images, overwrite_masks,
                            overwrite_metadata, overwrite_selected_metadata, overwrite_all_metadata,
                            upload_images, upload_segmentations, upload_metadata, )
    
                        if upload_tempfiles != []:
                            if mode == "active":
                                results = self.upload_bacseg_files(upload_tempfiles[0], num_user_keys)
    
                            else:
                                with Pool(4) as pool:
                                    iter = []
    
                                    def callback(*args):
                                        iter.append(1)
                                        progress = (len(iter) / len(upload_tempfiles)) * 100
    
                                        if progress_callback != None:
                                            try:
                                                progress_callback.emit(progress)
                                            except:
                                                pass
    
                                        return
    
                                    results = [pool.apply_async(self.upload_bacseg_files, args=(i, num_user_keys),
                                        callback=callback, ) for i in upload_tempfiles]
    
                                    try:
                                        results[-1].get()
                                    except:
                                        print(traceback.format_exc())
                                    else:
                                        results = [r.get() for r in results]
                                        results = [file_metadata for file_metadata_list in results for file_metadata in file_metadata_list if file_metadata != None]
                                        pool.close()
                                        pool.join()
    
                                    results = [dat for dat in results if results != None]
    
                            if upload_metadata is True:
                                for file_metadata in results:
                                    if file_metadata != None:
                                        akseg_hash = file_metadata["akseg_hash"]
    
                                        file_metadata = pd.DataFrame.from_dict(file_metadata, dtype=object)
    
                                        columns = file_metadata.columns.tolist()
                                        column_dict = {col: "first" for col in columns if col not in ["file_list", "channel_list"]}
    
                                        df = (file_metadata.groupby(["file_name"]).agg({**{"file_list": lambda x: x.tolist(), "channel_list": lambda x: x.tolist(), },
                                            **column_dict, })).reset_index(drop=True)
    
                                        file_metadata = df[columns]
    
                                        file_metadata = file_metadata.astype({"segmented": bool, "labelled": bool, "segmentation_curated": bool, "label_curated": bool, })
    
                                        user_metadata = user_metadata.astype({"segmented": bool, "labelled": bool, "segmentation_curated": bool, "label_curated": bool, })
    
                                        user_metadata.reset_index(drop=True, inplace=True)
                                        file_metadata.reset_index(drop=True, inplace=True)
    
                                        user_metadata = user_metadata.loc[:, ~user_metadata.columns.duplicated()].copy()
                                        file_metadata = file_metadata.loc[:, ~file_metadata.columns.duplicated()].copy()
    
                                        for (column) in user_metadata.columns.tolist():
                                            if (column not in file_metadata.columns.tolist()):
                                                file_metadata[column] = ""
    
                                        if akseg_hash in metadata_akseg_hash:
                                            user_metadata = pd.concat((user_metadata, file_metadata),
                                                ignore_index=True, axis=0, )  # .reset_index(drop=True)
    
                                            user_metadata.drop_duplicates(subset=["segmentation_file"], keep="last", inplace=True, )
    
                                        else:
                                            user_metadata = pd.concat((user_metadata, file_metadata),
                                                ignore_index=True, axis=0, ).reset_index(drop=True)
    
                            if upload_metadata is True:
    
                                user_metadata.drop_duplicates(subset=["segmentation_file"], keep="last", inplace=True, )
                                user_metadata = user_metadata[expected_columns]
    
                                self.user_metadata = user_metadata.copy()
    
                                user_metadata = user_metadata.astype("str")
                                user_metadata.to_csv(user_metadata_path, sep=",", index=False)
    
                show_info("Upload complete")
    
        except:
            print(traceback.format_exc())
    
    
    def read_user_metadata(self, user_metadata_path="", update=False,):
    
        try:
            if user_metadata_path == "":
                database_path = self.database_path
                user_initial = self.upload_initial.currentText()
    
                user_metadata_path = os.path.join(database_path, "Images", user_initial, f"{user_initial}_file_metadata.txt", )
    
            load_file = False
            if self.user_metadata_path != user_metadata_path or update == True or len(self.user_metadata) == 0:
                self.user_metadata_path = user_metadata_path
                load_file = True
    
            if os.path.isfile(self.user_metadata_path) == False:
                if self.widget_notifications:
                    show_info("Could not find metadata for user: " + user_initial)
    
                user_metadata = None
                expected_columns = None
    
            else:
    
                if load_file:
                    show_info("Loading metadata file: " + self.user_metadata_path)
    
                    sniffer = Sniffer()
    
                    # checks the delimiter of the metadata file
                    with open(self.user_metadata_path) as f:
                        line = next(f).strip()
                        delim = sniffer.sniff(line)
    
                    user_metadata = pd.read_csv(self.user_metadata_path, sep=delim.delimiter, low_memory=False)
                    user_metadata = user_metadata.astype("str")
    
                    user_metadata = user_metadata.drop_duplicates(subset="segmentation_file")
    
                    user_metadata["segmented"] = user_metadata["segmented"].apply(literal_eval)
                    user_metadata["labelled"] = user_metadata["labelled"].apply(literal_eval)
                    user_metadata["segmentation_curated"] = user_metadata["segmentation_curated"].apply(literal_eval)
                    user_metadata["label_curated"] = user_metadata["label_curated"].apply(literal_eval)
    
                    user_metadata[["treatment time (mins)"]] = user_metadata[["treatment time (mins)"]].apply(pd.to_numeric, downcast="float", errors="coerce")
    
                    user_metadata, expected_columns = self.check_metadata_format(user_metadata, self.metadata_columns)
    
                    user_metadata["segmentation_channel"] = user_metadata["segmentation_channel"].astype(str)
    
                    user_metadata['file_list'] = user_metadata['file_list'].apply(literal_eval)
                    user_metadata['channel_list'] = user_metadata['channel_list'].apply(literal_eval)
    
                    user_metadata["file_name"] = user_metadata["file_list"]
                    user_metadata["channel"] = user_metadata["channel_list"]
    
                    user_metadata = user_metadata.explode(['file_name', 'channel'])
    
                    self.user_metadata = user_metadata
                    self.expected_columns = expected_columns
    
                else:
                    user_metadata = self.user_metadata.copy()
                    expected_columns = self.expected_columns.copy()
    
                    user_metadata = user_metadata.drop_duplicates(subset="segmentation_file")
    
                    user_metadata["file_name"] = user_metadata["file_list"]
                    user_metadata["channel"] = user_metadata["channel_list"]
    
                    user_metadata = user_metadata.explode(['file_name', 'channel'])
    
                    self.user_metadata = user_metadata
                    self.expected_columns = expected_columns
    
        except:
            print(traceback.format_exc())
            user_metadata = None
            expected_columns = None
    
        return user_metadata, expected_columns
    
    
    def backup_user_metadata(self, progress_callback=None, user_metadata=""):
        try:
            if type(user_metadata) is str:
                user_metadata, _ = self.read_user_metadata()
    
            database_path = self.database_path
    
            user_initial = self.upload_initial.currentText()
    
            todays_date = str(datetime.datetime.now().strftime("%y%m%d-%H%M"))
    
            user_metadata_path = os.path.join(database_path, "Images", user_initial, "file metadata backup", f"{user_initial}_file_metadata_{todays_date}.txt", )
    
            if os.path.exists(os.path.dirname(user_metadata_path)) == False:
                os.makedirs(os.path.dirname(user_metadata_path))
    
            user_metadata = user_metadata.astype("str")
    
            user_metadata.drop_duplicates(subset=["segmentation_file"], keep="first", inplace=True)
    
            show_info("Creating metadata backup for user: " + user_initial)
    
            user_metadata.to_csv(user_metadata_path, sep=",", index=False)
    
        except:
            print(traceback.format_exc())
    
    
    def get_filtered_database_metadata(self):
    
        try:
            database_metadata = {"user_initial": self.upload_initial.currentText(),
                                 "content": self.upload_content.currentText(),
                                 "microscope": self.upload_microscope.currentText(),
                                 "phenotype": self.upload_phenotype.currentText(),
                                 "strain": self.upload_strain.currentText(),
                                 "antibiotic": self.upload_antibiotic.currentText(),
                                 "antibiotic concentration": self.upload_abxconcentration.currentText(),
                                 "treatment time (mins)": self.upload_treatmenttime.currentText(),
                                 "mounting method": self.upload_mount.currentText(),
                                 "protocol": self.upload_protocol.currentText(), }
    
            num_user_keys = self.user_metadata_keys
    
            for key in range(1, num_user_keys + 1):
                control_name = f"upload_usermeta{key}"
                combo_box = getattr(self, control_name)
                combo_box_text = combo_box.currentText()
                database_metadata[f"user_meta{key}"] = combo_box_text
    
            database_metadata = {key: val for key, val in database_metadata.items() if val not in ["", "Required for upload", "example_item1", "example_item2", "example_item3", ]}
    
            database_path = self.database_path
    
            user_initial = database_metadata["user_initial"]
    
            user_metadata_path = os.path.join(database_path, "Images",
                user_initial, f"{user_initial}_file_metadata.txt", )
    
            if os.path.isfile(user_metadata_path) == False:
                if self.widget_notifications:
                    show_info("Could not find metadata for user: " + user_initial)
    
                measurements = []
                file_paths = []
                channels = []
    
            else:
                user_metadata, expected_columns = self.read_user_metadata()
    
                user_metadata["segmentation_channel"] = user_metadata["segmentation_channel"].astype(str)
                user_metadata["treatment time (mins)"] = user_metadata["treatment time (mins)"].astype(str)
    
                for key, value in database_metadata.items():
                    if key in user_metadata.columns:
                        user_metadata = user_metadata[user_metadata[key] == value]
    
                if self.upload_segmentation_combo.currentIndex() == 1:
                    user_metadata = user_metadata[user_metadata["segmented"] == False]
                if self.upload_segmentation_combo.currentIndex() == 2:
                    user_metadata = user_metadata[user_metadata["segmented"] == True & (user_metadata["segmentation_curated"] == False)]
                if self.upload_segmentation_combo.currentIndex() == 3:
                    user_metadata = user_metadata[(user_metadata["segmented"] == True) & (user_metadata["segmentation_curated"] == True)]
                if self.upload_label_combo.currentIndex() == 1:
                    user_metadata = user_metadata[user_metadata["labelled"] == False]
                if self.upload_label_combo.currentIndex() == 2:
                    user_metadata = user_metadata[user_metadata["labelled"] == True & (user_metadata["label_curated"] == False)]
                if self.upload_label_combo.currentIndex() == 3:
                    user_metadata = user_metadata[(user_metadata["labelled"] == True) & (user_metadata["label_curated"] == True)]
    
                user_metadata.sort_values(by=["posX", "posY", "posZ"], ascending=True)
    
                sort_names = []
                sort_directions = []
    
                if self.download_sort_order_1.currentIndex() > 1:
                    if self.download_sort_direction_1.currentIndex() == 1:
                        sort_directions.append(True)
                    if self.download_sort_direction_1.currentIndex() == 2:
                        sort_directions.append(False)
    
                if self.download_sort_order_2.currentIndex() > 1:
                    if self.download_sort_direction_2.currentIndex() == 1:
                        sort_directions.append(True)
                    if self.download_sort_direction_2.currentIndex() == 2:
                        sort_directions.append(False)
    
                if self.download_sort_order_1.currentIndex() == 1:
                    user_metadata = user_metadata.sample(frac=1).reset_index(drop=True)
                if self.download_sort_order_1.currentIndex() == 2:
                    sort_names.append("date_uploaded")
                if self.download_sort_order_1.currentIndex() == 3:
                    sort_names.append("date_modified")
                if self.download_sort_order_1.currentIndex() == 4:
                    if "image_laplacian" in user_metadata.columns:
                        sort_names.append("image_laplacian")
                if self.download_sort_order_1.currentIndex() == 5:
                    if "num_segmentations" in user_metadata.columns:
                        sort_names.append("num_segmentations")
                if self.download_sort_order_1.currentIndex() == 6:
                    if "image_focus" in user_metadata.columns:
                        sort_names.append("image_focus")
                if self.download_sort_order_1.currentIndex() == 7:
                    if "image_debris" in user_metadata.columns:
                        sort_names.append("image_debris")
    
                if self.download_sort_order_2.currentIndex() == 1:
                    user_metadata = user_metadata.sample(frac=1).reset_index(drop=True)
                if self.download_sort_order_2.currentIndex() == 2:
                    sort_names.append("date_uploaded")
                if self.download_sort_order_2.currentIndex() == 3:
                    sort_names.append("date_modified")
                if self.download_sort_order_2.currentIndex() == 4:
                    if "image_laplacian" in user_metadata.columns:
                        sort_names.append("image_laplacian")
                if self.download_sort_order_2.currentIndex() == 5:
                    if "num_segmentations" in user_metadata.columns:
                        sort_names.append("num_segmentations")
                if self.download_sort_order_2.currentIndex() == 6:
                    if "image_focus" in user_metadata.columns:
                        sort_names.append("image_focus")
                if self.download_sort_order_2.currentIndex() == 7:
                    if "image_debris" in user_metadata.columns:
                        sort_names.append("image_debris")
    
                if len(sort_names) > 0:
                    if len(sort_names) != len(sort_directions):
                        sort_directions = [False] * len(sort_names)
                    user_metadata = user_metadata.sort_values(sort_names,
                        ascending=sort_directions).reset_index(drop=True)
    
                sort_columns = []
    
                user_metadata = user_metadata.astype("str")
                user_metadata = user_metadata.drop_duplicates()
    
                if "folder" in user_metadata.columns:
                    sort_columns.append("folder")
                    user_metadata.loc[user_metadata["folder"].isna(), "folder"] = "None"
                if "segmentation_file" in user_metadata.columns:
                    sort_columns.append("segmentation_file")
                    user_metadata.loc[user_metadata["segmentation_file"].isna(), "segmentation_file",] = "None"
                if "posX" in user_metadata.columns:
                    sort_columns.append("posX")
                    user_metadata['posX'] = user_metadata['posX'].fillna(0, inplace=True)
                    user_metadata.loc[user_metadata["posX"].isna(), "posX"] = 0
                if "posY" in user_metadata.columns:
                    sort_columns.append("posY")
                    user_metadata['posY'] = user_metadata['posY'].fillna(0, inplace=True)
                    user_metadata.loc[user_metadata["posY"].isna(), "posY"] = 0
                if "posZ" in user_metadata.columns:
                    sort_columns.append("posZ")
                    user_metadata['posZ'] = user_metadata['posZ'].fillna(0, inplace=True)
                    user_metadata.loc[user_metadata["posZ"].isna(), "posZ"] = 0
    
                import_limit = self.database_download_limit.currentText()
    
                segmentation_files = user_metadata["segmentation_file"].unique()
                num_measurements = len(segmentation_files)
    
                if import_limit == "All":
                    import_limit = num_measurements
                else:
                    if int(import_limit) > num_measurements:
                        import_limit = num_measurements
    
                user_metadata = user_metadata[user_metadata["segmentation_file"].isin(segmentation_files[: int(import_limit)])]
    
                user_metadata["path"] = user_metadata["image_save_path"]
    
                channels = user_metadata["channel"].unique().tolist()
                file_paths = user_metadata["image_save_path"].tolist()
    
                len1 = len(user_metadata)
    
                user_metadata = user_metadata[~user_metadata["segmentation_file"].isin(["None", None, np.nan])]
                user_metadata = user_metadata[~user_metadata["folder"].isin(["None", None, np.nan])]
    
                len2 = len(user_metadata)
    
                if len2 < len1:
                    show_info(f"{len1 - len2} files with missing critical metadata removed.")
    
                measurements = user_metadata.groupby(sort_columns)
    
                show_info(f"Found {len(measurements)} matching database files.")
    
        except:
            print(traceback.format_exc())
            measurements, file_paths, channels = None, None, None
    
        return measurements, file_paths, channels