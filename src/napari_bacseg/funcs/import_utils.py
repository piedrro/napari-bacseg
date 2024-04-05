import copy
import json
import os
import traceback
import cv2
import mat4py
import numpy as np
import pandas as pd
import tifffile
from glob2 import glob
from napari.utils.notifications import show_info

class _import_utils:

    def read_scanr_directory(self, path):
        measurements, file_paths, channels = None, None, None

        try:
            if isinstance(path, list) == False:
                path = [path]

            if len(path) == 1:
                path = os.path.abspath(path[0])

                if os.path.isfile(path) == True:
                    selected_paths = [path]
                    image_path = os.path.abspath(path)
                    file_directory = os.path.abspath(os.path.basename(image_path))
                    file_paths = glob(file_directory + r"*\*.tif")
                else:
                    file_paths = glob(path + r"*\**\*.tif", recursive=True)
                    selected_paths = []
            else:
                selected_paths = [os.path.abspath(path) for path in path]
                image_path = os.path.abspath(path[0])
                file_directory = os.path.abspath(os.path.basename(image_path))
                file_paths = glob(file_directory + r"*\*.tif")

            scanR_meta_files = [path.replace(os.path.basename(path), "") for path in file_paths]
            scanR_meta_files = np.unique(scanR_meta_files).tolist()
            scanR_meta_files = [glob(path + "*.ome.xml")[0] for path in scanR_meta_files if len(glob(path + "*.ome.xml")) > 0]

            file_info = self.read_xml(scanR_meta_files)

            files = []

            for path in file_paths:
                try:
                    file = file_info[path]
                    file["path"] = path

                    path = os.path.normpath(path)

                    split_path = path.split(os.sep)

                    if len(split_path) >= 4:
                        folder = path.split(os.sep)[-4]
                    else:
                        folder = ""

                    if len(split_path) >= 5:
                        parent_folder = path.split(os.sep)[-5]
                    else:
                        parent_folder = ""

                    file["folder"] = folder
                    file["parent_folder"] = parent_folder

                    files.append(file)

                except:
                    pass

            files = pd.DataFrame(files)

            num_measurements = len(files.position_index.unique())

            import_limit = self.import_limit.currentText()

            if import_limit == "None":
                import_limit = num_measurements
            else:
                if int(import_limit) > num_measurements:
                    import_limit = num_measurements

            acquisitions = files.position_index.unique()[: int(import_limit)]

            files = files[files["position_index"] <= acquisitions[-1]]

            measurements = files.groupby(by=["parent_folder", "position_index", "time_index", "z_index"])

            if selected_paths != []:
                filtered_measurements = []

                for i in range(len(measurements)):
                    measurement = measurements.get_group(list(measurements.groups)[i])
                    measurement_paths = measurement["path"].tolist()

                    selected_paths = [os.path.abspath(path) for path in selected_paths]
                    measurement_paths = [os.path.abspath(path) for path in measurement_paths]

                    if not set(selected_paths).isdisjoint(measurement_paths):
                        filtered_measurements.append(measurement)

                filtered_measurements = pd.concat(filtered_measurements)

                measurements = filtered_measurements.groupby(by=["folder", "position_index", "time_index", "z_index"])

            channels = files["channel"].drop_duplicates().to_list()

            channel_num = str(len(files["channel"].unique()))

            if self.widget_notifications:
                show_info("Found " + str(len(measurements)) + " measurments in ScanR Folder(s) with " + channel_num + " channels.")

        except:
            measurements, file_paths, channels = None, None, None
            print(traceback.format_exc())

        return measurements, file_paths, channels

    def read_scanr_images(self, progress_callback, measurements, channels):
        scanr_images = {}
        img_shape = (100, 100)
        img_type = np.uint16
        iter = 0

        num_measurements = len(measurements)

        import_limit = self.import_limit.currentText()

        if import_limit == "None":
            import_limit = num_measurements
        else:
            if int(import_limit) > num_measurements:
                import_limit = num_measurements

        for i in range(int(import_limit)):
            measurement = measurements.get_group(list(measurements.groups)[i])

            measurement_channels = measurement["channel"].tolist()

            for channel in channels:
                iter += 1
                progress = int((iter / (len(measurements) * len(channels))) * 100)

                try:
                    progress_callback.emit(progress)
                except:
                    pass

                if self.widget_notifications:
                    show_info("loading image[" + channel + "] " + str(i + 1) + " of " + str(len(measurements)))

                if channel in measurement_channels:
                    dat = measurement[measurement["channel"] == channel]

                    path = dat["path"].item()
                    laser = "LED"
                    folder = dat["folder"].item()
                    parent_folder = dat["parent_folder"].item()
                    modality = dat["modality"].item()

                    import_precision = self.import_precision.currentText()
                    multiframe_mode = self.import_multiframe_mode.currentIndex()
                    crop_mode = self.import_crop_mode.currentIndex()

                    image_list, meta = self.read_image_file(path, import_precision, multiframe_mode, crop_mode)
                    img = image_list[0]

                    contrast_limit, alpha, beta, gamma = self.autocontrast_values(img)

                    self.active_import_mode = "ScanR"

                    meta["image_name"] = os.path.basename(path)
                    meta["image_path"] = path
                    meta["folder"] = folder
                    meta["parent_folder"] = parent_folder
                    meta["akseg_hash"] = self.get_hash(img_path=path)
                    meta["import_mode"] = "ScanR"
                    meta["contrast_limit"] = contrast_limit
                    meta["contrast_alpha"] = alpha
                    meta["contrast_beta"] = beta
                    meta["contrast_gamma"] = gamma
                    meta["dims"] = [img.shape[-1], img.shape[-2]]
                    meta["crop"] = [0, img.shape[-2], 0, img.shape[-1]]

                    meta["InstrumentSerial"] = "NA"
                    meta["microscope"] = "ScanR"
                    meta["modality"] = modality
                    meta["light_source"] = "LED"

                    img_shape = img.shape
                    img_type = np.array(img).dtype

                    image_path = meta["image_path"]
                    image_path = os.path.normpath(image_path)

                    if "pos_" in image_path:
                        meta["folder"] = image_path.split(os.sep)[-4]
                        meta["parent_folder"] = image_path.split(os.sep)[-5]

                else:
                    img = np.zeros(img_shape, dtype=img_type)
                    meta = {}

                    self.active_import_mode = "ScanR"

                    meta["image_name"] = "missing image channel"
                    meta["image_path"] = "missing image channel"
                    meta["folder"] = (None,)
                    meta["parent_folder"] = (None,)
                    meta["akseg_hash"] = None
                    meta["fov_mode"] = None
                    meta["import_mode"] = "NIM"
                    meta["contrast_limit"] = None
                    meta["contrast_alpha"] = None
                    meta["contrast_beta"] = None
                    meta["contrast_gamma"] = None
                    meta["dims"] = [img.shape[-1], img.shape[-2]]
                    meta["crop"] = [0, img.shape[-2], 0, img.shape[-1]]
                    meta["light_source"] = channel

                if channel not in scanr_images:
                    scanr_images[channel] = dict(images=[img], masks=[], nmasks=[], classes=[], metadata={i: meta}, )
                else:
                    scanr_images[channel]["images"].append(img)
                    scanr_images[channel]["metadata"][i] = meta

        imported_data = dict(imported_images=scanr_images)

        return imported_data

    def read_nim_directory(self, path):
        if isinstance(path, list) == False:
            path = [path]

        if len(path) == 1:
            path = os.path.abspath(path[0])

            if os.path.isfile(path) == True:
                file_paths = [path]

            else:
                file_paths = glob(path + r"*\**\*.tif", recursive=True)
        else:
            file_paths = path

        file_paths = [os.path.normpath(file) for file in file_paths]

        file_paths = [file for file in file_paths if file.split(".")[-1] == "tif"]
        file_names = [path.split(os.sep)[-1] for path in file_paths]

        files = pd.DataFrame(columns=["path", "file_name", "folder", "parent_folder", "posX", "posY", "posZ", "laser", "timestamp", ])

        for i in range(len(file_paths)):
            try:
                path = file_paths[i]

                file_name = path.split(os.sep)[-1]
                folder = os.path.abspath(path).split(os.sep)[-2]
                parent_folder = os.path.abspath(path).split(os.sep)[-3]

                with tifffile.TiffFile(path) as tif:
                    tif_tags = {}
                    for tag in tif.pages[0].tags.values():
                        name, value = tag.name, tag.value
                        tif_tags[name] = value

                if "ImageDescription" in tif_tags:
                    metadata = tif_tags["ImageDescription"]
                    metadata = json.loads(metadata)

                    laseractive = metadata["LaserActive"]
                    laserpowers = metadata["LaserPowerPercent"]
                    laserwavelength_nm = metadata["LaserWavelength_nm"]
                    timestamp = metadata["timestamp_us"]

                    posX, posY, posZ = metadata["StagePos_um"]

                    if True in laseractive:
                        laseractive = np.array(laseractive, dtype=bool)
                        laserpowers = np.array(laserpowers, dtype=float)
                        laserwavelength_nm = np.array(laserwavelength_nm, dtype=str)

                        laser_index = np.where(laseractive == True)

                        n_active_lasers = len(laseractive[laseractive == True])

                        if n_active_lasers == 1:
                            laser = laserwavelength_nm[laser_index][0]

                        else:
                            max_power = laserpowers[laseractive == True].max()
                            laser_index = np.where(laserpowers == max_power)
                            laser = laserwavelength_nm[laser_index][0]

                    else:
                        laser = "White Light"

                    file_name = path.split(os.sep)[-1]

                    data = [path, file_name, posX, posY, posZ, laser, timestamp]

                    files.loc[len(files)] = [path, file_name, folder, parent_folder, posX, posY, posZ, laser, timestamp, ]

            except:
                pass

        files[["posX", "posY", "posZ"]] = files[["posX", "posY", "posZ"]].round(decimals=0)

        files = files.sort_values(by=["timestamp", "posX", "posY", "laser"], ascending=True)
        files = files.reset_index(drop=True)
        files["aquisition"] = 0

        positions = files[["posX", "posY"]].drop_duplicates()
        channels = files["laser"].drop_duplicates().to_list()

        acquisition = 0
        lasers = []

        for i in range(len(positions)):
            posX = positions["posX"].iloc[i]
            posY = positions["posY"].iloc[i]

            data = files[(files["posX"] == posX) & (files["posY"] == posY)]

            indicies = data.index.values

            for index in indicies:
                laser = files.at[index, "laser"]

                if laser in lasers:
                    acquisition += 1
                    lasers = [laser]

                else:
                    lasers.append(laser)

                files.at[index, "aquisition"] = acquisition

        num_measurements = len(files.aquisition.unique())

        import_limit = self.import_limit.currentText()

        if import_limit == "None":
            import_limit = num_measurements
        else:
            if int(import_limit) > num_measurements:
                import_limit = num_measurements

        acquisitions = files.aquisition.unique()[: int(import_limit)]

        files = files[files["aquisition"] <= acquisitions[-1]]

        folder, parent_folder = self.get_folder(files)

        files["folder"] = folder
        files["parent_folder"] = parent_folder

        measurements = files.groupby(by=["aquisition"])
        channels = files["laser"].drop_duplicates().to_list()

        channel_num = str(len(files["laser"].unique()))

        if self.widget_notifications:
            show_info("Found " + str(len(measurements)) + " measurments in NIM Folder with " + channel_num + " channels.")

        return measurements, file_paths, channels

    def read_nim_images(self, progress_callback, measurements, channels):
        nim_images = {}
        img_shape = (100, 100)
        num_frames = 1
        img_type = np.uint16
        iter = 0

        img_index = {}

        for i in range(len(measurements)):
            measurement = measurements.get_group(list(measurements.groups)[i])
            measurement_channels = measurement["laser"].tolist()

            for j in range(len(channels)):
                channel = channels[j]

                if channel not in img_index.keys():
                    img_index[channel] = 0

                iter += 1
                progress = int((iter / (len(measurements) * len(channels))) * 100)

                try:
                    progress_callback.emit(progress)
                except:
                    pass

                if self.widget_notifications:
                    show_info("loading image[" + channel + "] " + str(i + 1) + " of " + str(len(measurements)))

                if channel in measurement_channels:
                    dat = measurement[measurement["laser"] == channel]

                    path = os.path.normpath(dat["path"].item())
                    laser = dat["laser"].item()
                    folder = dat["folder"].item()
                    parent_folder = dat["parent_folder"].item()

                    import_precision = self.import_precision.currentText()
                    multiframe_mode = self.import_multiframe_mode.currentIndex()
                    crop_mode = self.import_crop_mode.currentIndex()

                    image_list, meta = self.read_image_file(path, import_precision, multiframe_mode, crop_mode)

                    num_frames = len(image_list)

                    akseg_hash = self.get_hash(img_path=path)

                    file_name = os.path.basename(path)

                    for index, frame in enumerate(image_list):
                        frame_name = copy.deepcopy(file_name)
                        frame_meta = copy.deepcopy(meta)

                        contrast_limit = np.percentile(frame, (1, 99))
                        contrast_limit = [int(contrast_limit[0] * 0.5), int(contrast_limit[1] * 2), ]

                        self.active_import_mode = "nim"

                        if len(image_list) > 1:
                            frame_name = (frame_name.replace(".", "_") + "_" + str(index) + ".tif")

                        self.active_import_mode = "NIM"

                        frame_meta["image_name"] = frame_name
                        frame_meta["image_path"] = path
                        frame_meta["folder"] = folder
                        frame_meta["parent_folder"] = parent_folder
                        frame_meta["akseg_hash"] = akseg_hash
                        frame_meta["import_mode"] = "NIM"
                        frame_meta["contrast_limit"] = contrast_limit
                        frame_meta["contrast_alpha"] = 0
                        frame_meta["contrast_beta"] = 0
                        frame_meta["contrast_gamma"] = 0
                        frame_meta["dims"] = [frame.shape[-1], frame.shape[-2]]
                        frame_meta["crop"] = [0, frame.shape[-2], 0, frame.shape[-1], ]

                        if frame_meta["InstrumentSerial"] == "6D699GN6":
                            frame_meta["microscope"] = "BIO-NIM"
                        elif frame_meta["InstrumentSerial"] == "2EC5XTUC":
                            frame_meta["microscope"] = "JR-NIM"
                        else:
                            frame_meta["microscope"] = None

                        if frame_meta["IlluminationAngle_deg"] < 1:
                            frame_meta["modality"] = "Epifluorescence"
                        elif 1 < frame_meta["IlluminationAngle_deg"] < 53:
                            frame_meta["modality"] = "HILO"
                        elif 53 < frame_meta["IlluminationAngle_deg"]:
                            frame_meta["modality"] = "TIRF"

                        frame_meta["light_source"] = channel

                        if frame_meta["light_source"] == "White Light":
                            frame_meta["modality"] = "Bright Field"

                        img_shape = frame.shape
                        img_type = np.array(frame).dtype

                        image_path = frame_meta["image_path"]
                        image_path = os.path.normpath(image_path)

                        if "pos_" in image_path:
                            frame_meta["folder"] = image_path.split(os.sep)[-4]
                            frame_meta["parent_folder"] = image_path.split(os.sep)[-5]

                        if channel not in nim_images:
                            nim_images[channel] = dict(images=[frame], masks=[], nmasks=[], classes=[], metadata={img_index[channel]: frame_meta}, )
                        else:
                            nim_images[channel]["images"].append(frame)
                            nim_images[channel]["metadata"][img_index[channel]] = frame_meta

                        img_index[channel] += 1
                else:
                    for index in range(num_frames):
                        frame = np.zeros(img_shape, dtype=img_type)
                        frame_meta = {}

                        self.active_import_mode = "NIM"

                        frame_meta["image_name"] = "missing image channel"
                        frame_meta["image_path"] = "missing image channel"
                        frame_meta["folder"] = (None,)
                        frame_meta["parent_folder"] = (None,)
                        frame_meta["akseg_hash"] = None
                        frame_meta["fov_mode"] = None
                        frame_meta["import_mode"] = "NIM"
                        frame_meta["contrast_limit"] = None
                        frame_meta["contrast_alpha"] = None
                        frame_meta["contrast_beta"] = None
                        frame_meta["contrast_gamma"] = None
                        frame_meta["dims"] = [frame.shape[-1], frame.shape[-2]]
                        frame_meta["crop"] = [0, frame.shape[-2], 0, frame.shape[-1], ]
                        frame_meta["light_source"] = channel

                        if channel not in nim_images:
                            nim_images[channel] = dict(images=[frame], masks=[], nmasks=[], classes=[], metadata={img_index[channel]: frame_meta}, )
                        else:
                            nim_images[channel]["images"].append(frame)
                            nim_images[channel]["metadata"][img_index[channel]] = frame_meta

                        img_index[channel] += 1

        imported_data = dict(imported_images=nim_images)

        return imported_data

    def import_dataset(self, progress_callback, paths):
        path = os.path.abspath(paths[0])

        if os.path.isfile(path):
            path = os.path.normpath(path)
            path = os.path.abspath(os.path.join(path, f"..{os.sep}.."))
            path = os.path.normpath(path)
            folders = glob(path + f"**{os.sep}*")
        else:
            folders = glob(path + f"*{os.sep}*")

        folders = [os.path.normpath(x).split(os.sep)[-1].lower() for x in folders]

        if "images" in folders and "masks" in folders:
            image_paths = glob(path + f"{os.sep}images{os.sep}*.tif")
            mask_paths = glob(path + f"{os.sep}masks{os.sep}*.tif")

            images = []
            masks = []
            metadata = {}
            imported_images = {}

            import_limit = self.import_limit.currentText()

            if import_limit != "None" and len(image_paths) > int(import_limit):
                image_paths = image_paths[: int(import_limit)]

            for i in range(len(image_paths)):
                progress = int(((i + 1) / len(image_paths)) * 100)
                progress_callback.emit(progress)

                if self.widget_notifications:
                    show_info("loading image " + str(i + 1) + " of " + str(len(image_paths)))

                image_path = os.path.abspath(image_paths[i])
                image_path = os.path.normpath(image_path)
                mask_path = image_path.replace(f"{os.sep}images{os.sep}", f"{os.sep}masks{os.sep}")

                image_name = image_path.split(os.sep)[-1]
                mask_name = mask_path.split(os.sep)[-1]

                import_precision = self.import_precision.currentText()
                multiframe_mode = self.import_multiframe_mode.currentIndex()
                crop_mode = self.import_crop_mode.currentIndex()

                image_list, meta = self.read_image_file(path, import_precision, multiframe_mode)
                image = image_list[0]

                crop_mode = self.import_crop_mode.currentIndex()
                image =self.crop_image(image, crop_mode)

                if os.path.exists(mask_path):
                    mask = tifffile.imread(mask_path)
                    mask =self.crop_image(mask, crop_mode)
                    assert (len(mask.shape) < 3), "Can only import single channel masks"

                else:
                    mask_name = None
                    mask_path = None
                    mask = np.zeros(image.shape, dtype=np.uint16)

                contrast_limit, alpha, beta, gamma = self.autocontrast_values(image)

                self.active_import_mode = "Dataset"

                metadata["akseg_hash"] = self.get_hash(img_path=image_path)
                meta["image_name"] = image_name
                meta["image_path"] = image_path
                meta["mask_name"] = mask_name
                meta["mask_path"] = mask_path
                meta["label_name"] = None
                meta["label_path"] = None
                meta["import_mode"] = "Dataset"
                meta["contrast_limit"] = contrast_limit
                meta["contrast_alpha"] = alpha
                meta["contrast_beta"] = beta
                meta["contrast_gamma"] = gamma
                meta["dims"] = [image.shape[-1], image.shape[-2]]
                meta["crop"] = [0, image.shape[-2], 0, image.shape[-1]]

                images.append(image)
                metadata[i] = meta

                if imported_images == {}:
                    imported_images["Image"] = dict(images=[image], masks=[mask], nmasks=[], classes=[], metadata={i: meta}, )
                else:
                    imported_images["Image"]["images"].append(image)
                    imported_images["Image"]["masks"].append(mask)
                    imported_images["Image"]["metadata"][i] = meta

        imported_data = dict(imported_images=imported_images)

        return imported_data

    def import_bacseg(self, progress_callback, file_paths):
        path = os.path.abspath(file_paths[0])
        path = os.path.normpath(path)

        if os.path.isfile(path):
            path = os.path.abspath(os.path.join(path, "../../.."))
            folders = glob(path + "**/*")
        else:
            folders = glob(path + "*/*")

        folders = [os.path.normpath(folder) for folder in folders]
        folders = [os.path.abspath(x).split(os.sep)[-1].lower() for x in folders]

        if "images" in folders and "json" in folders:
            image_paths = glob(path + "/images/*.tif")
            json_paths = glob(path + "/json/*.tif")

            metadata = {}
            imported_images = {}
            akmeta = {}

            import_limit = self.import_limit.currentText()

            if import_limit != "None" and len(image_paths) > int(import_limit):
                image_paths = image_paths[: int(import_limit)]

            for i in range(len(image_paths)):
                progress = int(((i + 1) / len(image_paths)) * 100)
                progress_callback.emit(progress)

                if self.widget_notifications:
                    show_info("loading image " + str(i + 1) + " of " + str(len(image_paths)))

                image_path = os.path.abspath(image_paths[i])
                json_path = image_path.replace("\\images\\", "\\json\\").replace(".tif", ".txt")

                import_precision = self.import_precision.currentText()
                image_list, meta_stack = self.read_image_file(path, import_precision, multiframe_mode=0)
                image = image_list[0]

                crop_mode = self.import_crop_mode.currentIndex()
                image =self.crop_image(image, crop_mode)

                if os.path.exists(json_path):
                    from napari_bacseg.funcs.json_utils import import_coco_json

                    mask, nmask, label = import_coco_json(json_path)
                    mask =self.crop_image(mask, crop_mode)
                    label =self.crop_image(label, crop_mode)

                else:
                    label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint16)
                    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint16)
                    nmask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint16)

                for j, channel in enumerate(meta_stack["channels"]):
                    img = image[j, :, :]

                    contrast_limit, alpha, beta, gamma = self.autocontrast_values(img)

                    self.active_import_mode = "BacSeg"

                    meta = meta_stack["layer_meta"][channel]
                    meta["import_mode"] = "BacSeg"
                    meta["contrast_limit"] = contrast_limit
                    meta["contrast_alpha"] = alpha
                    meta["contrast_beta"] = beta
                    meta["contrast_gamma"] = gamma
                    meta["dims"] = [img.shape[0], img.shape[1]]
                    meta["crop"] = [0, img.shape[1], 0, img.shape[0]]

                    if channel not in imported_images.keys():
                        imported_images[channel] = dict(images=[img], masks=[mask], nmasks=[], classes=[label], metadata={i: meta}, )
                    else:
                        imported_images[channel]["images"].append(img)
                        imported_images[channel]["masks"].append(mask)
                        imported_images[channel]["nmasks"].append(nmask)
                        imported_images[channel]["classes"].append(label)
                        imported_images[channel]["metadata"][i] = meta

        akmeta = meta_stack
        akmeta.pop("layer_meta")

        imported_data = dict(imported_images=imported_images, akmeta=akmeta)

        return imported_data

    def import_images(self, progress_callback, file_paths):
        if os.path.isdir(file_paths[0]):
            file_paths = glob(file_paths[0] + r"**\*", recursive=True)

        image_formats = ["tif", "png", "jpeg", "fits"]

        file_paths = [path for path in file_paths if path.split(".")[-1] in image_formats]

        import_limit = self.import_limit.currentText()

        if import_limit != "None" and len(file_paths) > int(import_limit):
            file_paths = file_paths[: int(import_limit)]

        images = []
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

            file_path = os.path.abspath(file_paths[i])
            file_name = os.path.basename(file_path)

            import_precision = self.import_precision.currentText()
            multiframe_mode = self.import_multiframe_mode.currentIndex()
            crop_mode = self.import_crop_mode.currentIndex()

            image_list, meta = self.read_image_file(file_path, import_precision, multiframe_mode, crop_mode)

            akseg_hash = self.get_hash(img_path=file_path)

            file_name = os.path.basename(file_path)

            for index, frame in enumerate(image_list):
                contrast_limit = np.percentile(frame, (1, 99))
                contrast_limit = [int(contrast_limit[0] * 0.5), int(contrast_limit[1] * 2), ]

                self.active_import_mode = "image"

                if len(image_list) > 1:
                    frame_name = (file_name.replace(".", "_") + "_" + str(index) + ".tif")
                else:
                    frame_name = copy.deepcopy(file_name)

                frame_meta = copy.deepcopy(meta)

                frame_meta["akseg_hash"] = akseg_hash
                frame_meta["image_name"] = frame_name
                frame_meta["image_path"] = file_path
                frame_meta["mask_name"] = None
                frame_meta["mask_path"] = None
                frame_meta["label_name"] = None
                frame_meta["label_path"] = None
                frame_meta["import_mode"] = "image"
                frame_meta["contrast_limit"] = contrast_limit
                frame_meta["contrast_alpha"] = 0
                frame_meta["contrast_beta"] = 0
                frame_meta["contrast_gamma"] = 0
                frame_meta["dims"] = [frame.shape[-1], frame.shape[-2]]
                frame_meta["crop"] = [0, frame.shape[-2], 0, frame.shape[-1]]

                images.append(frame)
                metadata[i] = frame_meta

                if imported_images == {}:
                    imported_images["Image"] = dict(images=[frame], masks=[], nmasks=[], classes=[], metadata={img_index: frame_meta}, )
                else:
                    imported_images["Image"]["images"].append(frame)
                    imported_images["Image"]["metadata"][img_index] = frame_meta

                img_index += 1

        imported_data = dict(imported_images=imported_images)

        return imported_data

    def import_cellpose(self, progress_callback, file_paths):
        if os.path.isdir(file_paths[0]):
            file_paths = glob(file_paths[0] + r"**\*", recursive=True)

        image_formats = ["npy"]

        file_paths = [path for path in file_paths if path.split(".")[-1] in image_formats]

        import_limit = self.import_limit.currentText()

        if import_limit != "None" and len(file_paths) > int(import_limit):
            file_paths = file_paths[: int(import_limit)]

        imported_images = {}

        for i in range(len(file_paths)):
            progress = int(((i + 1) / len(file_paths)) * 100)
            progress_callback.emit(progress)

            if self.widget_notifications:
                show_info("loading image " + str(i + 1) + " of " + str(len(file_paths)))

            file_path = os.path.abspath(file_paths[i])
            file_path = os.path.normpath(file_path)
            file_name = file_path.split(os.sep)[-1]

            dat = np.load(file_path, allow_pickle=True).item()

            mask = dat["masks"]
            mask = mask.astype(np.uint16)

            image_path = file_path.replace("_seg.npy", ".tif")

            if os.path.exists(image_path):
                image_name = image_path.split(os.sep)[-1]

                import_precision = self.import_precision.currentText()
                multiframe_mode = self.import_multiframe_mode.currentIndex()
                image_list, meta = self.read_image_file(image_path, import_precision, multiframe_mode)
                img = image_list[0]

                crop_mode = self.import_crop_mode.currentIndex()
                img =self.crop_image(img, crop_mode)
                mask =self.crop_image(mask, crop_mode)

                contrast_limit, alpha, beta, gamma = self.autocontrast_values(img)

                self.active_import_mode = "cellpose"

                meta["akseg_hash"] = self.get_hash(img_path=image_path)
                meta["image_name"] = image_name
                meta["image_path"] = image_path
                meta["mask_name"] = file_name
                meta["mask_path"] = file_path
                meta["label_name"] = None
                meta["label_path"] = None
                meta["import_mode"] = "cellpose"
                meta["contrast_limit"] = contrast_limit
                meta["contrast_alpha"] = alpha
                meta["contrast_beta"] = beta
                meta["contrast_gamma"] = gamma
                meta["dims"] = [img.shape[-1], img.shape[-2]]
                meta["crop"] = [0, img.shape[-2], 0, img.shape[-1]]

            else:
                image = dat["img"]

                contrast_limit, alpha, beta, gamma = self.autocontrast_values(image)

                self.active_import_mode = "cellpose"

                folder = os.path.abspath(file_path).split(os.sep)[-2]
                parent_folder = os.path.abspath(file_path).split(os.sep)[-3]

                meta = dict(image_name=file_name, image_path=file_path, mask_name=file_name, mask_path=file_path, label_name=None, label_path=None, folder=folder, parent_folder=parent_folder, contrast_limit=contrast_limit, contrast_alpha=alpha, contrast_beta=beta, contrast_gamma=gamma, akseg_hash=self.get_hash(img_path=file_path), import_mode="cellpose", dims=[
                    image.shape[0], image.shape[1]], crop=[0, image.shape[1], 0, image.shape[0]], )

            if imported_images == {}:
                imported_images["Image"] = dict(images=[img], masks=[mask], nmasks=[], classes=[], metadata={i: meta}, )
            else:
                imported_images["Image"]["images"].append(img)
                imported_images["Image"]["masks"].append(mask)
                imported_images["Image"]["metadata"][i] = meta

        imported_data = dict(imported_images=imported_images)

        return imported_data

    def import_oufti(self, progress_callback, file_paths):
        if os.path.isdir(file_paths[0]):
            file_paths = glob(file_paths[0] + r"**\*", recursive=True)

        image_formats = ["mat"]

        file_paths = [path for path in file_paths if path.split(".")[-1] in image_formats]

        file_path = os.path.abspath(file_paths[0])
        file_path = os.path.normpath(file_path)
        parent_dir = file_path.replace(file_path.split(os.sep)[-1], "")

        mat_paths = file_paths
        image_paths = glob(parent_dir + r"**\*", recursive=True)

        image_paths = [os.path.normpath(path) for path in image_paths]

        image_formats = ["tif"]
        image_paths = [path for path in image_paths if path.split(".")[-1] in image_formats]

        mat_files = [path.split(os.sep)[-1] for path in mat_paths]
        image_files = [path.split(os.sep)[-1] for path in image_paths]

        matching_image_paths = []
        matching_mat_paths = []

        for i in range(len(image_files)):
            image_file = image_files[i].replace(".tif", "")

            index = [i for i, x in enumerate(mat_files) if image_file in x]

            if index != []:
                image_path = image_paths[i]
                mat_path = mat_paths[index[0]]

                matching_mat_paths.append(mat_path)
                matching_image_paths.append(image_path)

        if self.import_limit.currentText() == "1":
            if file_path in matching_image_paths:
                index = matching_image_paths.index(file_path)
                image_files = [matching_image_paths[index]]
                mat_files = [matching_mat_paths[index]]

            elif file_path in matching_mat_paths:
                index = matching_mat_paths.index(file_path)
                image_files = [matching_image_paths[index]]
                mat_files = [matching_mat_paths[index]]

            else:
                if self.widget_notifications:
                    show_info("Matching image/mesh files could not be found")
                self.viewer.text_overlay.visible = True
                self.viewer.text_overlay.text = ("Matching image/mesh files could not be found")

        else:
            image_files = matching_image_paths
            mat_files = matching_mat_paths

        import_limit = self.import_limit.currentText()

        if import_limit != "None" and len(mat_files) > int(import_limit):
            mat_files = mat_files[: int(import_limit)]

        imported_images = {}

        for i in range(len(mat_files)):
            try:
                progress = int(((i + 1) / len(mat_files)) * 100)
                progress_callback.emit(progress)

                if self.widget_notifications:
                    show_info("loading image " + str(i + 1) + " of " + str(len(mat_files)))

                mat_path = mat_files[i]
                image_path = image_files[i]

                image_path = os.path.normpath(image_path)
                mat_path = os.path.normpath(mat_path)

                image_name = image_path.split(os.sep)[-1]
                mat_name = mat_path.split(os.sep)[-1]

                image, mask, meta = self.import_mat_data(self, image_path, mat_path)

                crop_mode = self.import_crop_mode.currentIndex()
                image =self.crop_image(image, crop_mode)
                mask =self.crop_image(mask, crop_mode)

                contrast_limit, alpha, beta, gamma = self.autocontrast_values(image)

                self.active_import_mode = "oufti"

                meta["akseg_hash"] = self.get_hash(img_path=image_path)
                meta["image_name"] = image_name
                meta["image_path"] = image_path
                meta["mask_name"] = mat_name
                meta["mask_path"] = mat_path
                meta["label_name"] = None
                meta["label_path"] = None
                meta["import_mode"] = "oufti"
                meta["contrast_limit"] = contrast_limit
                meta["contrast_alpha"] = alpha
                meta["contrast_beta"] = beta
                meta["contrast_gamma"] = gamma
                meta["dims"] = [image.shape[-1], image.shape[-2]]
                meta["crop"] = [0, image.shape[-2], 0, image.shape[-1]]

                if imported_images == {}:
                    imported_images["Image"] = dict(images=[image], masks=[mask], nmasks=[], classes=[], metadata={i: meta}, )
                else:
                    imported_images["Image"]["images"].append(image)
                    imported_images["Image"]["masks"].append(mask)
                    imported_images["Image"]["metadata"][i] = meta

            except:
                pass

        imported_data = dict(imported_images=imported_images)

        return imported_data

    def import_mat_data(self, image_path, mat_path):

        import_precision = self.import_precision.currentText()
        multiframe_mode = self.import_multiframe_mode.currentIndex()
        crop_mode = self.import_crop_mode.currentIndex()
        image_list, meta = self.read_image_file(image_path, import_precision, multiframe_mode)
        image = image_list[0]

        mat_data = mat4py.loadmat(mat_path)

        mat_data = mat_data["cellList"]

        contours = []

        for dat in mat_data:
            if type(dat) == dict:
                cnt = dat["model"]
                cnt = np.array(cnt).reshape((-1, 1, 2)).astype(np.int32)
                contours.append(cnt)

        mask = np.zeros(image.shape, dtype=np.uint16)

        for i in range(len(contours)):
            cnt = contours[i]

            cv2.drawContours(mask, [cnt], -1, i + 1, -1)

        return image, mask, meta

    def import_JSON(self, progress_callback, file_paths):
        if os.path.isdir(file_paths[0]):
            file_paths = glob(file_paths[0] + r"**\*", recursive=True)

        image_formats = ["txt"]

        json_paths = [path for path in file_paths if path.split(".")[-1] in image_formats]

        file_path = os.path.abspath(file_paths[0])
        file_path = os.path.normpath(file_path)

        parent_dir = file_path.replace(file_path.split(os.sep)[-1], "")

        image_paths = glob(parent_dir + "*.tif", recursive=True)

        json_paths = [os.path.normpath(path) for path in json_paths]
        image_paths = [os.path.normpath(path) for path in image_paths]

        json_files = [path.split(os.sep)[-1] for path in json_paths]
        image_files = [path.split(os.sep)[-1] for path in image_paths]

        matching_image_paths = []
        matching_json_paths = []

        images = []
        masks = []
        metadata = {}

        import_limit = self.import_limit.currentText()

        for i in range(len(image_files)):
            image_file = image_files[i].replace(".tif", "")

            index = [i for i, x in enumerate(json_files) if image_file in x]

            if index != []:
                image_path = image_paths[i]
                json_path = json_paths[index[0]]

                matching_json_paths.append(json_path)
                matching_image_paths.append(image_path)

        if self.import_limit.currentText() == "1":
            if file_path in matching_image_paths:
                index = matching_image_paths.index(file_path)
                image_files = [matching_image_paths[index]]
                json_files = [matching_json_paths[index]]

            elif file_path in matching_json_paths:
                index = matching_json_paths.index(file_path)
                image_files = [matching_image_paths[index]]
                json_files = [matching_json_paths[index]]

            else:
                if self.widget_notifications:
                    show_info("Matching image/mesh files could not be found")

                self.viewer.text_overlay.visible = True
                self.viewer.text_overlay.text = ("Matching image/mesh files could not be found")

        else:
            image_files = matching_image_paths
            json_files = matching_json_paths

        imported_images = {}

        if import_limit != "None" and len(json_files) > int(import_limit):
            json_files = json_files[: int(import_limit)]

        for i in range(len(json_files)):
            try:
                progress = int(((i + 1) / len(json_files)) * 100)
                progress_callback.emit(progress)

                if self.widget_notifications:
                    show_info("loading image " + str(i + 1) + " of " + str(len(json_files)))

                json_path = json_files[i]
                image_path = image_files[i]

                json_path = os.path.normpath(json_path)
                image_path = os.path.normpath(image_path)

                image_name = image_path.split(os.sep)[-1]
                json_name = json_path.split(os.sep)[-1]

                import_precision = self.import_precision.currentText()
                multiframe_mode = self.import_multiframe_mode.currentIndex()
                crop_mode = self.import_crop_mode.currentIndex()
                image_list, meta = self.read_image_file(image_path, import_precision, multiframe_mode)
                image = image_list[0]

                from napari_bacseg.funcs.json_utils import import_coco_json

                mask, nmask, labels = import_coco_json(json_path)

                crop_mode = self.import_crop_mode.currentIndex()
                image = self.crop_image(image, crop_mode)
                mask = self.crop_image(mask, crop_mode)
                nmask = self.crop_image(nmask, crop_mode)
                labels = self.crop_image(labels, crop_mode)

                contrast_limit, alpha, beta, gamma = self.autocontrast_values(image)

                self.active_import_mode = "JSON"

                meta["akseg_hash"] = self.get_hash(img_path=image_path)
                meta["image_name"] = image_name
                meta["image_path"] = image_path
                meta["mask_name"] = json_name
                meta["mask_path"] = json_path
                meta["label_name"] = json_name
                meta["label_path"] = json_path
                meta["import_mode"] = "JSON"
                meta["contrast_limit"] = contrast_limit
                meta["contrast_alpha"] = alpha
                meta["contrast_beta"] = beta
                meta["contrast_gamma"] = gamma
                meta["dims"] = [image.shape[-1], image.shape[-2]]
                meta["crop"] = [0, image.shape[-2], 0, image.shape[-1]]

                if imported_images == {}:
                    imported_images["Image"] = dict(images=[image], masks=[mask], nmasks=[nmask], classes=[labels], metadata={i: meta}, )
                else:
                    imported_images["Image"]["images"].append(image)
                    imported_images["Image"]["masks"].append(mask)
                    imported_images["Image"]["nmasks"].append(nmask)
                    imported_images["Image"]["classes"].append(labels)
                    imported_images["Image"]["metadata"][i] = meta

            except:
                pass

        imported_data = dict(imported_images=imported_images)

        return imported_data

    def import_masks(self, file_paths, file_extension=""):
        mask_stack = self.segLayer.data.copy()
        nmask_stack = self.nucLayer.data.copy()
        class_stack = self.classLayer.data.copy()

        if os.path.isdir(file_paths[0]):
            file_paths = os.path.abspath(file_paths[0])
            file_paths = os.path.normpath(file_paths)
            import_folder = file_paths

        if os.path.isfile(file_paths[0]):
            file_paths = os.path.abspath(file_paths[0])
            file_paths = os.path.normpath(file_paths)
            import_folder = file_paths.replace(file_paths.split(os.sep)[-1], "")

        import_folder = os.path.abspath(import_folder)
        mask_paths = glob(import_folder + r"**\**\*" + file_extension, recursive=True)

        mask_files = [path.split(os.sep)[-1] for path in mask_paths]
        mask_search = [file.split(file.split(".")[-1])[0][:-1] for file in mask_files]

        layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Nucleoid", "Classes", "center_lines", "Localisations"]]

        matching_masks = []

        for layer in layer_names:
            image_stack = self.viewer.layers[layer].data
            meta_stack = self.viewer.layers[layer].metadata

            for i in range(len(image_stack)):
                meta = meta_stack[i]
                extension = meta["image_name"].split(".")[-1]
                image_name = meta["image_name"].split(extension)[0][:-1]
                image_path = meta["image_path"]
                crop = meta["crop"]

                indices = [i for i, x in enumerate(mask_search) if image_name in x]

                for index in indices:
                    mask_path = mask_paths[index]

                    if mask_path != image_path:
                        matching_masks.append([i, mask_path, image_path, crop])

        for mask_data in matching_masks:
            i, mask_path, image_path, crop = mask_data

            [y1, y2, x1, x2] = crop

            file_format = mask_path.split(".")[-1]

            if file_format == "tif":
                mask = tifffile.imread(mask_path)
                mask_stack[i, :, :][y1:y2, x1:x2] = mask
                self.segLayer.data = mask_stack.astype(np.uint16)

            if file_format == "txt":
                from napari_bacseg.funcs.json_utils import import_coco_json

                mask, nmask, label = import_coco_json(mask_path)
                mask_stack[i, :, :][y1:y2, x1:x2] = mask
                nmask_stack[i, :, :][y1:y2, x1:x2] = nmask
                class_stack[i, :, :][y1:y2, x1:x2] = label

                self.segLayer.data = mask_stack.astype(np.uint16)
                self.nucLayer.data = nmask_stack.astype(np.uint16)
                self.classLayer.data = class_stack.astype(np.uint16)

            if file_format == "npy":
                dat = np.load(mask_path, allow_pickle=True).item()

                mask = dat["masks"]
                mask = mask.astype(np.uint16)
                mask_stack[i, :, :][y1:y2, x1:x2] = mask
                self.segLayer.data = mask_stack.astype(np.uint16)

            if file_format == "mat":
                image, mask, meta = self.import_mat_data(self, image_path, mask_path)
                mask_stack[i, :, :][y1:y2, x1:x2] = mask
                self.segLayer.data = mask_stack.astype(np.uint16)