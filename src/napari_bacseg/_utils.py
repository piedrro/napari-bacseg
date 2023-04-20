import hashlib
import json
import os

# from napari_bacseg._utils_cellpose import export_cellpose
# from napari_bacseg._utils_oufti import  export_oufti
# from napari_bacseg._utils_imagej import export_imagej
# from napari_bacseg._utils_json import import_coco_json, export_coco_json
import tempfile
import traceback
import warnings

import cv2
import mat4py
import numpy as np
import pandas as pd
import scipy
import tifffile
import xmltodict
from astropy.io import fits
from glob2 import glob
from napari.utils.notifications import show_info

# from napari_bacseg._utils_imagej import read_imagej_file
from skimage import exposure
from skimage.registration import phase_cross_correlation


def normalize99(X):
    """normalize image so 0.0 is 0.01st percentile and 1.0 is 99.99th percentile"""

    if np.max(X) > 0:
        X = X.copy()
        v_min, v_max = np.percentile(X[X != 0], (0.1, 99.9))
        X = exposure.rescale_intensity(X, in_range=(v_min, v_max))

    return X


def rescale01(x):
    """normalize image from 0 to 1"""

    if np.max(x) > 0:
        x = (x - np.min(x)) / (np.max(x) - np.min(x))

    return x


    
def read_xml(paths):
    try:
        files = {}

        for path in paths:
            with open(path) as fd:
                dat = xmltodict.parse(fd.read())["OME"]

                image_list = dat["Image"]

                if type(image_list) == dict:
                    image_list = [image_list]

                for i in range(len(image_list)):
                    img = image_list[i]

                    objective_id = int(
                        img["ObjectiveSettings"]["@ID"].split(":")[-1]
                    )
                    objective_dat = dat["Instrument"]["Objective"][
                        objective_id
                    ]
                    objective_mag = float(
                        objective_dat["@NominalMagnification"]
                    )
                    objective_na = float(objective_dat["@LensNA"])

                    pixel_size = float(img["Pixels"]["@PhysicalSizeX"])

                    position_index = i
                    microscope = "ScanR"
                    light_source = "LED"

                    channel_dict = {}

                    for j in range(len(img["Pixels"]["Channel"])):
                        channel_data = img["Pixels"]["Channel"][j]

                        channel_dict[j] = dict(
                            modality=channel_data["@IlluminationType"],
                            channel=channel_data["@Name"],
                            mode=channel_data["@AcquisitionMode"],
                            well=channel_data["@ID"]
                            .split("W")[1]
                            .split("P")[0],
                        )

                    primary_channel = ""

                    for j in range(len(img["Pixels"]["TiffData"])):
                        num_channels = img["Pixels"]["@SizeC"]
                        num_zstack = img["Pixels"]["@SizeZ"]

                        tiff_data = img["Pixels"]["TiffData"][j]

                        file_name = tiff_data["UUID"]["@FileName"]
                        file_path = os.path.abspath(
                            path.replace(os.path.basename(path), file_name)
                        )

                        try:
                            plane_data = img["Pixels"]["Plane"][j]
                            exposure_time = plane_data["@ExposureTime"]
                            posX = float(plane_data["@PositionX"])
                            posY = float(plane_data["@PositionY"])
                            posZ = float(plane_data["@PositionZ"])
                        except:
                            exposure_time = None
                            posX = None
                            posY = None
                            posZ = None

                        try:
                            channel_index = int(tiff_data["@FirstC"])
                            time_index = int(tiff_data["@FirstT"])
                            z_index = int(tiff_data["@FirstZ"])
                            channel_dat = channel_dict[channel_index]
                            modality = channel_dat["modality"]
                            channel = channel_dat["channel"]
                            well_index = int(channel_dat["well"])
                        except:
                            channel_index = None
                            time_index = None
                            z_index = None
                            channel_dat = None
                            modality = None
                            channel = None
                            well_index = None

                        files[file_path] = dict(
                            file_name=file_name,
                            well_index=well_index,
                            position_index=position_index,
                            channel_index=channel_index,
                            time_index=time_index,
                            z_index=z_index,
                            microscope=microscope,
                            light_source=light_source,
                            channel=channel,
                            modality=modality,
                            pixel_size=pixel_size,
                            objective_magnification=objective_mag,
                            objective_na=objective_na,
                            exposure_time=exposure_time,
                            posX=posX,
                            posY=posY,
                            posZ=posZ,
                        )
    except:
        print(traceback.format_exc())

    return files


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
                file_directory = os.path.abspath(
                    image_path.split(image_path.split("\\")[-1])[0]
                )
                file_paths = glob(file_directory + r"*\*.tif")

            else:
                file_paths = glob(path + r"*\**\*.tif", recursive=True)
                selected_paths = []
        else:
            selected_paths = [os.path.abspath(path) for path in path]
            image_path = os.path.abspath(path[0])
            file_directory = os.path.abspath(
                image_path.split(image_path.split("\\")[-1])[0]
            )
            file_paths = glob(file_directory + r"*\*.tif")

        scanR_meta_files = [
            path.replace(os.path.basename(path), "") for path in file_paths
        ]
        scanR_meta_files = np.unique(scanR_meta_files).tolist()
        scanR_meta_files = [
            glob(path + "*.ome.xml")[0]
            for path in scanR_meta_files
            if len(glob(path + "*.ome.xml")) > 0
        ]

        file_info = read_xml(scanR_meta_files)

        files = []

        for path in file_paths:
            try:
                file = file_info[path]
                file["path"] = path

                split_path = path.split("\\")

                if len(split_path) >= 4:
                    folder = path.split("\\")[-4]
                else:
                    folder = ""

                if len(split_path) >= 5:
                    parent_folder = path.split("\\")[-5]
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

        measurements = files.groupby(
            by=["parent_folder", "position_index", "time_index", "z_index"]
        )

        if selected_paths != []:
            filtered_measurements = []

            for i in range(len(measurements)):
                measurement = measurements.get_group(
                    list(measurements.groups)[i]
                )
                measurement_paths = measurement["path"].tolist()

                selected_paths = [
                    os.path.abspath(path) for path in selected_paths
                ]
                measurement_paths = [
                    os.path.abspath(path) for path in measurement_paths
                ]

                if not set(selected_paths).isdisjoint(measurement_paths):
                    filtered_measurements.append(measurement)

            filtered_measurements = pd.concat(filtered_measurements)

            measurements = filtered_measurements.groupby(
                by=["folder", "position_index", "time_index", "z_index"]
            )

        channels = files["channel"].drop_duplicates().to_list()

        channel_num = str(len(files["channel"].unique()))

        if self.widget_notifications:
            show_info(
                "Found "
                + str(len(measurements))
                + " measurments in ScanR Folder(s) with "
                + channel_num
                + " channels."
            )

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
                show_info(
                    "loading image["
                    + channel
                    + "] "
                    + str(i + 1)
                    + " of "
                    + str(len(measurements))
                )

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

                img, meta = read_image_file(
                    path, import_precision, multiframe_mode, crop_mode
                )

                contrast_limit, alpha, beta, gamma = autocontrast_values(img)

                self.active_import_mode = "ScanR"

                meta["image_name"] = os.path.basename(path)
                meta["image_path"] = path
                meta["folder"] = folder
                meta["parent_folder"] = parent_folder
                meta["akseg_hash"] = get_hash(img_path=path)
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

                if "pos_" in image_path:
                    meta["folder"] = image_path.split("\\")[-4]
                    meta["parent_folder"] = image_path.split("\\")[-5]

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
                scanr_images[channel] = dict(
                    images=[img], masks=[], classes=[], metadata={i: meta}
                )
            else:
                scanr_images[channel]["images"].append(img)
                scanr_images[channel]["metadata"][i] = meta

    imported_data = dict(imported_images=scanr_images)

    return imported_data


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
    metadata = {}
    imported_images = {}

    for i in range(len(file_paths)):
        progress = int(((i + 1) / len(file_paths)) * 100)
        try:
            progress_callback.emit(progress)
        except:
            pass

        if self.widget_notifications:
            show_info(
                "loading image " + str(i + 1) + " of " + str(len(file_paths))
            )

        paths = file_paths[i]
        paths = os.path.abspath(paths)

        import_precision = self.import_precision.currentText()
        multiframe_mode = self.import_multiframe_mode.currentIndex()
        crop_mode = self.import_crop_mode.currentIndex()
        image, meta = read_image_file(paths, import_precision, multiframe_mode)

        from napari_bacseg._utils_imagej import read_imagej_file

        mask = read_imagej_file(paths, image)

        contrast_limit, alpha, beta, gamma = autocontrast_values(image)

        self.active_import_mode = "Dataset"

        meta["akseg_hash"] = get_hash(img_path=paths)
        meta["image_name"] = os.path.basename(paths)
        meta["image_path"] = paths
        meta["mask_name"] = os.path.basename(paths)
        meta["mask_path"] = paths
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
        masks.append(mask)
        metadata[i] = meta

        if imported_images == {}:
            imported_images["Image"] = dict(
                images=[image], masks=[mask], classes=[], metadata={i: meta}
            )
        else:
            imported_images["Image"]["images"].append(image)
            imported_images["Image"]["masks"].append(mask)
            imported_images["Image"]["metadata"][i] = meta

    imported_data = dict(imported_images=imported_images)

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

    file_paths = [file for file in file_paths if file.split(".")[-1] == "tif"]

    file_names = [path.split("\\")[-1] for path in file_paths]

    files = pd.DataFrame(
        columns=[
            "path",
            "file_name",
            "folder",
            "parent_folder",
            "posX",
            "posY",
            "posZ",
            "laser",
            "timestamp",
        ]
    )

    for i in range(len(file_paths)):
        try:
            path = file_paths[i]
            path = os.path.abspath(path)

            file_name = path.split("\\")[-1]
            folder = os.path.abspath(path).split("\\")[-2]
            parent_folder = os.path.abspath(path).split("\\")[-3]

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
                    laserwavelength_nm = np.array(
                        laserwavelength_nm, dtype=str
                    )

                    # finds maximum active power
                    power = laserpowers[laseractive == True].max()

                    laser_index = np.where(laserpowers == power)

                    laser = laserwavelength_nm[laser_index][0]
                else:
                    laser = "White Light"

                file_name = path.split("\\")[-1]

                data = [path, file_name, posX, posY, posZ, laser, timestamp]

                files.loc[len(files)] = [
                    path,
                    file_name,
                    folder,
                    parent_folder,
                    posX,
                    posY,
                    posZ,
                    laser,
                    timestamp,
                ]

        except:
            pass

    files[["posX", "posY", "posZ"]] = files[["posX", "posY", "posZ"]].round(
        decimals=0
    )

    files = files.sort_values(
        by=["timestamp", "posX", "posY", "laser"], ascending=True
    )
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

    folder, parent_folder = get_folder(files)

    files["folder"] = folder
    files["parent_folder"] = parent_folder

    measurements = files.groupby(by=["aquisition"])
    channels = files["laser"].drop_duplicates().to_list()

    channel_num = str(len(files["laser"].unique()))

    if self.widget_notifications:
        show_info(
            "Found "
            + str(len(measurements))
            + " measurments in NIM Folder with "
            + channel_num
            + " channels."
        )

    return measurements, file_paths, channels


def get_folder(files):
    folder = ""
    parent_folder = ""

    paths = files["path"].tolist()

    if len(paths) > 1:
        paths = np.array([path.split("\\") for path in paths]).T

        for i in range(len(paths)):
            if len(set(paths[i])) != 1:
                folder = str(paths[i - 1][0])
                parent_folder = str(paths[i - 2][0])

                break

    else:
        folder = paths[0].split("\\")[-2]
        parent_folder = paths[0].split("\\")[-3]

    return folder, parent_folder


def read_image_file(path, precision="native", multiframe_mode=0, crop_mode=0):
    image_name = os.path.basename(path)

    if os.path.splitext(image_name)[1] == ".fits":
        with fits.open(path, ignore_missing_simple=True) as hdul:
            image = hdul[0].data
            try:
                metadata = dict(hdul[0].header)

                unserializable_keys = [
                    key
                    for key, value in metadata.items()
                    if type(value) not in [bool, int, float, str]
                ]

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

    image = crop_image(image, crop_mode)

    image = get_frame(image, multiframe_mode)
    image = rescale_image(image, precision=precision)

    folder = os.path.abspath(path).split("\\")[-2]
    parent_folder = os.path.abspath(path).split("\\")[-3]

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
        metadata["dims"] = [image.shape[-1], image.shape[-2]]
        metadata["crop"] = [0, image.shape[-2], 0, image.shape[-1]]

    return image, metadata


def get_frame(img, multiframe_mode):
    if len(img.shape) > 2:
        if multiframe_mode == 0:
            img = img[0, :, :]

        elif multiframe_mode == 1:
            img = np.max(img, axis=0)

        elif multiframe_mode == 2:
            img = np.mean(img, axis=0).astype(np.uint16)

        elif multiframe_mode == 3:
            img = np.sum(img, axis=0)

    return img


def crop_image(img, crop_mode=0):
    if crop_mode != 0:
        if len(img.shape) > 2:
            imgL = img[:, :, : img.shape[-1] // 2]
            imgR = img[:, :, img.shape[-1] // 2 :]
        else:
            imgL = img[:, : img.shape[-1] // 2]
            imgR = img[:, img.shape[-1] // 2 :]

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


def rescale_image(image, precision="int16"):
    precision_dict = {
        "int8": np.uint8,
        "int16": np.uint16,
        "int32": np.uint32,
        "native": image.dtype,
    }

    dtype = precision_dict[precision]

    if "int" in str(dtype):
        max_value = np.iinfo(dtype).max - 1
    else:
        max_value = np.finfo(dtype).max - 1

    if precision != "native":
        image = ((image - np.min(image)) / np.max(image)) * max_value
        image = image.astype(dtype)

    return image


def read_nim_images(self, progress_callback, measurements, channels):
    nim_images = {}
    img_shape = (100, 100)
    img_type = np.uint16
    iter = 0

    for i in range(len(measurements)):
        measurement = measurements.get_group(list(measurements.groups)[i])
        measurement_channels = measurement["laser"].tolist()

        for j in range(len(channels)):
            channel = channels[j]

            iter += 1
            progress = int((iter / (len(measurements) * len(channels))) * 100)

            try:
                progress_callback.emit(progress)
            except:
                pass

            if self.widget_notifications:
                show_info(
                    "loading image["
                    + channel
                    + "] "
                    + str(i + 1)
                    + " of "
                    + str(len(measurements))
                )

            if channel in measurement_channels:
                dat = measurement[measurement["laser"] == channel]

                path = dat["path"].item()
                laser = dat["laser"].item()
                folder = dat["folder"].item()
                parent_folder = dat["parent_folder"].item()

                import_precision = self.import_precision.currentText()
                multiframe_mode = self.import_multiframe_mode.currentIndex()
                crop_mode = self.import_crop_mode.currentIndex()
                img, meta = read_image_file(
                    path, import_precision, multiframe_mode, crop_mode
                )

                contrast_limit, alpha, beta, gamma = autocontrast_values(img)

                self.active_import_mode = "NIM"

                meta["image_name"] = os.path.basename(path)
                meta["image_path"] = path
                meta["folder"] = folder
                meta["parent_folder"] = parent_folder
                meta["akseg_hash"] = get_hash(img_path=path)
                meta["import_mode"] = "NIM"
                meta["contrast_limit"] = contrast_limit
                meta["contrast_alpha"] = alpha
                meta["contrast_beta"] = beta
                meta["contrast_gamma"] = gamma
                meta["dims"] = [img.shape[-1], img.shape[-2]]
                meta["crop"] = [0, img.shape[-2], 0, img.shape[-1]]

                if meta["InstrumentSerial"] == "6D699GN6":
                    meta["microscope"] = "BIO-NIM"
                elif meta["InstrumentSerial"] == "2EC5XTUC":
                    meta["microscope"] = "JR-NIM"
                else:
                    meta["microscope"] = None

                if meta["IlluminationAngle_deg"] < 1:
                    meta["modality"] = "Epifluorescence"
                elif 1 < meta["IlluminationAngle_deg"] < 53:
                    meta["modality"] = "HILO"
                elif 53 < meta["IlluminationAngle_deg"]:
                    meta["modality"] = "TIRF"

                meta["light_source"] = channel

                if meta["light_source"] == "White Light":
                    meta["modality"] = "Bright Field"

                img_shape = img.shape
                img_type = np.array(img).dtype

                image_path = meta["image_path"]

                if "pos_" in image_path:
                    meta["folder"] = image_path.split("\\")[-4]
                    meta["parent_folder"] = image_path.split("\\")[-5]

            else:
                img = np.zeros(img_shape, dtype=img_type)
                meta = {}

                self.active_import_mode = "NIM"

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

            if channel not in nim_images:
                nim_images[channel] = dict(
                    images=[img], masks=[], classes=[], metadata={i: meta}
                )
            else:
                nim_images[channel]["images"].append(img)
                nim_images[channel]["metadata"][i] = meta

    imported_data = dict(imported_images=nim_images)

    return imported_data


def get_brightest_fov(image):
    imageL = image[0, :, : image.shape[2] // 2]
    imageR = image[0, :, image.shape[2] // 2 :]

    if np.mean(imageL) > np.mean(imageR):
        image = image[:, :, : image.shape[2] // 2]
    else:
        image = image[:, :, : image.shape[2] // 2]

    return image


def imadjust(img):
    v_min, v_max = np.percentile(img, (1, 99))
    img = exposure.rescale_intensity(img, in_range=(v_min, v_max))

    return img


def get_channel(img, multiframe_mode):
    if len(img.shape) > 2:
        if multiframe_mode == 0:
            img = img[0, :, :]

        elif multiframe_mode == 1:
            img = np.max(img, axis=0)

        elif multiframe_mode == 2:
            img = np.mean(img, axis=0).astype(np.uint16)

    return img


def get_fov(img, channel_mode):
    imgL = img[:, : img.shape[1] // 2]
    imgR = img[:, img.shape[1] // 2 :]

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


def process_image(image, multiframe_mode, channel_mode):
    image = get_channel(image, multiframe_mode)

    image = get_fov(image, channel_mode)

    # if len(image.shape) < 3:
    #
    #     image = np.expand_dims(image, axis=0)

    return image


def stack_images(images, metadata=None):
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


def import_dataset(self, progress_callback, paths):
    path = os.path.abspath(paths[0])

    if os.path.isfile(path):
        path = os.path.abspath(os.path.join(path, "../.."))
        folders = glob(path + "**/*")
    else:
        folders = glob(path + "*/*")

    folders = [os.path.abspath(x).split("\\")[-1].lower() for x in folders]

    if "images" in folders and "masks" in folders:
        image_paths = glob(path + "/images/*.tif")
        mask_paths = glob(path + "/masks/*.tif")

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
                show_info(
                    "loading image "
                    + str(i + 1)
                    + " of "
                    + str(len(image_paths))
                )

            image_path = os.path.abspath(image_paths[i])
            mask_path = image_path.replace("\\images\\", "\\masks\\")

            image_name = image_path.split("\\")[-1]
            mask_name = mask_path.split("\\")[-1]

            import_precision = self.import_precision.currentText()
            multiframe_mode = self.import_multiframe_mode.currentIndex()
            crop_mode = self.import_crop_mode.currentIndex()
            image, meta = read_image_file(
                path, import_precision, multiframe_mode
            )

            crop_mode = self.import_crop_mode.currentIndex()
            image = crop_image(image, crop_mode)

            if os.path.exists(mask_path):
                mask = tifffile.imread(mask_path)
                mask = crop_image(mask, crop_mode)
                assert (
                    len(mask.shape) < 3
                ), "Can only import single channel masks"

            else:
                mask_name = None
                mask_path = None
                mask = np.zeros(image.shape, dtype=np.uint16)

            contrast_limit, alpha, beta, gamma = autocontrast_values(image)

            self.active_import_mode = "Dataset"

            metadata["akseg_hash"] = get_hash(img_path=image_path)
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
                imported_images["Image"] = dict(
                    images=[image],
                    masks=[mask],
                    classes=[],
                    metadata={i: meta},
                )
            else:
                imported_images["Image"]["images"].append(image)
                imported_images["Image"]["masks"].append(mask)
                imported_images["Image"]["metadata"][i] = meta

    imported_data = dict(imported_images=imported_images)

    return imported_data


def import_bacseg(self, progress_callback, file_paths):
    path = os.path.abspath(file_paths[0])

    if os.path.isfile(path):
        path = os.path.abspath(os.path.join(path, "../.."))
        folders = glob(path + "**/*")
    else:
        folders = glob(path + "*/*")

    folders = [os.path.abspath(x).split("\\")[-1].lower() for x in folders]

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
                show_info(
                    "loading image "
                    + str(i + 1)
                    + " of "
                    + str(len(image_paths))
                )

            image_path = os.path.abspath(image_paths[i])
            json_path = image_path.replace("\\images\\", "\\json\\").replace(
                ".tif", ".txt"
            )

            import_precision = self.import_precision.currentText()
            image, meta_stack = read_image_file(
                path, import_precision, multiframe_mode=0
            )

            crop_mode = self.import_crop_mode.currentIndex()
            image = crop_image(image, crop_mode)

            if os.path.exists(json_path):
                from napari_bacseg._utils_json import import_coco_json

                mask, label = import_coco_json(json_path)
                mask = crop_image(mask, crop_mode)
                label = crop_image(label, crop_mode)

            else:
                label = np.zeros(
                    (image.shape[0], image.shape[1]), dtype=np.uint16
                )
                mask = np.zeros(
                    (image.shape[0], image.shape[1]), dtype=np.uint16
                )

            for j, channel in enumerate(meta_stack["channels"]):
                img = image[j, :, :]

                contrast_limit, alpha, beta, gamma = autocontrast_values(img)

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
                    imported_images[channel] = dict(
                        images=[img],
                        masks=[mask],
                        classes=[label],
                        metadata={i: meta},
                    )
                else:
                    imported_images[channel]["images"].append(img)
                    imported_images[channel]["masks"].append(mask)
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

    file_paths = [
        path for path in file_paths if path.split(".")[-1] in image_formats
    ]

    import_limit = self.import_limit.currentText()

    if import_limit != "None" and len(file_paths) > int(import_limit):
        file_paths = file_paths[: int(import_limit)]

    images = []
    metadata = {}
    imported_images = {}

    for i in range(len(file_paths)):
        progress = int(((i + 1) / len(file_paths)) * 100)

        try:
            progress_callback.emit(progress)
        except:
            pass
        if self.widget_notifications:
            show_info(
                "loading image " + str(i + 1) + " of " + str(len(file_paths))
            )

        file_path = os.path.abspath(file_paths[i])
        file_name = os.path.basename(file_path)

        import_precision = self.import_precision.currentText()
        multiframe_mode = self.import_multiframe_mode.currentIndex()
        crop_mode = self.import_crop_mode.currentIndex()

        image, meta = read_image_file(
            file_path, import_precision, multiframe_mode, crop_mode
        )

        contrast_limit, alpha, beta, gamma = autocontrast_values(image)

        self.active_import_mode = "image"

        meta["akseg_hash"] = get_hash(img_path=file_path)
        meta["image_name"] = file_name
        meta["image_path"] = file_path
        meta["mask_name"] = None
        meta["mask_path"] = None
        meta["label_name"] = None
        meta["label_path"] = None
        meta["import_mode"] = "image"
        meta["contrast_limit"] = contrast_limit
        meta["contrast_alpha"] = alpha
        meta["contrast_beta"] = beta
        meta["contrast_gamma"] = gamma
        meta["dims"] = [image.shape[-1], image.shape[-2]]
        meta["crop"] = [0, image.shape[-2], 0, image.shape[-1]]

        images.append(image)
        metadata[i] = meta

        if imported_images == {}:
            imported_images["Image"] = dict(
                images=[image], masks=[], classes=[], metadata={i: meta}
            )
        else:
            imported_images["Image"]["images"].append(image)
            imported_images["Image"]["metadata"][i] = meta

    imported_data = dict(imported_images=imported_images)

    return imported_data


def import_cellpose(self, progress_callback, file_paths):
    if os.path.isdir(file_paths[0]):
        file_paths = glob(file_paths[0] + r"**\*", recursive=True)

    image_formats = ["npy"]

    file_paths = [
        path for path in file_paths if path.split(".")[-1] in image_formats
    ]

    import_limit = self.import_limit.currentText()

    if import_limit != "None" and len(file_paths) > int(import_limit):
        file_paths = file_paths[: int(import_limit)]

    imported_images = {}

    for i in range(len(file_paths)):
        progress = int(((i + 1) / len(file_paths)) * 100)
        progress_callback.emit(progress)

        if self.widget_notifications:
            show_info(
                "loading image " + str(i + 1) + " of " + str(len(file_paths))
            )

        file_path = os.path.abspath(file_paths[i])
        file_name = file_path.split("\\")[-1]

        dat = np.load(file_path, allow_pickle=True).item()

        mask = dat["masks"]
        mask = mask.astype(np.uint16)

        image_path = file_path.replace("_seg.npy", ".tif")

        if os.path.exists(image_path):
            image_name = image_path.split("\\")[-1]

            import_precision = self.import_precision.currentText()
            multiframe_mode = self.import_multiframe_mode.currentIndex()
            img, meta = read_image_file(
                image_path, import_precision, multiframe_mode
            )

            crop_mode = self.import_crop_mode.currentIndex()
            img = crop_image(img, crop_mode)
            mask = crop_image(mask, crop_mode)

            contrast_limit, alpha, beta, gamma = autocontrast_values(img)

            self.active_import_mode = "cellpose"

            meta["akseg_hash"] = get_hash(img_path=image_path)
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

            contrast_limit, alpha, beta, gamma = autocontrast_values(image)

            self.active_import_mode = "cellpose"

            folder = os.path.abspath(file_path).split("\\")[-2]
            parent_folder = os.path.abspath(file_path).split("\\")[-3]

            meta = dict(
                image_name=file_name,
                image_path=file_path,
                mask_name=file_name,
                mask_path=file_path,
                label_name=None,
                label_path=None,
                folder=folder,
                parent_folder=parent_folder,
                contrast_limit=contrast_limit,
                contrast_alpha=alpha,
                contrast_beta=beta,
                contrast_gamma=gamma,
                akseg_hash=get_hash(img_path=file_path),
                import_mode="cellpose",
                dims=[image.shape[0], image.shape[1]],
                crop=[0, image.shape[1], 0, image.shape[0]],
            )

        if imported_images == {}:
            imported_images["Image"] = dict(
                images=[img], masks=[mask], classes=[], metadata={i: meta}
            )
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

    file_paths = [
        path for path in file_paths if path.split(".")[-1] in image_formats
    ]

    file_path = os.path.abspath(file_paths[0])
    parent_dir = file_path.replace(file_path.split("\\")[-1], "")

    mat_paths = file_paths
    image_paths = glob(parent_dir + r"**\*", recursive=True)

    image_formats = ["tif"]
    image_paths = [
        path for path in image_paths if path.split(".")[-1] in image_formats
    ]

    mat_files = [path.split("\\")[-1] for path in mat_paths]
    image_files = [path.split("\\")[-1] for path in image_paths]

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
            self.viewer.text_overlay.text = (
                "Matching image/mesh files could not be found"
            )

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
                show_info(
                    "loading image "
                    + str(i + 1)
                    + " of "
                    + str(len(mat_files))
                )

            mat_path = mat_files[i]
            image_path = image_files[i]

            image_name = image_path.split("\\")[-1]
            mat_name = mat_path.split("\\")[-1]

            image, mask, meta = import_mat_data(self, image_path, mat_path)

            crop_mode = self.import_crop_mode.currentIndex()
            image = crop_image(image, crop_mode)
            mask = crop_image(mask, crop_mode)

            contrast_limit, alpha, beta, gamma = autocontrast_values(image)

            self.active_import_mode = "oufti"

            meta["akseg_hash"] = get_hash(img_path=image_path)
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
                imported_images["Image"] = dict(
                    images=[image],
                    masks=[mask],
                    classes=[],
                    metadata={i: meta},
                )
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
    image, meta = read_image_file(
        image_path, import_precision, multiframe_mode
    )

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


def unstack_images(stack, axis=0):
    images = [
        np.squeeze(e, axis)
        for e in np.split(stack, stack.shape[axis], axis=axis)
    ]

    return images


def append_image_stacks(
    current_metadata, new_metadata, current_image_stack, new_image_stack
):
    current_image_stack = unstack_images(current_image_stack)

    new_image_stack = unstack_images(new_image_stack)

    appended_image_stack = current_image_stack + new_image_stack

    appended_metadata = current_metadata

    for key, value in new_metadata.items():
        new_key = max(appended_metadata.keys()) + 1

        appended_metadata[new_key] = value

    appended_image_stack, appended_metadata = stack_images(
        appended_image_stack, appended_metadata
    )

    return appended_image_stack, appended_metadata


def append_metadata(current_metadata, new_metadata):
    appended_metadata = current_metadata

    for key, value in new_metadata.items():
        new_key = max(appended_metadata.keys()) + 1

        appended_metadata[new_key] = value

        return appended_metadata


def read_ak_metadata(self):
    meta_path = os.path.join(
        self.database_path, "Metadata", "AKSEG Metadata.xlsx"
    )

    ak_meta = pd.read_excel(meta_path)

    user_initials = list(ak_meta["User Initials"].dropna())
    microscope = list(ak_meta["Microscope"].dropna())
    modality = list(ak_meta["Image Modality"].dropna())

    ak_meta = dict(
        user_initials=user_initials, microscope=microscope, modality=modality
    )

    return ak_meta


def get_hash(img_path=None, img=None):
    if img is not None:
        img_path = tempfile.TemporaryFile(suffix=".tif").name
        tifffile.imwrite(img_path, img)

    with open(img_path, "rb") as f:
        bytes = f.read()  # read entire file as bytes

        hash_code = hashlib.sha256(bytes).hexdigest()

        return hash_code


def align_image_channels(self):
    layer_names = [
        layer.name
        for layer in self.viewer.layers
        if layer.name not in ["Segmentations", "Classes", "center_lines"]
    ]

    if self.import_align.isChecked() and len(layer_names) > 1:
        primary_image = layer_names[-1]

        layer_names.remove(primary_image)

        dim_range = int(self.viewer.dims.range[0][1])

        for i in range(dim_range):
            img = self.viewer.layers[primary_image].data[i, :, :]

            for layer in layer_names:
                shifted_img = self.viewer.layers[layer].data[i, :, :]

                try:
                    shift, error, diffphase = phase_cross_correlation(
                        img, shifted_img, upsample_factor=100
                    )
                    shifted_img = scipy.ndimage.shift(shifted_img, shift)

                except:
                    pass

                self.viewer.layers[layer].data[i, :, :] = shifted_img


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

                    cnt, _ = cv2.findContours(
                        cnt_mask.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_NONE,
                    )

                    contours.append(cnt[0])

        export_mask_stack[i, :, :][y1:y2, x1:x2] = export_mask
        export_label_stack[i, :, :][y1:y2, x1:x2] = export_label
        export_contours[i] = contours

    return export_mask_stack, export_label_stack, export_contours


def import_JSON(self, progress_callback, file_paths):
    if os.path.isdir(file_paths[0]):
        file_paths = glob(file_paths[0] + r"**\*", recursive=True)

    image_formats = ["txt"]

    json_paths = [
        path for path in file_paths if path.split(".")[-1] in image_formats
    ]

    file_path = os.path.abspath(file_paths[0])
    parent_dir = file_path.replace(file_path.split("\\")[-1], "")

    image_paths = glob(parent_dir + "*.tif", recursive=True)

    json_files = [path.split("\\")[-1] for path in json_paths]
    image_files = [path.split("\\")[-1] for path in image_paths]

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
            self.viewer.text_overlay.text = (
                "Matching image/mesh files could not be found"
            )

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
                show_info(
                    "loading image "
                    + str(i + 1)
                    + " of "
                    + str(len(json_files))
                )

            json_path = json_files[i]
            image_path = image_files[i]

            image_name = image_path.split("\\")[-1]
            json_name = json_path.split("\\")[-1]

            import_precision = self.import_precision.currentText()
            multiframe_mode = self.import_multiframe_mode.currentIndex()
            crop_mode = self.import_crop_mode.currentIndex()
            image, meta = read_image_file(
                image_path, import_precision, multiframe_mode
            )

            from napari_bacseg._utils_json import import_coco_json

            mask, labels = import_coco_json(json_path)

            crop_mode = self.import_crop_mode.currentIndex()
            image = crop_image(image, crop_mode)
            mask = crop_image(mask, crop_mode)
            labels = crop_image(labels, crop_mode)

            contrast_limit, alpha, beta, gamma = autocontrast_values(image)

            self.active_import_mode = "JSON"

            meta["akseg_hash"] = get_hash(img_path=image_path)
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
                imported_images["Image"] = dict(
                    images=[image],
                    masks=[mask],
                    classes=[labels],
                    metadata={i: meta},
                )
            else:
                imported_images["Image"]["images"].append(image)
                imported_images["Image"]["masks"].append(mask)
                imported_images["Image"]["classes"].append(labels)
                imported_images["Image"]["metadata"][i] = meta

        except:
            pass

    imported_data = dict(imported_images=imported_images)

    return imported_data


def get_histogram(image, bins):
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


def cumsum(a):
    """cumulative sum function"""

    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)


def autocontrast_values(image, clip_hist_percent=0.001):
    # calculate histogram
    hist, bin_edges = np.histogram(image, bins=(2**16) - 1)
    hist_size = len(hist)

    # calculate cumulative distribution from the histogram
    accumulator = cumsum(hist)

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


def import_masks(self, file_paths, file_extension=""):
    mask_stack = self.segLayer.data.copy()
    class_stack = self.classLayer.data.copy()

    if os.path.isdir(file_paths[0]):
        file_paths = os.path.abspath(file_paths[0])
        import_folder = file_paths

    if os.path.isfile(file_paths[0]):
        file_paths = os.path.abspath(file_paths[0])
        import_folder = file_paths.replace(file_paths.split("\\")[-1], "")

    import_folder = os.path.abspath(import_folder)
    mask_paths = glob(
        import_folder + r"**\**\*" + file_extension, recursive=True
    )

    mask_files = [path.split("\\")[-1] for path in mask_paths]
    mask_search = [
        file.split(file.split(".")[-1])[0][:-1] for file in mask_files
    ]

    layer_names = [
        layer.name
        for layer in self.viewer.layers
        if layer.name not in ["Segmentations", "Classes", "center_lines"]
    ]

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
            from napari_bacseg._utils_json import import_coco_json

            mask, label = import_coco_json(mask_path)
            mask_stack[i, :, :][y1:y2, x1:x2] = mask
            class_stack[i, :, :][y1:y2, x1:x2] = label

            self.segLayer.data = mask_stack.astype(np.uint16)
            self.classLayer.data = class_stack.astype(np.uint16)

        if file_format == "npy":
            dat = np.load(mask_path, allow_pickle=True).item()

            mask = dat["masks"]
            mask = mask.astype(np.uint16)
            mask_stack[i, :, :][y1:y2, x1:x2] = mask
            self.segLayer.data = mask_stack.astype(np.uint16)

        if file_format == "mat":
            image, mask, meta = import_mat_data(self, image_path, mask_path)
            mask_stack[i, :, :][y1:y2, x1:x2] = mask
            self.segLayer.data = mask_stack.astype(np.uint16)


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


def get_contours_from_mask(mask, label, export_labels):
    export_mask = np.zeros(mask.shape, dtype=np.uint16)
    export_label = np.zeros(mask.shape, dtype=np.uint16)

    contours = []

    mask_ids = np.unique(mask)

    for mask_id in mask_ids:
        try:
            if mask_id != 0:
                cnt_mask = np.zeros(mask.shape, dtype=np.uint8)

                cnt_mask[mask == mask_id] = 255
                label_id = np.unique(label[cnt_mask == 255])[0]

                if label_id in export_labels:
                    new_mask_id = np.max(np.unique(export_mask)) + 1
                    export_mask[cnt_mask == 255] = new_mask_id
                    export_label[cnt_mask == 255] = label_id

                    cnt, _ = cv2.findContours(
                        cnt_mask.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_NONE,
                    )

                    contours.append(cnt[0])

        except:
            pass

    return contours


def automatic_brightness_and_contrast(image, clip_hist_percent=0.1):
    if np.max(image) > 0:
        # Calculate grayscale histogram
        hist = cv2.calcHist([image], [0], None, [2**16], [0, 2**16])
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


def add_scale_bar(image, pixel_resolution=100, pixel_resolution_units="nm", scalebar_size=20, scalebar_size_units="um", scalebar_colour="white", scalebar_thickness=10, scalebar_margin=50):

    try:

        if float(pixel_resolution) > 0 and float(scalebar_size) > 0:

            h, w = image.shape

            pixel_resolution = float(pixel_resolution)
            scalebar_size = float(scalebar_size)

            if pixel_resolution_units != "nm":
                pixel_resolution = pixel_resolution * 1000

            if scalebar_size_units != "nm":
                scalebar_size = scalebar_size * 1000

            scalebar_len = int(scalebar_size / pixel_resolution)

            if scalebar_len > 0 and scalebar_len < w:

                if scalebar_colour == "White":
                    bit_depth = str(image.dtype)
                    bit_depth = int(bit_depth.replace("uint", ""))
                    colour = ((2 ** bit_depth) - 1)

                scalebar_pos = (w - scalebar_margin - scalebar_len, h - scalebar_margin)  # Position of the scale bar in the image (in pixels)

            image = cv2.rectangle(image, scalebar_pos, (scalebar_pos[0] + scalebar_len, scalebar_pos[1] + int(scalebar_thickness)), colour, -1)

    except:
        print(traceback.format_exc())
        pass

    return image


def generate_export_image(
    self,
    export_channel,
    dim,
    normalize=False,
    invert=False,
    autocontrast=False,
    scalebar = False
):
    layer_names = [
        layer.name
        for layer in self.viewer.layers
        if layer.name not in ["Segmentations", "Classes", "center_lines"]
    ]

    layer_names.reverse()

    if export_channel == "All Channels (Stack)":
        mode = "stack"
    elif export_channel == "First Three Channels (RGB)":
        mode = "rgb"
    else:
        mode = "single"
        layer_names = [export_channel]

    if mode == "rgb":
        layer_names = layer_names[:3]

    image = []

    for layer in layer_names:
        img = self.viewer.layers[layer].data

        img = img[dim]

        if invert:
            img = cv2.bitwise_not(img)

        if normalize:
            img = normalize99(img)

        if autocontrast:
            img = automatic_brightness_and_contrast(img)

        if scalebar:

            pixel_resolution = self.export_scalebar_resolution.text()
            pixel_resolution_units = self.export_scalebar_resolution_units.currentText()
            scalebar_size = self.export_scalebar_size.text()
            scalebar_size_units = self.export_scalebar_size_units.currentText()
            scalebar_colour = self.export_scalebar_colour.currentText()
            scalebar_thickness = self.export_scalebar_thickness.currentText()

            img = add_scale_bar(img,
            pixel_resolution = pixel_resolution,
            pixel_resolution_units = pixel_resolution_units,
            scalebar_size = scalebar_size,
            scalebar_size_units = scalebar_size_units,
            scalebar_colour = scalebar_colour,
            scalebar_thickness = scalebar_thickness)

        image.append(img)

    if mode == "rgb":
        while len(image) < 3:
            image.append(np.zeros(img.shape))

    if mode == "rgb":
        image = np.stack(image, axis=-1)
    elif mode == "stack":
        image = np.stack(image, axis=0)

    image = rescale01(image)
    image = image * (2**16 - 1)
    image = image.astype(np.uint16)

    mask = self.segLayer.data
    label = self.classLayer.data
    metadata = self.viewer.layers[layer_names[0]].metadata

    mask = mask[dim]
    label = label[dim]
    metadata = metadata[dim[0]]

    return image, mask, label, metadata, mode


def export_files(self, progress_callback, mode):

    desktop = os.path.expanduser("~/Desktop")

    overwrite = self.export_overwrite_setting.isChecked()
    export_images = self.export_image_setting.isChecked()
    normalise = self.export_normalise.isChecked()
    invert = self.export_invert.isChecked()
    autocontrast = self.export_autocontrast.isChecked()
    scalebar = self.export_scalebar.isChecked()

    export_channel = self.export_channel.currentText()
    export_modifier = self.export_modifier.text()
    export_labels = get_export_labels(self)

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
        image, mask, label, meta, mode = generate_export_image(
            self, export_channel, dim, normalise, invert, autocontrast, scalebar
        )
        contours = get_contours_from_mask(mask, label, export_labels)

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

        file_name, file_extension = os.path.splitext(file_name)

        if len(dim) == 2:
            file_name = file_name + f"_{dim}"

        file_name = file_name + export_modifier + ".tif"
        image_path = image_path.replace(image_path.split("\\")[-1], file_name)

        if (
            self.export_location.currentText() == "Import Directory"
            and file_name != None
            and image_path != None
        ):
            export_path = os.path.abspath(image_path.replace(file_name, ""))

        elif self.export_location.currentText() == "Select Directory":
            export_path = os.path.abspath(self.export_directory)

        else:
            export_path = None

        if os.path.isdir(export_path) != True:
            if self.widget_notifications:
                show_info(
                    "Directory does not exist, try selecting a directory instead!"
                )

        else:
            y1, y2, x1, x2 = meta["crop"]

            if len(image.shape) > 2:
                image = image[:, y1:y2, x1:x2]
            else:
                image = image[y1:y2, x1:x2]

            mask = mask[y1:y2, x1:x2]
            label = label[y1:y2, x1:x2]

            if os.path.isdir(export_path) == False:
                os.makedirs(file_path)

            file_path = export_path + "\\" + file_name

            if os.path.isfile(file_path) == True and overwrite == False:
                if self.widget_notifications:
                    show_info(
                        file_name
                        + " already exists, BacSeg will not overwrite files!"
                    )

            else:
                if self.export_mode.currentText() == "Export .tif Images":
                    tifffile.imwrite(file_path, image, metadata=meta)

                if self.export_mode.currentText() == "Export .tif Masks":
                    tifffile.imwrite(file_path, mask, metadata=meta)

                if (
                    self.export_mode.currentText()
                    == "Export .tif Images and Masks"
                ):
                    image_path = os.path.abspath(export_path + "\\images")
                    mask_path = os.path.abspath(export_path + "\\masks")

                    if not os.path.exists(image_path):
                        os.makedirs(image_path)

                    if not os.path.exists(mask_path):
                        os.makedirs(mask_path)

                    image_path = os.path.abspath(image_path + "\\" + file_name)
                    mask_path = os.path.abspath(mask_path + "\\" + file_name)

                    tifffile.imwrite(image_path, image, metadata=meta)
                    tifffile.imwrite(mask_path, mask, metadata=meta)

                if self.export_mode.currentText() == "Export Cellpose":
                    from napari_bacseg._utils_cellpose import export_cellpose

                    export_cellpose(file_path, image, mask)

                    if export_images:
                        tifffile.imwrite(file_path, image, metadata=meta)

                if self.export_mode.currentText() == "Export Oufti":
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")

                            from napari_bacseg._utils_oufti import (
                                export_oufti,
                                get_oufti_data,
                            )

                            oufti_data = get_oufti_data(
                                self, image, mask, midlines
                            )

                            if "midlines" in meta.keys():
                                meta.pop("midlines")

                            export_oufti(image, oufti_data, file_path)

                            if export_images:
                                tifffile.imwrite(
                                    file_path, image, metadata=meta
                                )

                    except:
                        raise Exception(
                            "BacSeg can't load Cellpose and OUFTI dependencies simultaneously. Restart BacSeg, reload images/masks, then export Oufti"
                        )

                if self.export_mode.currentText() == "Export ImageJ":
                    from napari_bacseg._utils_imagej import export_imagej

                    if mode == "rgb":
                        if self.widget_notifications:
                            show_info(
                                "ImageJ can't handle RGB images with annotations, export as image stack instead..."
                            )

                    export_imagej(image, contours, meta, file_path)

                if self.export_mode.currentText() == "Export JSON":
                    from napari_bacseg._utils_json import export_coco_json

                    export_coco_json(file_name, image, mask, label, file_path)

                    if export_images:
                        tifffile.imwrite(file_path, image, metadata=meta)

                if self.export_mode.currentText() == "Export CSV":
                    export_csv(image, contours, meta, file_path)

                    if export_images:
                        tifffile.imwrite(file_path, image, metadata=meta)

        progress = int(((i + 1) / len(dim_list)) * 100)
        try:
            progress_callback.emit(progress)
        except:
            pass


def export_csv(image, contours, meta, file_path):
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
                    cnt = np.pad(
                        cnt,
                        ((0, max_length - cnt.shape[0]), (0, 0)),
                        "constant",
                        constant_values="",
                    )

                processed_contours.append(cnt)

            except:
                pass

        try:
            file_extension = file_path.split(".")[-1]
            file_path = file_path.replace(file_extension, "csv")

            processed_contours = np.hstack(processed_contours)
            headers = np.array(
                [
                    [f"x[{str(x)}]", f"y[{str((x))}]"]
                    for x in range(processed_contours.shape[-1] // 2)
                ]
            ).flatten()

            pd.DataFrame(processed_contours, columns=headers).to_csv(
                file_path, index=False, header=True
            )

        except:
            print(traceback.format_exc())


def _manualImport(self):
    try:
        if (
            self.viewer.layers.index("Segmentations")
            != len(self.viewer.layers) - 1
        ):
            # reshapes masks to be same shape as active image
            self.active_layer = self.viewer.layers[-1]

            if self.active_layer.metadata == {}:
                active_image = self.active_layer.data

                if len(active_image.shape) < 3:
                    active_image = np.expand_dims(active_image, axis=0)
                    self.active_layer.data = active_image

                if self.classLayer.data.shape != self.active_layer.data.shape:
                    self.classLayer.data = np.zeros(
                        active_image.shape, np.uint16
                    )

                if self.segLayer.data.shape != self.active_layer.data.shape:
                    self.segLayer.data = np.zeros(
                        active_image.shape, np.uint16
                    )

                image_name = str(self.viewer.layers[-1]) + ".tif"

                meta = {}
                for i in range(active_image.shape[0]):
                    img = active_image[i, :, :]

                    contrast_limit, alpha, beta, gamma = autocontrast_values(
                        img, clip_hist_percent=1
                    )

                    img_meta = dict(
                        image_name=image_name,
                        image_path="Unknown",
                        mask_name=None,
                        mask_path=None,
                        label_name=None,
                        label_path=None,
                        folder=None,
                        parent_folder=None,
                        contrast_limit=contrast_limit,
                        contrast_alpha=alpha,
                        contrast_beta=beta,
                        contrast_gamma=gamma,
                        akseg_hash=None,
                        import_mode="manual",
                        dims=[img.shape[1], img.shape[0]],
                        crop=[0, img.shape[0], 0, img.shape[1]],
                        frame=i,
                        frames=active_image.shape[0],
                    )

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
