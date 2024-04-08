import copy
import json
import os
import numpy as np
import pandas as pd
import tifffile
from glob2 import glob
from napari.utils.notifications import show_info

class _oni_utils:

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

