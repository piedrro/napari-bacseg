import os
import traceback
import numpy as np
import pandas as pd
import xmltodict
from glob2 import glob
from napari.utils.notifications import show_info

class _olympus_utils:

    def read_xml(self, paths):
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

                        objective_id = int(img["ObjectiveSettings"]["@ID"].split(":")[-1])
                        objective_dat = dat["Instrument"]["Objective"][objective_id]
                        objective_mag = float(objective_dat["@NominalMagnification"])
                        objective_na = float(objective_dat["@LensNA"])

                        pixel_size = float(img["Pixels"]["@PhysicalSizeX"])

                        # print(pixel_size)

                        position_index = i
                        microscope = "ScanR"
                        light_source = "LED"

                        channel_dict = {}

                        for j in range(len(img["Pixels"]["Channel"])):
                            channel_data = img["Pixels"]["Channel"][j]

                            channel_dict[j] = dict(modality=channel_data["@IlluminationType"], channel=channel_data["@Name"], mode=channel_data["@AcquisitionMode"], well=
                            channel_data["@ID"].split("W")[1].split("P")[0], )

                        primary_channel = ""

                        for j in range(len(img["Pixels"]["TiffData"])):
                            num_channels = img["Pixels"]["@SizeC"]
                            num_zstack = img["Pixels"]["@SizeZ"]

                            tiff_data = img["Pixels"]["TiffData"][j]

                            file_name = tiff_data["UUID"]["@FileName"]
                            file_path = os.path.abspath(path.replace(os.path.basename(path), file_name))

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

                            files[
                                file_path] = dict(file_name=file_name, well_index=well_index, position_index=position_index, channel_index=channel_index, time_index=time_index, z_index=z_index, microscope=microscope, light_source=light_source, channel=channel, modality=modality, pixel_size=pixel_size, objective_magnification=objective_mag, objective_na=objective_na, exposure_time=exposure_time, posX=posX, posY=posY, posZ=posZ, )
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

