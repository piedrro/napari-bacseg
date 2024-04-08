import copy
import itertools
import os
from collections import ChainMap
import numpy as np
import pandas as pd
from aicspylibczi import CziFile

class _zeiss_utils:

    def get_ziess_channel_dict(self, path):

        import xmltodict
        from czifile import CziFile

        czi = CziFile(path)
        metadata = czi.metadata()

        metadata = xmltodict.parse(metadata)["ImageDocument"]["Metadata"]

        channels_metadata = metadata["Information"]["Image"]["Dimensions"]["Channels"]["Channel"]

        channel_dict = {}

        for channel_name, channel_meta in enumerate(channels_metadata):
            channel_dict[channel_name] = {}

            for key, value in channel_meta.items():
                if key == "@Name":
                    if value == "Bright":
                        value = "Bright Field"
                    if value == "Phase":
                        value = "Phase Contrast"
                    if value == "nilRe":
                        value = "Nile Red"

                channel_dict[channel_name][key] = value

        return channel_dict


    def get_czi_dim_list(self, path):
        czi = CziFile(path)

        img_dims = czi.dims
        img_dims_shape = czi.get_dims_shape()
        img_size = czi.size
        pixel_type = czi.pixel_type

        index_dims = []

        for index_name in ["S", "T", "M", "Z", "C"]:
            if index_name in img_dims_shape[0].keys():
                index_shape = img_dims_shape[0][index_name][-1]

                dim_list = np.arange(index_shape).tolist()
                dim_list = [{index_name: dim} for dim in dim_list]

                index_dims.append(dim_list)

        dim_list = list(itertools.product(*index_dims))
        dim_list = [dict(ChainMap(*list(dim))) for dim in dim_list]

        for dim in dim_list:
            dim.update({"path": path})

        dim_list = pd.DataFrame(dim_list)

        return dim_list


    def get_zeiss_measurements(self, paths):
        import_limit = self.import_limit.currentText()

        czi_measurements = []

        for path in paths:
            if os.path.exists(path):
                dim_list = self.get_czi_dim_list(path)

                czi_measurements.append(dim_list)

        czi_measurements = pd.concat(czi_measurements)

        groupby_columns = czi_measurements.drop(["C"], axis=1).columns.tolist()

        if len(groupby_columns) == 1:
            groupby_columns = groupby_columns[0]

        czi_fovs = []

        for group, data in czi_measurements.groupby(groupby_columns):
            czi_fovs.append(data)

        if import_limit != "None":
            import_limit = int(import_limit)

            czi_fovs = czi_fovs[:import_limit]
            num_measurements = len(czi_fovs)
            czi_measurements = pd.concat(czi_fovs)

        else:
            num_measurements = len(czi_fovs)
            czi_measurements = pd.concat(czi_fovs)

        channel_names = []

        for path in paths:
            channel_dict = self.get_ziess_channel_dict(path)

            for key, value in channel_dict.items():
                channel_names.append(value["@Name"])

        channel_names = np.unique(channel_names)

        return czi_measurements, channel_names, num_measurements


    def read_zeiss_image_files(self, progress_callback, zeiss_measurements, channel_names, num_measurements, ):

        from aicspylibczi import CziFile

        zeiss_images = {}

        import_limit = self.import_limit.currentText()

        num_loaded = 0
        img_index = {}

        for path, dim_list in zeiss_measurements.groupby("path"):
            path = os.path.normpath(path)

            dim_list = dim_list.drop("path", axis=1).dropna(axis=1)

            czi = CziFile(path)
            channel_dict = self.get_ziess_channel_dict(path)

            key_dim_cols = dim_list.columns.tolist()
            key_dim_cols = dim_list.columns.drop(["C"]).tolist()

            if key_dim_cols == []:
                images, img_shape = czi.read_image()

                num_loaded += 1
                progress = int(((num_loaded) / num_measurements) * 100)
                progress_callback.emit(progress)

                fov_channels = []

                for channel_index, img_channel in enumerate(images):

                    akseg_hash = self.get_hash(img=img_channel)

                    contrast_limit = np.percentile(img_channel, (1, 99))
                    contrast_limit = [int(contrast_limit[0] * 0.5), int(contrast_limit[1] * 2), ]

                    meta = copy.deepcopy(channel_dict[channel_index])

                    image_name = os.path.basename(path).replace(".czi", "")
                    image_name = image_name + "_" + meta["@Name"].replace(" ", "")

                    meta["akseg_hash"] = akseg_hash
                    meta["image_name"] = image_name
                    meta["image_path"] = path
                    meta["folder"] = path.split(os.sep)[-2]
                    meta["mask_name"] = None
                    meta["mask_path"] = None
                    meta["label_name"] = None
                    meta["label_path"] = None
                    meta["import_mode"] = "image"
                    meta["contrast_limit"] = contrast_limit
                    meta["contrast_alpha"] = 0
                    meta["contrast_beta"] = 0
                    meta["contrast_gamma"] = 0
                    meta["dims"] = [img_channel.shape[-1], img_channel.shape[-2]]
                    meta["crop"] = [0, img_channel.shape[-2], 0, img_channel.shape[-1], ]

                    channel_name = meta["@Name"]

                    if channel_name not in img_index.keys():
                        img_index[channel_name] = 0

                    fov_channels.append(channel_name)

                    if channel_name not in zeiss_images:
                        zeiss_images[channel_name] = dict(images=[img_channel], masks=[], nmasks=[], classes=[], metadata={img_index[channel_name]: meta}, )
                    else:
                        zeiss_images[channel_name]["images"].append(img_channel)
                        zeiss_images[channel_name]["metadata"][img_index[channel_name]] = meta

                    img_index[channel_name] += 1

                missing_channels = [channel for channel in channel_names if channel not in fov_channels]

                for channel_name in missing_channels:

                    img_channel = np.zeros_like(img_channel)

                    meta = {}
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
                    meta["dims"] = [img_channel.shape[-1], img_channel.shape[-2]]
                    meta["crop"] = [0, img_channel.shape[-2], 0, img_channel.shape[-1], ]
                    meta["light_source"] = channel_name

                    if channel_name not in img_index.keys():
                        img_index[channel_name] = 0

                    if channel_name not in zeiss_images:
                        zeiss_images[channel_name] = dict(images=[img_channel], masks=[], nmasks=[], classes=[], metadata={img_index[channel_name]: {}}, )
                    else:
                        zeiss_images[channel_name]["images"].append(img_channel)
                        zeiss_images[channel_name]["metadata"][img_index[channel_name]] = meta

                    img_index[channel_name] += 1
            else:
                iter = 0

                for i, (_, data) in enumerate(dim_list.groupby(key_dim_cols)):
                    data = data.reset_index(drop=True).dropna().astype(int)

                    num_loaded += 1
                    progress = int(((num_loaded) / num_measurements) * 100)
                    progress_callback.emit(progress)

                    fov_channels = []

                    for channel_index, czi_indeces in data.iterrows():

                        czi_indeces = czi_indeces.to_dict()

                        img, img_shape = czi.read_image(**czi_indeces)

                        img_channel = img.reshape(img.shape[-2:])

                        akseg_hash = self.get_hash(img=img_channel)

                        contrast_limit = np.percentile(img_channel, (1, 99))
                        contrast_limit = [int(contrast_limit[0] * 0.5), int(contrast_limit[1] * 2), ]

                        meta = copy.deepcopy(channel_dict[channel_index])

                        image_name = os.path.basename(path).replace(".czi", "")

                        for key, value in czi_indeces.items():
                            image_name = image_name + "_" + str(key) + str(value)

                        image_name = (image_name + "_" + meta["@Name"].replace(" ", "") + ".tif")

                        meta["akseg_hash"] = akseg_hash
                        meta["image_name"] = copy.deepcopy(image_name)
                        meta["image_path"] = path
                        meta["folder"] = path.split(os.sep)[-2]
                        meta["mask_name"] = None
                        meta["mask_path"] = None
                        meta["label_name"] = None
                        meta["label_path"] = None
                        meta["import_mode"] = "image"
                        meta["contrast_limit"] = contrast_limit
                        meta["contrast_alpha"] = 0
                        meta["contrast_beta"] = 0
                        meta["contrast_gamma"] = 0
                        meta["dims"] = [img_channel.shape[-1], img_channel.shape[-2], ]
                        meta["crop"] = [0, img_channel.shape[-2], 0, img_channel.shape[-1], ]

                        channel_name = copy.deepcopy(meta["@Name"])

                        if channel_name not in img_index.keys():
                            img_index[channel_name] = 0

                        fov_channels.append(channel_name)

                        if channel_name not in zeiss_images.keys():
                            zeiss_images[channel_name] = dict(images=[img_channel], masks=[], nmasks=[], classes=[], metadata={img_index[channel_name]: meta}, )
                        else:
                            zeiss_images[channel_name]["images"].append(img_channel)
                            zeiss_images[channel_name]["metadata"][img_index[channel_name]] = meta

                        img_index[channel_name] += 1

                    missing_channels = [channel for channel in channel_names if channel not in fov_channels]

                    for channel_name in missing_channels:
                        img_channel = np.zeros_like(img_channel)

                        meta = {}
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
                        meta["dims"] = [img_channel.shape[-1], img_channel.shape[-2], ]
                        meta["crop"] = [0, img_channel.shape[-2], 0, img_channel.shape[-1], ]
                        meta["light_source"] = channel_name

                        if channel_name not in img_index.keys():
                            img_index[channel_name] = 0

                        if channel_name not in zeiss_images:
                            zeiss_images[channel_name] = dict(images=[img_channel], masks=[], nmasks=[], classes=[], metadata={img_index[channel_name]: meta}, )
                        else:
                            zeiss_images[channel_name]["images"].append(img_channel)
                            zeiss_images[channel_name]["metadata"][img_index[channel_name]] = meta

                        img_index[channel_name] += 1

        imported_data = dict(imported_images=zeiss_images)

        return imported_data
