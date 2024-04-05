
from napari.utils.notifications import show_info
import os
import traceback
from qtpy.QtWidgets import QFileDialog
import pandas as pd
import numpy as np
from functools import partial
import time
from napari_bacseg.funcs.threading_utils import Worker


class _picasso_utils:

    def export_picasso_localisations(self, event=None):
        try:
            if (hasattr(self, "detected_locs") == False and hasattr(self, "fitted_locs") == False):
                show_info("No localisations found, please detect + fit localisations first")
            elif (hasattr(self, "detected_locs") == True and hasattr(self, "fitted_locs") == False):
                show_info("No fitted localisations found, please fit localisations first")
            elif (hasattr(self, "detected_locs") == True and hasattr(self, "fitted_locs") == True):
                desktop = os.path.expanduser("~/Desktop")

                export_mode = self.picasso_export_mode.currentText()

                if export_mode == ".csv":
                    export_extension = "csv"
                elif export_mode == "Picasso HDF5":
                    export_extension = "hdf5"
                elif export_mode == ".pos.out":
                    export_extension = "pos.out"

                import_path = self.viewer.layers[self.picasso_image_channel.currentText()].metadata[0]["image_path"]
                # replace extension with export extension
                import_path = os.path.splitext(import_path)[0] + "." + export_extension


                export_path = QFileDialog.getSaveFileName(self, "Select Save Directory", import_path, f"(*.{export_extension})")[0]
                export_directory = os.path.dirname(export_path)

                if os.path.isdir(export_directory):
                    image_layers = [layer.name for layer in self.viewer.layers]
                    image_channel = self.picasso_image_channel.currentText()

                    if image_channel in image_layers:

                        image_name = os.path.basename(export_path)

                        image_path = self.viewer.layers[image_channel].metadata[0]["image_path"]

                        localisation_data = self.get_localisation_data(self.fitted_locs)

                        if export_mode == ".csv":

                            def replace_image_name(dat, image_names):
                                frame_index = dat["frame"]
                                image_name = image_names[frame_index]
                                dat["image_name"] = image_name
                                return dat

                            channel_metadata = self.viewer.layers[image_channel].metadata
                            frame_list = localisation_data["frame"].unique()
                            image_names = [channel_metadata[frame]["image_name"] for frame in frame_list]

                            localisation_data.insert(0, "image_name", "")
                            localisation_data = localisation_data.apply(lambda x: replace_image_name(x, image_names), axis=1)

                            export_path = os.path.join(export_directory, image_name)
                            export_path = os.path.abspath(export_path)

                            localisation_data = pd.DataFrame(localisation_data)

                            localisation_data.sort_values(by=["frame", "cell_index"], inplace=True)

                            localisation_data.to_csv(export_path, index=False)

                            show_info("Picasso localisations exported to: " + export_path)

                        elif export_mode == ".pos.out":
                            export_path = os.path.join(export_directory, image_name)
                            export_path = os.path.abspath(export_path)

                            st_locs = localisation_data[["frame", "x", "y", "photons", "bg", "sx", "sy", ]].copy()

                            st_locs.dropna(axis=0, inplace=True)

                            st_locs.columns = ["FRAME", "XCENTER", "YCENTER", "BRIGHTNESS", "BG", "S_X", "S_Y", ]

                            st_locs.loc[:, "I0"] = 0
                            st_locs.loc[:, "THETA"] = 0
                            st_locs.loc[:, "ECC"] = (st_locs["S_X"] / st_locs["S_Y"])
                            st_locs.loc[:, "FRAME"] = st_locs["FRAME"] + 1

                            st_locs = st_locs[["FRAME", "XCENTER", "YCENTER", "BRIGHTNESS", "BG", "I0", "S_X", "S_Y", "THETA", "ECC", ]]

                            st_locs.to_csv(export_path, sep="\t", index=False)

                            show_info("Picasso localisations exported to: " + export_path)

                        elif export_mode == "Picasso HDF5":
                            # get fil
                            hdf5_path = os.path.join(export_directory, image_name)
                            yaml_path = os.path.join(export_directory, image_name.replace(".hdf5",".yaml"))
                            hdf5_path = os.path.abspath(hdf5_path)
                            yaml_path = os.path.abspath(yaml_path)

                            localisation_data = pd.DataFrame(localisation_data)

                            columns = ["frame", "x", "y", "photons", "sx", "sy", "bg", "lpx", "lpy", "ellipticity", "net_gradient", ]
                            localisation_data = localisation_data[columns]

                            import h5py
                            import yaml

                            structured_arr = localisation_data.to_records(index=False)

                            # Create a new HDF5 file (or open an existing one).
                            with h5py.File(hdf5_path, "w") as hdf_file:
                                # Create a dataset within the file, named 'locs'.
                                dataset = hdf_file.create_dataset("locs", data=structured_arr)

                            picasso_channel = (self.picasso_image_channel.currentText())
                            image_shape = self.viewer.layers[picasso_channel].data.shape
                            min_net_gradient = int(self.picasso_min_net_gradient.text())
                            box_size = int(self.picasso_box_size.currentText())

                            data = {"Byte Order": "<", "Data Type": "uint16", "File": image_path, "Frames": image_shape[0], "Height": image_shape[1], "Micro-Manager Acquisiton Comments": "", "Width":
                                image_shape[2], }

                            data2 = {"Box Size": box_size, "Fit method": "LQ, Gaussian", "Generated by": "Picasso Localize", "Min. Net Gradient": min_net_gradient, "Pixelsize": 130, "ROI": None, }

                            # Write YAML file with "---" as a separator between documents
                            with open(yaml_path, "w") as file:
                                yaml.dump(data, file, default_flow_style=False)
                                file.write("---\n")  # Document separator
                                yaml.dump(data2, file, default_flow_style=False)

                            show_info("Picasso localisations exported to: " + hdf5_path)

        except:
            print(traceback.format_exc())

    def get_localisation_data(self, locs):
        import pandas as pd

        try:
            param_list = ["frame", "cell_index", "x", "y", "photons", "sx", "sy", "bg", "lpx", "lpy", "ellipticity", "net_gradient", "z", "d_zcalib", "likelihood", "iterations", "group", "len", "n",
                          "photon_rate", ]

            loc_data = []

            for loc in locs:
                loc_dict = {}

                for param_name in param_list:
                    if hasattr(loc, param_name):
                        loc_dict[param_name] = getattr(loc, param_name)

                loc_data.append(loc_dict)

            loc_data = pd.DataFrame(loc_data)

        except:
            print(traceback.format_exc())
            loc_data = []

        return loc_data

    def get_localisation_centres(self, locs):
        try:
            loc_centres = []
            for loc in locs:
                frame = int(loc.frame)
                # if frame not in loc_centres.keys():
                #     loc_centres[frame] = []
                loc_centres.append([frame, loc.y, loc.x])

        except:
            print(traceback.format_exc())
            loc_centres = []

        return loc_centres

    def _fit_localisations_cleanup(self):
        try:
            self._Progresbar(100, "picasso")

            self.localisation_centres = self.get_localisation_centres(self.fitted_locs)

            self._reorderLayers()
            self.display_localisations()

            # select picasso channel

            picasso_channel = self.picasso_image_channel.currentText()
            self.viewer.layers.selection.active = self.viewer.layers[picasso_channel]

            num_frames = len(np.unique([loc[0] for loc in self.localisation_centres]))
            num_locs = len(self.detected_locs)

            show_info(f"Picasso Fitted {num_locs} localisations in {num_frames} frame(s)")

        except:
            pass

    def _fit_localisations(self, progress_callback=None, min_net_gradient=100, box_size=3, camera_info={}, image_channel="", method="lq", gain=1, use_gpufit=False, ):
        try:
            from picasso import gausslq, lib, localize

            image_frames = self.picasso_image_frames.currentText()
            n_detected_frames = len(np.unique([loc[0] for loc in self.localisation_centres]))

            if image_frames.lower() == "active":
                image_data = self.viewer.layers[image_channel].data[self.viewer.dims.current_step[0]]
                image_data = np.expand_dims(image_data, axis=0)
            else:
                image_data = self.viewer.layers[image_channel].data

            if n_detected_frames != image_data.shape[0]:
                show_info("Picasso can only Detect AND Fit localisations with same image frame mode")
            else:
                detected_loc_spots = localize.get_spots(image_data, self.detected_locs, box_size, camera_info)

                show_info(f"Picasso fitting {len(self.detected_locs)} spots...")

                if method == "lq":
                    if use_gpufit:
                        theta = gausslq.fit_spots_gpufit(detected_loc_spots)
                        em = gain > 1
                        self.fitted_locs = gausslq.locs_from_fits_gpufit(self.detected_locs, theta, box_size, em)
                    else:
                        fs = gausslq.fit_spots_parallel(detected_loc_spots, asynch=True)

                        n_tasks = len(fs)
                        while lib.n_futures_done(fs) < n_tasks:
                            progress = (lib.n_futures_done(fs) / n_tasks) * 100
                            progress_callback.emit(progress)
                            time.sleep(0.1)

                        theta = gausslq.fits_from_futures(fs)
                        em = gain > 1
                        self.fitted_locs = gausslq.locs_from_fits(self.detected_locs, theta, box_size, em)

                show_info(f"Picasso fitted {len(self.fitted_locs)} spots")

                self.fitted_locs = self.process_localisations(self.fitted_locs)

        except:
            print(traceback.format_exc())

    def _detect_localisations(self, progress_callback=None, min_net_gradient=100, box_size=3, camera_info={}, image_channel="", ):
        self.detected_locs = []
        self.localisation_centres = {}

        try:
            from picasso import localize

            min_net_gradient = int(min_net_gradient)

            image_frames = self.picasso_image_frames.currentText()

            if image_frames.lower() == "active":
                image_data = self.viewer.layers[image_channel].data[self.viewer.dims.current_step[0]]
                image_data = np.expand_dims(image_data, axis=0)
            else:
                image_data = self.viewer.layers[image_channel].data

            curr, futures = localize.identify_async(image_data, min_net_gradient, box_size, roi=None)
            self.detected_locs = localize.identifications_from_futures(futures)

            if image_frames.lower() == "active":
                for loc in self.detected_locs:
                    loc.frame = self.viewer.dims.current_step[0]

            self.detected_locs = self.process_localisations(self.detected_locs)

        except:
            print(traceback.format_exc())

        return self.detected_locs

    def process_localisations(self, locs):
        filter = self.picasso_filter_localisations.isChecked()

        processed_locs = []

        try:
            loc_centres = {}

            for index, loc in enumerate(locs):
                frame = loc.frame
                if frame not in loc_centres.keys():
                    loc_centres[frame] = []
                loc_centres[frame].append([index, loc, loc.x, loc.y])

            for frame, loc_centres in loc_centres.items():
                mask = self.segLayer.data[frame]

                unique_mask_values = len(np.unique(mask))

                if unique_mask_values > 1 or filter == False:
                    for loc_dat in loc_centres:
                        index, loc, x, y = loc_dat

                        # Ensure the coordinates are within the boundaries of the instance mask
                        if (x < 0 or y < 0 or x >= mask.shape[1] or y >= mask.shape[0]):
                            inside = False

                        else:
                            # Query the value at the given coordinates in the instance mask
                            mask_value = mask[int(y), int(x)]

                            # If the value is not 0, then the point is inside an instance
                            inside = mask_value != 0

                            if mask_value == 0:
                                mask_value = -1

                            # append new field to dtype
                            dtype = np.dtype(loc.dtype.descr + [('cell_index', '<f4')])

                            appended_loc = np.zeros(1, dtype=dtype)[0]

                            for field in loc.dtype.names:
                                appended_loc[field] = loc[field].copy()

                            appended_loc['cell_index'] = mask_value
                            appended_loc = appended_loc.view(np.recarray)

                            if filter == False:
                                processed_locs.append(appended_loc)
                            elif filter == True and mask_value != -1:
                                processed_locs.append(appended_loc)

            processed_locs = np.array(processed_locs).view(np.recarray)

            n_removed = len(locs) - len(processed_locs)

            if filter:
                show_info(f"Picasso removed {n_removed} localisations outside of segmentations")

        except:
            print(traceback.format_exc())

        return processed_locs

    def _detect_localisations_cleanup(self):
        try:
            self.localisation_centres = self.get_localisation_centres(self.detected_locs)

            self._reorderLayers()
            self.display_localisations()

            num_frames = len(np.unique([loc[0] for loc in self.localisation_centres]))
            num_locs = len(self.detected_locs)

            show_info(f"Picasso Detected {num_locs} localisations in {num_frames} frame(s)")

            picasso_channel = self.picasso_image_channel.currentText()

            self._reorderLayers()
            self.viewer.layers.selection.active = self.viewer.layers[picasso_channel]

        except:
            print(traceback.format_exc())

    def update_localisation_visualisation(self):
        try:
            if self.picasso_show_vis.isChecked():
                self.display_localisations()
            else:
                if "Localisations" in [layer.name for layer in self.viewer.layers]:
                    self.viewer.layers["Localisations"].data = []
        except:
            print(traceback.format_exc())

    def display_localisations(self):
        if (hasattr(self, "localisation_centres") and self.picasso_show_vis.isChecked()):
            try:
                layer_names = [layer.name for layer in self.viewer.layers]

                vis_mode = self.picasso_vis_mode.currentText()
                vis_size = float(self.picasso_vis_size.currentText())
                vis_opacity = float(self.picasso_vis_opacity.currentText())
                vis_edge_width = float(self.picasso_vis_edge_width.currentText())

                if vis_mode.lower() == "square":
                    symbol = "square"
                elif vis_mode.lower() == "disk":
                    symbol = "disc"
                elif vis_mode.lower() == "x":
                    symbol = "cross"

                box_centres = self.localisation_centres.copy()

                if len(box_centres) > 0:
                    if "Localisations" not in layer_names:
                        self.viewer.add_points(box_centres, edge_color="blue", face_color=[0, 0, 0,
                                                                                           0], opacity=vis_opacity, name="Localisations", symbol=symbol, size=vis_size, edge_width=vis_edge_width, )
                    else:
                        self.viewer.layers["Localisations"].data = []

                        self.viewer.layers["Localisations"].data = box_centres
                        self.viewer.layers["Localisations"].symbol = symbol
                        self.viewer.layers["Localisations"].size = vis_size
                        self.viewer.layers["Localisations"].opacity = vis_opacity
                        self.viewer.layers["Localisations"].edge_width = vis_edge_width
                        self.viewer.layers["Localisations"].edge_color = "blue"

                else:
                    if "Localisations" in layer_names:
                        self.viewer.layers["Localisations"].data = []

            except:
                print(traceback.format_exc())

    def detect_picasso_localisations(self):
        try:
            image_channel = self.picasso_image_channel.currentText()
            box_size = int(self.picasso_box_size.currentText())
            min_net_gradient = self.picasso_min_net_gradient.text()

            camera_info = {"baseline": 100.0, "gain": 1, "sensitivity": 1.0, "qe": 0.9, }

            # check min net gradient is a number
            if min_net_gradient.isdigit() and image_channel != "":
                worker = Worker(self._detect_localisations, min_net_gradient=min_net_gradient, box_size=box_size, camera_info=camera_info, image_channel=image_channel, )
                worker.signals.finished.connect(self._detect_localisations_cleanup)
                self.threadpool.start(worker)

        except:
            print(traceback.format_exc())

    def fit_picasso_localisations(self):
        if hasattr(self, "localisation_centres") == False:
            show_info("No localisations detected, please detect localisations first")
        else:
            image_channel = self.picasso_image_channel.currentText()
            box_size = int(self.picasso_box_size.currentText())
            min_net_gradient = self.picasso_min_net_gradient.text()

            camera_info = {"baseline": 100.0, "gain": 1, "sensitivity": 1.0, "qe": 0.9, }

            # check min net gradient is a number
            if min_net_gradient.isdigit() and image_channel != "":
                worker = Worker(self._fit_localisations, min_net_gradient=min_net_gradient, box_size=box_size, camera_info=camera_info, image_channel=image_channel, )
                worker.signals.progress.connect(partial(self._Progresbar, progressbar="picasso"))
                worker.signals.finished.connect(self._fit_localisations_cleanup)
                self.threadpool.start(worker)

