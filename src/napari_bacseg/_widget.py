"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/guides.html?#widgets

Replace code below according to your needs.
"""

import os
import shutil
import tempfile
import time
import traceback
from functools import partial

import cv2
import napari
import numpy as np
from napari.utils.colormaps import label_colormap
from napari.utils.notifications import show_info
from qtpy.QtCore import QThreadPool

from qtpy.QtWidgets import (QComboBox, QFileDialog, QLabel, QSlider, QWidget, )


from napari_bacseg.GUI.gui import Ui_Form as gui

from napari_bacseg.funcs.utils import _utils
from napari_bacseg.funcs.IO.import_utils import _import_utils
from napari_bacseg.funcs.IO.oni_utils import _oni_utils
from napari_bacseg.funcs.IO.olympus_utils import _olympus_utils
from napari_bacseg.funcs.IO.export_utils import _export_utils
from napari_bacseg.funcs.database_utils import _database_utils
from napari_bacseg.funcs.databaseIO_utils import _databaseIO
from napari_bacseg.funcs.cellpose_utils import _cellpose_utils
from napari_bacseg.funcs.tiler_utils import _tiler_utils
from napari_bacseg.funcs.undrift_utils import _undrift_utils
from napari_bacseg.funcs.IO.zeiss_utils import _zeiss_utils
from napari_bacseg.funcs.events_utils import _events_utils
from napari_bacseg.funcs.statistics_utils import _stats_utils
from napari_bacseg.funcs.IO.oufti_utils import _oufti_utils
from napari_bacseg.funcs.IO.imagej_utils import _imagej_utils
from napari_bacseg.funcs.picasso_utils import _picasso_utils
from napari_bacseg.funcs.bactfit_utils import _bactfit_utils
from napari_bacseg.funcs.cell_events import _cell_events

from napari_bacseg.funcs.threading_utils import Worker

sub_classes = [_picasso_utils, _utils, _import_utils, _export_utils,
    _database_utils, _databaseIO, _cellpose_utils, _events_utils,
    _tiler_utils, _zeiss_utils, _stats_utils, _oufti_utils, _imagej_utils,
    _oni_utils, _olympus_utils, _undrift_utils, _bactfit_utils, _cell_events]

class QWidget(QWidget, gui, *sub_classes):

    def __init__(self, viewer: napari.Viewer):

        super().__init__()

        self.viewer = viewer

        from napari_bacseg.__init__ import __version__ as version
        show_info(f"napari-bacseg version: {version}")

        self.initialise_widget_ui()
        self.initialise_label_layers()
        self.initialise_pyqt_events()
        self.initialise_keybindings()
        self.initialise_viewer_events()
        self.initialise_global_variables()

        self.update_import_limit()

        self.threadpool = QThreadPool()  # self.load_dev_data()

    def initialise_widget_ui(self):

        # create UI
        self.gui = gui()
        self.gui.setupUi(self)

    def initialise_pyqt_events(self):

        # import events
        self.gui.import_auto_contrast.stateChanged.connect(self._autoContrast)
        self.gui.import_import.clicked.connect(self._importDialog)
        self.gui.label_overwrite.clicked.connect(self.overwrite_channel_info)
        self.gui.import_mode.currentIndexChanged.connect(self._set_available_multiframe_modes)

        self.gui.fold.clicked.connect(self.fold_images)
        self.gui.unfold.clicked.connect(self.unfold_images)

        self.gui.align_active_image.clicked.connect(partial(self._align_images, mode="active"))
        self.gui.align_all_images.clicked.connect(partial(self._align_images, mode="all"))
        self.gui.undrift_images.clicked.connect(self._undrift_images)

        self.gui.scalebar_show.stateChanged.connect(self._updateScaleBar)
        self.gui.scalebar_resolution.textChanged.connect(self._updateScaleBar)
        self.gui.scalebar_units.currentTextChanged.connect(self._updateScaleBar)
        self.gui.overlay_filename.stateChanged.connect(self._updateFileName)
        self.gui.overlay_folder.stateChanged.connect(self._updateFileName)
        self.gui.overlay_microscope.stateChanged.connect(self._updateFileName)
        self.gui.overlay_datemodified.stateChanged.connect(self._updateFileName)
        self.gui.overlay_content.stateChanged.connect(self._updateFileName)
        self.gui.overlay_phenotype.stateChanged.connect(self._updateFileName)
        self.gui.overlay_strain.stateChanged.connect(self._updateFileName)
        self.gui.overlay_antibiotic.stateChanged.connect(self._updateFileName)
        self.gui.overlay_stain.stateChanged.connect(self._updateFileName)
        self.gui.overlay_staintarget.stateChanged.connect(self._updateFileName)
        self.gui.overlay_modality.stateChanged.connect(self._updateFileName)
        self.gui.overlay_lightsource.stateChanged.connect(self._updateFileName)
        self.gui.overlay_focus.stateChanged.connect(self._updateFileName)
        self.gui.overlay_debris.stateChanged.connect(self._updateFileName)
        self.gui.overlay_laplacian.stateChanged.connect(self._updateFileName)
        self.gui.overlay_range.stateChanged.connect(self._updateFileName)
        self.gui.zoom_apply.clicked.connect(self._applyZoom)

        # cellpose events
        self.gui.cellpose_flowthresh.valueChanged.connect(lambda: self._updateSliderLabel("cellpose_flowthresh", "cellpose_flowthresh_label"))
        self.gui.cellpose_maskthresh.valueChanged.connect(lambda: self._updateSliderLabel("cellpose_maskthresh", "cellpose_maskthresh_label"))
        self.gui.cellpose_minsize.valueChanged.connect(lambda: self._updateSliderLabel("cellpose_minsize", "cellpose_minsize_label"))
        self.gui.cellpose_diameter.valueChanged.connect(lambda: self._updateSliderLabel("cellpose_diameter", "cellpose_diameter_label"))

        self.gui.cellpose_select_custom_model.clicked.connect(self._select_custom_cellpose_model)
        self.gui.cellpose_save_dir.clicked.connect(self._select_cellpose_save_directory)
        self.gui.cellpose_segment_all.clicked.connect(self._segmentAll)
        self.gui.cellpose_segment_active.clicked.connect(self._segmentActive)
        self.gui.cellpose_train_model.clicked.connect(self._trainCellpose)
        self.gui.cellpose_segchannel.currentTextChanged.connect(self._updateSegChannels)

        # modify tab events
        self.gui.modify_panzoom.clicked.connect(self._modifyMode(mode="panzoom"))
        self.gui.modify_segment.clicked.connect(self._modifyMode(mode="segment"))
        self.gui.modify_classify.clicked.connect(self._modifyMode(mode="classify"))
        self.gui.modify_refine.clicked.connect(self._modifyMode(mode="refine"))
        self.gui.modify_add.clicked.connect(self._modifyMode(mode="add"))
        self.gui.modify_extend.clicked.connect(self._modifyMode(mode="extend"))
        self.gui.modify_join.clicked.connect(self._modifyMode(mode="join"))
        self.gui.modify_split.clicked.connect(self._modifyMode(mode="split"))
        self.gui.modify_delete.clicked.connect(self._modifyMode(mode="delete"))
        self.gui.classify_single.clicked.connect(self._modifyMode(mode="single"))
        self.gui.classify_dividing.clicked.connect(self._modifyMode(mode="dividing"))
        self.gui.classify_divided.clicked.connect(self._modifyMode(mode="divided"))
        self.gui.classify_vertical.clicked.connect(self._modifyMode(mode="vertical"))
        self.gui.classify_broken.clicked.connect(self._modifyMode(mode="broken"))
        self.gui.classify_edge.clicked.connect(self._modifyMode(mode="edge"))

        self.viewer.bind_key("b", func=self.set_blurred, overwrite=True)
        self.gui.set_focus_1.clicked.connect(partial(self.set_image_quality, mode="focus", value=1))
        self.gui.set_focus_2.clicked.connect(partial(self.set_image_quality, mode="focus", value=2))
        self.gui.set_focus_3.clicked.connect(partial(self.set_image_quality, mode="focus", value=3))
        self.gui.set_focus_4.clicked.connect(partial(self.set_image_quality, mode="focus", value=4))
        self.gui.set_focus_5.clicked.connect(partial(self.set_image_quality, mode="focus", value=5))
        self.viewer.bind_key("f", func=self.set_focused, overwrite=True)

        self.gui.set_debris_1.clicked.connect(partial(self.set_image_quality, mode="debris", value=1))
        self.gui.set_debris_2.clicked.connect(partial(self.set_image_quality, mode="debris", value=2))
        self.gui.set_debris_3.clicked.connect(partial(self.set_image_quality, mode="debris", value=3))
        self.gui.set_debris_4.clicked.connect(partial(self.set_image_quality, mode="debris", value=4))
        self.gui.set_debris_5.clicked.connect(partial(self.set_image_quality, mode="debris", value=5))

        self.gui.modify_viewmasks.stateChanged.connect(self._viewerControls("viewmasks"))
        self.gui.modify_viewlabels.stateChanged.connect(self._viewerControls("viewlabels"))
        self.gui.refine_all.clicked.connect(self._refine_bacseg)
        self.gui.modify_copymasktoall.clicked.connect(self._copymasktoall)
        self.gui.modify_deleteallmasks.clicked.connect(self._deleteallmasks(mode="all"))
        self.gui.modify_deleteactivemasks.clicked.connect(self._deleteallmasks(mode="active"))
        self.gui.modify_deleteactiveimage.clicked.connect(self._delete_active_image(mode="active"))
        self.gui.modify_deleteotherimages.clicked.connect(self._delete_active_image(mode="other"))
        self.gui.find_next.clicked.connect(self._sort_cells("next"))
        self.gui.find_previous.clicked.connect(self._sort_cells("previous"))
        self.gui.modify_channel.currentTextChanged.connect(self._modify_channel_changed)

        self.gui.filter_report.clicked.connect(partial(self._filter_segmentations, remove=False))
        self.gui.filter_remove.clicked.connect(partial(self._filter_segmentations, remove=True))

        # export events
        self.gui.export_active.clicked.connect(self._export("active"))
        self.gui.export_all.clicked.connect(self._export("all"))
        self.gui.export_stack_active.clicked.connect(self._export_stack("active"))
        self.gui.export_stack_all.clicked.connect(self._export_stack("all"))
        self.gui.export_statistics_active.clicked.connect(self._export_statistics("active"))
        self.gui.export_statistics_all.clicked.connect(self._export_statistics("all"))

        # oufti events
        self.gui.oufti_generate_all_midlines.clicked.connect(self.generate_midlines(mode="all"))
        self.gui.oufti_generate_active_midlines.clicked.connect(self.generate_midlines(mode="active"))
        self.viewer.bind_key("m", func=self.midline_edit_toggle, overwrite=True)
        self.gui.oufti_edit_mode.clicked.connect(self.midline_edit_toggle)
        self.gui.oufti_panzoom_mode.clicked.connect(self.midline_edit_toggle)
        self.gui.oufti_centre_all_midlines.clicked.connect(self.centre_oufti_midlines(mode="all"))
        self.gui.oufti_centre_active_midlines.clicked.connect(self.centre_oufti_midlines(mode="active"))

        # upload tab events
        self.gui.upload_all.clicked.connect(self._uploadDatabase(mode="all"))
        self.gui.upload_active.clicked.connect(self._uploadDatabase(mode="active"))
        self.gui.database_download.clicked.connect(self._downloadDatabase)
        self.gui.create_database.clicked.connect(self._create_bacseg_database)
        self.gui.load_database.clicked.connect(self._load_bacseg_database)
        self.gui.store_metadata.clicked.connect(self.update_database_metadata)

        self.gui.picasso_detect.clicked.connect(self.detect_picasso_localisations)
        self.gui.picasso_fit.clicked.connect(self.fit_picasso_localisations)
        self.gui.picasso_vis_size.currentIndexChanged.connect(self.update_localisation_visualisation)
        self.gui.picasso_vis_mode.currentIndexChanged.connect(self.update_localisation_visualisation)
        self.gui.picasso_vis_opacity.currentIndexChanged.connect(self.update_localisation_visualisation)
        self.gui.picasso_vis_edge_width.currentIndexChanged.connect(self.update_localisation_visualisation)
        self.gui.picasso_export.clicked.connect(self.export_picasso_localisations)

        self.gui.upload_initial.currentTextChanged.connect(self.populate_upload_combos)

        self.gui.filter_metadata.clicked.connect(self.update_upload_combos)
        self.gui.reset_metadata.clicked.connect(self.populate_upload_combos)

        self.viewer.dims.events.current_step.connect(self._sliderEvent)

        self.gui.import_filemode.currentIndexChanged.connect(self.update_import_limit)
        self.gui.import_mode.currentIndexChanged.connect(self.update_import_limit)

        self.gui.export_channel.currentIndexChanged.connect(self.update_export_options)

        self.gui.fit_segmentations.clicked.connect(self.initialise_bactfit)

    def initialise_keybindings(self):

        self.viewer.bind_key("a", func=self._modifyMode(mode="add"), overwrite=True)
        self.viewer.bind_key("e", func=self._modifyMode(mode="extend"), overwrite=True)
        self.viewer.bind_key("j", func=self._modifyMode(mode="join"), overwrite=True)
        self.viewer.bind_key("s", func=self._modifyMode(mode="split"), overwrite=True)
        self.viewer.bind_key("d", func=self._modifyMode(mode="delete"), overwrite=True)
        self.viewer.bind_key("r", func=self._modifyMode(mode="refine"), overwrite=True)
        self.viewer.bind_key("k", func=self._modifyMode(mode="clicktozoom"), overwrite=True)
        self.viewer.bind_key("Control-1", func=self._modifyMode(mode="single"), overwrite=True, )
        self.viewer.bind_key("Control-2", func=self._modifyMode(mode="dividing"), overwrite=True, )
        self.viewer.bind_key("Control-3", func=self._modifyMode(mode="divided"), overwrite=True, )
        self.viewer.bind_key("Control-4", func=self._modifyMode(mode="vertical"), overwrite=True, )
        self.viewer.bind_key("Control-5", func=self._modifyMode(mode="broken"), overwrite=True, )
        self.viewer.bind_key("Control-6", func=self._modifyMode(mode="edge"), overwrite=True)
        self.viewer.bind_key("F1", func=self._modifyMode(mode="panzoom"), overwrite=True)
        self.viewer.bind_key("F2", func=self._modifyMode(mode="segment"), overwrite=True)
        self.viewer.bind_key("F3", func=self._modifyMode(mode="classify"), overwrite=True)
        self.viewer.bind_key("h", func=self._viewerControls("h"), overwrite=True)
        self.viewer.bind_key("i", func=self._viewerControls("i"), overwrite=True)
        self.viewer.bind_key("o", func=self._viewerControls("o"), overwrite=True)
        self.viewer.bind_key("x", func=self._viewerControls("x"), overwrite=True)
        self.viewer.bind_key("z", func=self._viewerControls("z"), overwrite=True)
        self.viewer.bind_key("c", func=self._viewerControls("c"), overwrite=True)
        self.viewer.bind_key("Right", func=self._imageControls("Right"), overwrite=True)
        self.viewer.bind_key("Left", func=self._imageControls("Left"), overwrite=True)
        self.viewer.bind_key("u", func=self._imageControls("Upload"), overwrite=True)
        self.viewer.bind_key("Control-d", func=self._deleteallmasks(mode="active"), overwrite=True, )
        self.viewer.bind_key("Control-Shift-d", func=self._deleteallmasks(mode="all"), overwrite=True, )
        self.viewer.bind_key("Control-i", func=self._delete_active_image(mode="active"), overwrite=True, )
        self.viewer.bind_key("Control-Shift-i", func=self._delete_active_image(mode="other"), overwrite=True, )

        # self.viewer.bind_key("Control-l", func=self._downloadDatabase(), overwrite=True)
        self.viewer.bind_key("Control-u", func=self._uploadDatabase(mode="active"), overwrite=True, )
        self.viewer.bind_key("Control-Shift-u", func=self._uploadDatabase(mode="all"), overwrite=True, )
        #
        self.viewer.bind_key("Control-Left", func=self._manual_align_channels("left", mode="active"), overwrite=True, )
        self.viewer.bind_key("Control-Right", func=self._manual_align_channels("right", mode="active"), overwrite=True, )
        self.viewer.bind_key("Control-Up", func=self._manual_align_channels("up", mode="active"), overwrite=True, )
        self.viewer.bind_key("Control-Down", func=self._manual_align_channels("down", mode="active"), overwrite=True, )

        self.viewer.bind_key("Alt-Left", func=self._manual_align_channels("left", mode="all"), overwrite=True, )
        self.viewer.bind_key("Alt-Right", func=self._manual_align_channels("right", mode="all"), overwrite=True, )
        self.viewer.bind_key("Alt-Up", func=self._manual_align_channels("up", mode="all"), overwrite=True, )
        self.viewer.bind_key("Alt-Down", func=self._manual_align_channels("down", mode="all"), overwrite=True, )

    def initialise_viewer_events(self):

        self.viewer.layers.events.inserted.connect(self._manualImport)
        self.viewer.layers.events.removed.connect(self._updateSegmentationCombo)
        self.viewer.layers.selection.events.changed.connect(self._updateFileName)

    def initialise_global_variables(self):

        # import controls from Qt Desinger References
        self.path_list = []
        self.active_import_mode = ""

        # cellpose controls + variables from Qt Desinger References
        self.cellpose_segmentation = False
        self.cellpose_model = None
        self.cellpose_custom_model_path = ""
        self.cellpose_train_model_path = ""
        self.cellpose_log_file = None

        # modify tab controls + variables from Qt Desinger References
        self.interface_mode = "panzoom"
        self.segmentation_mode = "add"
        self.class_mode = "single"
        self.class_colour = 1

        # upload tab controls from Qt Desinger References
        self.database_path = ""
        self.user_metadata_path = ""
        self.user_metadata = None
        self.expected_columns = None

        self.user_metadata_keys = 6

        self.metadata_columns = ["date_uploaded", "date_created", "date_modified", "file_name", "channel", "file_list", "channel_list", "segmentation_file", "segmentation_channel", "akseg_hash",
            "user_initial", "content", "microscope", "modality", "source", "strain", "phenotype", "stain", "stain_target", "antibiotic", "treatment time (mins)", "antibiotic concentration",
            "mounting method", "protocol", "folder", "parent_folder", "num_segmentations", "image_laplacian", "image_focus", "image_debris", "segmented", "labelled", "segmentation_curated",
            "label_curated", "posX", "posY", "posZ", "image_load_path", "image_save_path", "mask_load_path", "mask_save_path", "label_load_path", "label_save_path", ]

        user_key_list = np.arange(1, self.user_metadata_keys + 1).tolist()
        user_key_list.reverse()

        for key in user_key_list:
            user_key = f"user_meta{key}"
            self.metadata_columns.insert(22, str(user_key))
            setattr(self, f"upload_usermeta{key}", self.findChild(QComboBox, f"upload_usermeta{key}"), )

        self.tiler_object = None
        self.tile_dict = {"Segmentations": [], "Classes": [], "Nucleoid": []}
        self.unfolded = False
        self.updating_combos = False
        self.contours = []

        self.widget_notifications = True
        self._show_database_controls(False)

    def initialise_label_layers(self):

        self.class_colours = {
            0: (0 / 255, 0 / 255, 0 / 255, 1),
            1: (255 / 255, 255 / 255, 255 / 255, 1),
            2: (0 / 255, 255 / 255, 0 / 255, 1),
            3: (0 / 255, 170 / 255, 255 / 255, 1),
            4: (170 / 255, 0 / 255, 255 / 255, 1),
            5: (255 / 255, 170 / 255, 0 / 255, 1),
            6: (255 / 255, 0 / 255, 0 / 255, 1),
            None: (255 / 255, 255 / 255, 255 / 255, 1),
        }

        for key,value in self.class_colours.items():
            self.class_colours[key] = np.array(value).astype(np.float32)

        self.class_cmap = napari.utils.colormaps.DirectLabelColormap(
            color_dict=self.class_colours)

        self.classLayer = self.viewer.add_labels(np.zeros((1, 100, 100),
            dtype=np.uint16), opacity=0.25, name="Classes",
            colormap =self.class_cmap,
            metadata={0: {"image_name": ""}}, visible=False, )

        self.nucLayer = self.viewer.add_labels(np.zeros((1, 100, 100),
            dtype=np.uint16), opacity=1, name="Nucleoid",
            metadata={0: {"image_name": ""}}, )

        self.segLayer = self.viewer.add_labels(np.zeros((1, 100, 100),
            dtype=np.uint16), opacity=1, name="Segmentations",
            metadata={0: {"image_name": ""}}, )

        self.segLayer.mouse_drag_callbacks.append(self._segmentationEvents)
        self.nucLayer.mouse_drag_callbacks.append(self._segmentationEvents)
        self.segLayer.mouse_double_click_callbacks.append(self._doubeClickEvents)

        self.segLayer.contour = 1

    def update_export_options(self):

        try:

            export_channel = self.gui.export_channel.currentText()

            if export_channel == "Multi Channel":
                self.gui.export_multi_channel_mode.setEnabled(True)
                self.gui.export_multi_channel_mode_label.setEnabled(True)

                self.gui.export_multi_channel_mode.setVisible(True)
                self.gui.export_multi_channel_mode_label.setVisible(True)

                self.gui.export_multi_channel_mode
            else:
                self.gui.export_multi_channel_mode.setEnabled(False)
                self.gui.export_multi_channel_mode_label.setEnabled(False)

                self.gui.export_multi_channel_mode.setVisible(False)
                self.gui.export_multi_channel_mode_label.setVisible(False)

        except:
            print(traceback.format_exc())


    def _check_number_string(self, string):

        try:
            float(string)
            return True
        except:
            return False

    def _filter_segmentations(self,viewer = None, remove = True):

        try:

            metric = self.gui.filter_metric.currentText()
            criteria = self.gui.filter_criteria.currentText()
            threshold = self.gui.filter_threshold.text()
            fov_mode = self.gui.filter_mode.currentText()
            ignore_edge = self.gui.filter_ignore_edge.isChecked()

            if self._check_number_string(threshold):

                threshold = float(threshold)

                if hasattr(self, "segLayer"):

                    self._filter_cells(remove=remove, fov_mode=fov_mode, metric=metric,
                        criteria=criteria, threshold=threshold, ignore_edge=ignore_edge)

            else:
                show_info("Thereshold must be a number")

        except:
            print(traceback.format_exc())
            pass


    def _set_available_multiframe_modes(self):
        multiframe_items = [self.gui.import_multiframe_mode.itemText(i) for i in range(self.gui.import_multiframe_mode.count())]
        mode = self.gui.import_mode.currentText()

        if mode in ["Images", "NanoImager Data", "ImageJ files(s)"]:
            if "Keep All Frames (BETA)" not in multiframe_items:
                self.gui.import_multiframe_mode.addItem("Keep All Frames (BETA)")
        else:
            if "Keep All Frames (BETA)" in multiframe_items:
                self.gui.import_multiframe_mode.removeItem(multiframe_items.index("Keep All Frames (BETA)"))

    def _applyZoom(self):
        try:
            import re

            magnification = self.gui.zoom_magnification.currentText()
            pixel_resolution = float(self.gui.scalebar_resolution.text())
            magnification = (magnification.lower().replace("x", "").replace("%", ""))

            magnification = re.findall(r"\b\d+\b", magnification)[0]

            if magnification.isdigit():
                magnification = int(magnification)

                if magnification == 0:
                    self.viewer.reset_view()
                elif magnification > 0:
                    magnification = 1 + magnification / 100

                    self.viewer.camera.zoom = magnification * (1 / pixel_resolution)

        except:
            print(traceback.format_exc())

    def _align_images(self, viewer=None, mode="active"):
        import scipy
        from skimage.registration import phase_cross_correlation

        try:
            layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Nucleoid", "Classes", "center_lines", "Localisations", ]]

            if len(layer_names) > 2:
                num_images = self.viewer.layers[layer_names[0]].data.shape[0]

                if mode == "active":
                    fov_list = [self.viewer.dims.current_step[0]]
                else:
                    fov_list = range(num_images)

                alignment_channel = self.gui.alignment_channel.currentText()
                current_fov = self.viewer.dims.current_step[0]

                target_channels = [layer for layer in layer_names if layer != alignment_channel]

                for channel in target_channels:
                    image_stack = self.viewer.layers[channel].data.copy()
                    target_image_stack = self.viewer.layers[alignment_channel].data.copy()

                    for fov in fov_list:
                        target_image = image_stack[fov, :, :]
                        alignment_image = target_image_stack[fov, :, :]

                        shift, error, diffphase = phase_cross_correlation(alignment_image, target_image, upsample_factor=100)

                        shifted_img = scipy.ndimage.shift(target_image, shift)

                        image_stack[fov, :, :] = shifted_img

                    self.viewer.layers[channel].data = image_stack

                show_info(f"{len(fov_list)} Image(s) aligned to channel: " + alignment_channel)

        except:
            pass

    def set_blurred(self, viewer=None):
        self.set_image_quality(mode="focus", value=1)

    def set_focused(self, viewer=None):
        self.set_image_quality(mode="focus", value=5)

    def set_image_quality(self, mode="", value=""):
        try:
            layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Nucleoid", "Classes", "center_lines", "Localisations", ]]

            update_mode = self.gui.set_quality_mode.currentIndex()

            if len(layer_names) > 0:
                current_fov = self.viewer.dims.current_step[0]
                n_frames = self.viewer.dims.nsteps[0]
                active_layer = self.viewer.layers.selection.active.name

                if update_mode == 0:
                    frames = [current_fov]
                    layers = [active_layer]
                if update_mode == 1:
                    frames = [current_fov]
                    layers = layer_names
                if update_mode == 2:
                    frames = np.arange(n_frames)
                    layers = [active_layer]
                if update_mode == 3:
                    frames = np.arange(n_frames)
                    layers = layer_names

                for layer in layers:
                    for frame in frames:
                        meta = self.viewer.layers[layer].metadata.copy()

                        meta[frame][f"image_{mode}"] = value

                        self.viewer.layers[layer].metadata = meta

            self._updateFileName()

        except:
            print(traceback.format_exc())

    def wrapper(self, func, *args, **kwargs):
        try:
            func_name = func.__name__

            func = partial(func, self, *args, **kwargs)
            func.__name__ = func_name

            setattr(self, func_name, func)

        except:
            pass

        return func

    def overwrite_channel_info(self):

        all_layers = [layer.name for layer in self.viewer.layers]
        selected_layers = [layer.name for layer in self.viewer.layers.selection]

        if len(selected_layers) == 1:
            selected_layer = selected_layers[0]
            all_layers.pop(all_layers.index(selected_layer))

            if selected_layer not in ["Segmentations", "Nucleoid", "Classes", "center_lines", "Localisations", ]:
                metadata = self.viewer.layers[selected_layer].metadata.copy()

                img_modality = self.gui.img_modality.currentText()
                img_light_source = self.gui.img_light_source.currentText()
                img_stain = self.gui.img_stain.currentText()
                img_stain_target = self.gui.img_stain_target.currentText()

                if img_stain != "":
                    channel = img_stain
                else:
                    channel = img_modality

                if channel in ["", None]:
                    channel = selected_layer

                self.viewer.layers[selected_layer].name = channel

                for i in range(len(metadata)):
                    metadata[i]["channel"] = channel
                    metadata[i]["modality"] = img_modality
                    metadata[i]["light_source"] = img_light_source
                    metadata[i]["stain"] = img_stain
                    metadata[i]["stain_target"] = img_stain_target

                self.viewer.layers[channel].metadata = metadata

                self._updateFileName()
                self._updateSegmentationCombo()
                self._updateSegChannels()

    def _export_statistics(self, mode="active"):
        def _event(viewer):
            multithreaded = self.gui.export_statistics_multithreaded.isChecked()

            if self.unfolded == True:
                self.fold_images()

            pixel_size = float(self.gui.export_statistics_pixelsize.text())

            colicoords_channel = self.gui.export_colicoords_mode.currentText()
            colicoords_channel = colicoords_channel.replace("Mask + ", "")

            if pixel_size <= 0:
                pixel_size = 1

            desktop = os.path.expanduser("~/Desktop")

            path = QFileDialog.getExistingDirectory(self, "Select Directory", desktop)

            colicoords_dir = os.path.join(tempfile.gettempdir(), "colicoords")

            if os.path.isdir(colicoords_dir) != True:
                os.mkdir(colicoords_dir)
            else:
                shutil.rmtree(colicoords_dir)
                os.mkdir(colicoords_dir)

            if os.path.isdir(path):
                path = os.path.abspath(path)

                worker = Worker(self.get_cell_statistics, mode=mode, pixel_size=pixel_size, colicoords_dir=colicoords_dir, )
                worker.signals.progress.connect(partial(self._Progresbar, progressbar="export"))
                worker.signals.result.connect(partial(self.process_cell_statistics, path=path))
                self.threadpool.start(worker)
                cell_data = worker.result()

                if self.gui.export_colicoords_mode.currentIndex() != 0:

                    from napari_bacseg.funcs.colicoords_utils import run_colicoords
                    self.run_colicoords = self.wrapper(run_colicoords)

                    worker = Worker(self.run_colicoords, cell_data=cell_data, colicoords_channel=colicoords_channel, pixel_size=pixel_size, statistics=True, multithreaded=multithreaded, )

                    worker.signals.progress.connect(partial(self._Progresbar, progressbar="export"))
                    worker.signals.result.connect(partial(self.process_cell_statistics, path=path))
                    self.threadpool.start(worker)

        return _event

    def update_import_limit(self):
        if (self.gui.import_filemode.currentIndex() == 1 or self.gui.import_mode.currentText() == "Zeiss (.czi) Data"):
            self.gui.import_limit.setEnabled(True)
            self.gui.import_limit.setCurrentIndex(0)
            self.gui.import_limit.show()
            self.gui.import_limit_label.show()

            if self.gui.import_mode.currentText() == "Zeiss (.czi) Data":
                self.gui.import_limit_label.setText("Import Limit (FOV)")
            else:
                self.gui.import_limit_label.setText("Import Limit (Files)")

        else:
            self.gui.import_limit.setEnabled(False)
            self.gui.import_limit.setCurrentIndex(6)
            self.gui.import_limit.hide()
            self.gui.import_limit_label.hide()

    def _sort_cells(self, order):
        def _event(viewer):
            if self.unfolded == True:
                self.fold_images()

            try:
                current_fov = self.viewer.dims.current_step[0]

                meta = self.segLayer.metadata[current_fov]

                self._compute_simple_cell_stats()

                find_criterion = self.gui.find_criterion.currentText()
                find_mode = self.gui.find_mode.currentText()

                cell_centre = meta["simple_cell_stats"]["cell_centre"]
                cell_zoom = meta["simple_cell_stats"]["cell_zoom"]

                if find_criterion == "Cell Area":
                    criterion = meta["simple_cell_stats"]["cell_area"]
                if find_criterion == "Cell Solidity":
                    criterion = meta["simple_cell_stats"]["cell_solidity"]
                if find_criterion == "Cell Aspect Ratio":
                    criterion = meta["simple_cell_stats"]["cell_aspect_ratio"]

                if find_mode == "Ascending":
                    criterion, cell_centre, cell_zoom = zip(*sorted(zip(criterion, cell_centre, cell_zoom), key=lambda x: x[0], ))
                else:
                    criterion, cell_centre, cell_zoom = zip(*sorted(zip(criterion, cell_centre, cell_zoom), key=lambda x: x[0], reverse=True, ))

                current_position = tuple(np.array(self.viewer.camera.center).round())

                if current_position not in cell_centre:
                    self.viewer.camera.center = cell_centre[0]
                    self.viewer.camera.zoom = cell_zoom[0]

                else:
                    current_index = cell_centre.index(current_position)

                    if order == "next":
                        new_index = current_index + 1

                    if order == "previous":
                        new_index = current_index - 1

                    new_index = max(current_fov, min(new_index, len(cell_centre) - 1))

                    self.viewer.camera.center = cell_centre[new_index]
                    self.viewer.camera.zoom = cell_zoom[new_index]

            except:
                pass

        return _event

    def _manual_align_channels(self, key, viewer=None, mode="active"):
        def _event(viewer):
            if self.unfolded == True:
                self.fold_images()

            from scipy.ndimage import shift

            current_fov = self.viewer.dims.current_step[0]
            active_layer = self.viewer.layers.selection.active

            if key == "up":
                shift_vector = [-1.0, 0.0]
            elif key == "down":
                shift_vector = [1.0, 0.0]
            elif key == "left":
                shift_vector = [0.0, -1.0]
            elif key == "right":
                shift_vector = [0.0, 1.0]
            else:
                shift_vector = [0.0, 0.0]

            shift_image = False
            if active_layer != None:
                if active_layer.name not in ["Segmentations", "Nucleoid", "Classes", "center_lines", "Localisations", ]:
                    shift_image = True

            if shift_image is True:
                if mode == "active":
                    image_stack = active_layer.data.copy()
                    image = image_stack[current_fov, :, :]
                    image = shift(image, shift=shift_vector)
                    image_stack[current_fov, :, :] = np.expand_dims(image, 0)

                    active_layer.data = image_stack

                else:
                    image_stack = active_layer.data.copy()

                    for i in range(len(image_stack)):
                        image = image_stack[i, :, :]
                        image = shift(image, shift=shift_vector)
                        image_stack[i, :, :] = np.expand_dims(image, 0)

                    active_layer.data = image_stack
            else:
                mask_stack = self.segLayer.data.copy()
                label_stack = self.classLayer.data.copy()

                mask = mask_stack[current_fov, :, :]
                label = label_stack[current_fov, :, :]

                mask = shift(mask, shift=shift_vector)
                label = shift(label, shift=shift_vector)

                mask_stack[current_fov, :, :] = np.expand_dims(mask, 0)
                label_stack[current_fov, :, :] = np.expand_dims(label, 0)

                self.segLayer.data = mask_stack
                self.classLayer.data = label_stack

        return _event

    def _refine_bacseg(self):
        if self.unfolded == True:
            self.fold_images()

        pixel_size = float(self.gui.export_statistics_pixelsize.text())

        if pixel_size <= 0:
            pixel_size = 1

        current_fov = self.viewer.dims.current_step[0]

        channel = self.gui.refine_channel.currentText()
        colicoords_channel = channel.replace("Mask + ", "")

        mask_stack = self.segLayer.data
        mask = mask_stack[current_fov, :, :].copy()

        colicoords_dir = os.path.join(tempfile.gettempdir(), "colicoords")

        worker = Worker(self.get_cell_statistics, mode="active", pixel_size=pixel_size, colicoords_dir=colicoords_dir, )

        self.threadpool.start(worker)
        cell_data = worker.result()

        worker = Worker(self.run_colicoords, cell_data=cell_data, colicoords_channel=colicoords_channel, pixel_size=pixel_size, multithreaded=True, )

        worker.signals.progress.connect(partial(self._Progresbar, progressbar="modify"))
        worker.signals.result.connect(self.process_colicoords)
        self.threadpool.start(worker)

    def _uploadDatabase(self, viewer=None, mode=""):
        def _event(viewer):
            print(True)
            try:
                if (self.database_path != "" and os.path.exists(self.database_path) == True):
                    if self.unfolded == True:
                        self.fold_images()

                    if self.gui.upload_initial.currentText() in ["", "Required for upload", ]:
                        show_info("Please select the user initial.")
                    else:

                        worker = Worker(self.backup_user_metadata)
                        self.threadpool.start(worker)

                        worker = Worker(self._upload_bacseg_database, mode=mode)
                        worker.signals.progress.connect(partial(self._Progresbar, progressbar="database_upload"))
                        self.threadpool.start(worker)
            except:
                print(traceback.format_exc())
                pass

        return _event

    def _downloadDatabase(self, viewer=None):
        try:
            if (self.database_path != "" and os.path.exists(self.database_path) == True):
                if self.unfolded == True:
                    self.fold_images()

                if self.gui.upload_initial.currentText() in ["", "Required for upload", ]:
                    show_info("Please select the user initial.")

                else:

                    self.active_import_mode = "BacSeg"

                    (measurements, file_paths, channels,) = self.get_filtered_database_metadata()

                    if len(file_paths) == 0:
                        if self.widget_notifications:
                            show_info("no matching database files found")

                    else:

                        worker = Worker(self.read_bacseg_images, measurements=measurements, channels=channels, )
                        worker.signals.result.connect(self._process_import)
                        worker.signals.progress.connect(partial(self._Progresbar, progressbar="database_download"))
                        self.threadpool.start(worker)

        except:
            print(traceback.format_exc())

    def _updateSegChannels(self):
        layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Nucleoid", "Classes", "center_lines", "Localisations", ]]

        segChannel = self.gui.cellpose_segchannel.currentText()

        self.gui.export_channel.setCurrentText(segChannel)


    def _Progresbar(self, progress, progressbar):
        if progressbar == "picasso":
            self.gui.picasso_progressbar.setValue(progress)
        if progressbar == "import":
            self.gui.import_progressbar.setValue(progress)
        if progressbar == "export":
            self.gui.exporttab_progressbar.setValue(progress)
        if progressbar == "cellpose":
            self.gui.cellpose_progressbar.setValue(progress)
        if progressbar == "database_upload":
            self.gui.upload_progressbar.setValue(progress)
        if progressbar == "database_download":
            self.gui.download_progressbar.setValue(progress)
        if progressbar == "modify":
            self.gui.modify_progressbar.setValue(progress)
        if progressbar == "undrift":
            self.gui.undrift_progressbar.setValue(progress)
        if progressbar == "bactfit":
            self.gui.bactfit_progressbar.setValue(progress)


        if progress == 100:
            time.sleep(1)
            self.gui.import_progressbar.setValue(0)
            self.gui.exporttab_progressbar.setValue(0)
            self.gui.cellpose_progressbar.setValue(0)
            self.gui.modify_progressbar.setValue(0)
            self.gui.undrift_progressbar.setValue(0)
            self.gui.download_progressbar.setValue(0)
            self.gui.upload_progressbar.setValue(0)
            self.gui.bactfit_progressbar.setValue(0)
            self.gui.picasso_progressbar.setValue(0)

    def _importDialog(self, paths=None):
        if self.unfolded == True:
            self.fold_images()

        import_mode = self.gui.import_mode.currentText()
        import_filemode = self.gui.import_filemode.currentText()

        file_extension = "*.tif"

        if import_mode == "Images":
            file_extension = "*.tif *.png *.jpeg *.fits"
        if import_mode == "Cellpose (.npy) Segmentation(s)":
            file_extension = "*.npy"
        if import_mode == "Oufti (.mat) Segmentation(s)":
            file_extension = "*.mat"
        if import_mode == "JSON (.txt) Segmentation(s)":
            file_extension = "*.txt"
        if import_mode == "Zeiss (.czi) Data":
            file_extension = "*.czi"

        desktop = os.path.expanduser("~/Desktop")

        if type(paths) is not list:
            if import_filemode == "Import File(s)":
                paths, _ = QFileDialog.getOpenFileNames(self, "Open Files", desktop, f"Files ({file_extension})")

            if import_filemode == "Import Directory":
                path = QFileDialog.getExistingDirectory(self, "Select Directory", desktop)

                paths = [path]

        if "" in paths or paths == [] or type(paths) is not list:
            if self.widget_notifications:
                show_info("No file/folder selected")

        else:
            if import_mode == "Images":

                worker = Worker(self.import_images, file_paths=paths)
                worker.signals.result.connect(self._process_import)
                worker.signals.progress.connect(partial(self._Progresbar, progressbar="import"))
                self.threadpool.start(worker)

            if import_mode == "NanoImager Data":

                measurements, file_paths, channels = self.read_nim_directory(paths)

                worker = Worker(self.read_nim_images, measurements=measurements, channels=channels, )
                worker.signals.result.connect(self._process_import)
                worker.signals.progress.connect(partial(self._Progresbar, progressbar="import"))
                self.threadpool.start(worker)

            if import_mode == "Mask (.tif) Segmentation(s)":

                self.import_masks(paths, file_extension=".tif")
                self._autoClassify()

            if import_mode == "Cellpose (.npy) Segmentation(s)":

                self.import_masks(paths, file_extension=".npy")
                self._autoClassify()

            if import_mode == "Oufti (.mat) Segmentation(s)":

                self.import_masks(paths, file_extension=".mat")
                self._autoClassify()

            if import_mode == "JSON (.txt) Segmentation(s)":

                self.import_masks(paths, file_extension=".txt")
                self._autoClassify()

            if import_mode == "ImageJ files(s)":

                worker = Worker(self.import_imagej, paths=paths)
                worker.signals.result.connect(self._process_import)
                worker.signals.progress.connect(partial(self._Progresbar, progressbar="import"))
                self.threadpool.start(worker)

            if import_mode == "ScanR Data":

                measurements, file_paths, channels = self.read_scanr_directory(paths)

                worker = Worker(self.read_scanr_images, measurements=measurements, channels=channels, )
                worker.signals.result.connect(self._process_import)
                worker.signals.progress.connect(partial(self._Progresbar, progressbar="import"))
                self.threadpool.start(worker)

            if import_mode == "Zeiss (.czi) Data":
                try:

                    (zeiss_measurements, channel_names, num_measurements,) = self.get_zeiss_measurements(paths)

                    worker = Worker(self.read_zeiss_image_files, zeiss_measurements=zeiss_measurements, channel_names=channel_names, num_measurements=num_measurements, )
                    worker.signals.result.connect(self._process_import)
                    worker.signals.progress.connect(partial(self._Progresbar, progressbar="import"))
                    self.threadpool.start(worker)

                except:
                    print(traceback.format_exc())

    def _export_stack(self, mode, viewer=None):
        def _event(viewer):
            execute_export = True

            if self.gui.export_location.currentIndex() == 1:
                desktop = os.path.expanduser("~/Desktop")
                self.export_directory = QFileDialog.getExistingDirectory(self, "Select Directory", desktop)

                if self.export_directory == "":
                    execute_export = False

            if execute_export == True:

                worker = Worker(self.export_stacks, mode=mode)
                worker.signals.progress.connect(partial(self._Progresbar, progressbar="export"))
                self.threadpool.start(worker)

        return _event

    def _export(self, mode, viewer=None):
        def _event(viewer):
            # if self.unfolded == True:
            #     self.fold_images()

            execute_export = True

            if self.gui.export_location.currentIndex() == 1:
                desktop = os.path.expanduser("~/Desktop")
                self.export_directory = QFileDialog.getExistingDirectory(self, "Select Directory", desktop)

                if self.export_directory == "":
                    execute_export = False

            if execute_export == True:

                worker = Worker(self.export_files, mode=mode)
                worker.signals.progress.connect(partial(self._Progresbar, progressbar="export"))
                self.threadpool.start(worker)

        return _event

    def _trainCellpose(self):
        if self.unfolded == True:
            self.fold_images()

        worker = Worker(self.train_cellpose_model)
        worker.signals.progress.connect(partial(self._Progresbar, progressbar="cellpose_train"))
        self.threadpool.start(worker)

    def _segmentActive(self):
        if self.unfolded == True:
            self.fold_images()

        current_fov = int(self.viewer.dims.current_step[0])
        chanel = str(self.gui.cellpose_segchannel.currentText())

        images = self.viewer.layers[chanel].data.copy()

        image = [images[current_fov, :, :]]

        worker = Worker(self._run_cellpose, images=image)
        worker.signals.result.connect(self._process_cellpose)
        worker.signals.progress.connect(partial(self._Progresbar, progressbar="cellpose"))
        self.threadpool.start(worker)

    def _segmentAll(self):
        if self.unfolded == True:
            self.fold_images()

        channel = str(self.gui.cellpose_segchannel.currentText())
        images = self.viewer.layers[channel].data.copy()

        images = self.unstack_images(images)

        worker = Worker(self._run_cellpose, images=images)
        worker.signals.result.connect(self._process_cellpose)
        worker.signals.progress.connect(partial(self._Progresbar, progressbar="cellpose"))
        self.threadpool.start(worker)


    def _updateSliderLabel(self, slider_name, label_name):
        self.slider = self.findChild(QSlider, slider_name)
        self.gui.label = self.findChild(QLabel, label_name)

        slider_value = self.slider.value()

        if (slider_name == "cellpose_flowthresh" or slider_name == "cellpose_maskthresh"):
            self.gui.label.setText(str(slider_value / 100))
        else:
            self.gui.label.setText(str(slider_value))

    def _updateSegmentationCombo(self):
        layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Nucleoid", "Classes", "center_lines", "Localisations", ]]

        self.gui.cellpose_segchannel.clear()
        self.gui.cellpose_segchannel.addItems(layer_names)

        self.gui.cellpose_trainchannel.clear()
        self.gui.cellpose_trainchannel.addItems(layer_names)

        self.gui.cellpose_trainchannel.clear()
        self.gui.cellpose_trainchannel.addItems(layer_names)

        self.gui.alignment_channel.clear()
        self.gui.alignment_channel.addItems(layer_names)

        self.gui.undrift_channel.clear()
        self.gui.undrift_channel.addItems(layer_names)

        self.gui.export_stack_channel.clear()
        self.gui.export_stack_channel.addItems(layer_names)

        self.gui.picasso_image_channel.clear()
        self.gui.picasso_image_channel.addItems(layer_names)

        self.gui.export_channel.clear()
        export_layers = layer_names
        export_layers.extend(["Multi Channel", ])
        self.gui.export_channel.addItems(export_layers)

        self.gui.refine_channel.clear()
        refine_layers = ["Mask + " + layer for layer in layer_names]
        self.gui.refine_channel.addItems(["Mask"] + refine_layers)

        self.gui.export_colicoords_mode.clear()
        refine_layers = ["Mask + " + layer for layer in layer_names]
        self.gui.export_colicoords_mode.addItems(["None (OpenCV Stats)", "Mask"] + refine_layers)

    def _sliderEvent(self, current_step):
        try:
            active_layer = self.viewer.layers.selection.active

            crop = active_layer.corner_pixels.T

            y_range = crop[-2]
            x_range = crop[-1]
        except:
            pass

        self._updateFileName()
        self._autoContrast()
        self._updateScaleBar()
        self._update_active_midlines()  # self.display_localisations()

    def _updateScaleBar(self):
        layer_names = [layer.name for layer in self.viewer.layers]

        try:
            if self.gui.scalebar_show.isChecked() and len(layer_names) > 0:
                pixel_resolution = float(self.gui.scalebar_resolution.text())
                scalebar_units = self.gui.scalebar_units.currentText()

                if pixel_resolution > 0:
                    for layer in layer_names:
                        self.viewer.layers[layer].scale = [1, pixel_resolution, pixel_resolution, ]

                        self.viewer.scale_bar.visible = True
                        self.viewer.scale_bar.unit = scalebar_units
                        self.viewer.reset_view()

            else:
                self.viewer.scale_bar.visible = False

        except:
            self.viewer.scale_bar.visible = False

    def _autoContrast(self):
        try:
            if self.gui.import_auto_contrast.isChecked():
                layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Nucleoid", "Classes", "center_lines", "Localisations", ]]

                if len(layer_names) != 0:
                    active_layer = layer_names[-1]

                    image_dims = tuple(list(self.viewer.dims.current_step[:-2]) + [...])

                    image = (self.viewer.layers[str(active_layer)].data[image_dims].copy())

                    crop = self.viewer.layers[str(active_layer)].corner_pixels[:, -2:]

                    [[y1, x1], [y2, x2]] = crop

                    image_crop = image[y1:y2, x1:x2]

                    contrast_limit = np.percentile(image_crop[image_crop != 0], (1, 99))
                    contrast_limit = [int(contrast_limit[0] * 0.5), int(contrast_limit[1] * 2), ]

                    if contrast_limit[1] > contrast_limit[0]:
                        self.viewer.layers[str(active_layer)].contrast_limits = contrast_limit

        except:
            pass

    def _updateFileName(self):
        try:
            current_fov = self.viewer.dims.current_step[0]
            active_layer = self.viewer.layers.selection.active

            image = self.viewer.layers[str(active_layer)].data[current_fov]
            metadata = self.viewer.layers[str(active_layer)].metadata[current_fov]

            viewer_text = ""

            if (self.gui.overlay_filename.isChecked() and "image_name" in metadata.keys()):
                viewer_text = f"File Name: {metadata['image_name']}"
            if self.gui.overlay_folder.isChecked() and "folder" in metadata.keys():
                viewer_text = viewer_text + f"\nFolder: {metadata['folder']}"
            if (self.gui.overlay_microscope.isChecked() and "microscope" in metadata.keys()):
                viewer_text = (viewer_text + f"\nMicroscope: {metadata['microscope']}")
            if (self.gui.overlay_datemodified.isChecked() and "date_modified" in metadata.keys()):
                viewer_text = (viewer_text + f"\nDate Modified: {metadata['date_modified']}")
            if (self.gui.overlay_content.isChecked() and "content" in metadata.keys()):
                viewer_text = viewer_text + f"\nContent: {metadata['content']}"
            if self.gui.overlay_strain.isChecked() and "strain" in metadata.keys():
                viewer_text = viewer_text + f"\nStrain: {metadata['strain']}"
            if (self.gui.overlay_phenotype.isChecked() and "phenotype" in metadata.keys()):
                viewer_text = (viewer_text + f"\nPhenotype: {metadata['phenotype']}")
            if (self.gui.overlay_antibiotic.isChecked() and "antibiotic" in metadata.keys()):
                viewer_text = (viewer_text + f"\nAntibiotic: {metadata['antibiotic']}")
            if self.gui.overlay_stain.isChecked() and "stain" in metadata.keys():
                viewer_text = viewer_text + f"\nStain: {metadata['stain']}"
            if (self.gui.overlay_staintarget.isChecked() and "stain_target" in metadata.keys()):
                viewer_text = (viewer_text + f"\nStain Target: {metadata['stain_target']}")
            if (self.gui.overlay_modality.isChecked() and "modality" in metadata.keys()):
                viewer_text = (viewer_text + f"\nModality: {metadata['modality']}")
            if (self.gui.overlay_lightsource.isChecked() and "source" in metadata.keys()):
                viewer_text = (viewer_text + f"\nLight Source: {metadata['source']}")
            if (self.gui.overlay_focus.isChecked() and "image_focus" in metadata.keys()):
                viewer_text = (viewer_text + f"\nImage Focus: {metadata['image_focus']}")
            if (self.gui.overlay_debris.isChecked() and "image_debris" in metadata.keys()):
                viewer_text = (viewer_text + f"\nImage Debris: {metadata['image_debris']}")
            if self.gui.overlay_laplacian.isChecked():
                image_laplacian = np.mean(cv2.Laplacian(image, cv2.CV_64F))
                viewer_text = viewer_text + f"\nLaplacian: {image_laplacian}"
            if self.gui.overlay_range.isChecked():
                image_range = np.max(image) - np.min(image)
                viewer_text = viewer_text + f"\nRange: {image_range}"

            if viewer_text != "":
                self.viewer.text_overlay.visible = True
                self.viewer.text_overlay.text = viewer_text.lstrip("\n")
            else:
                self.viewer.text_overlay.visible = False

        except:
            # print(traceback.format_exc())
            pass

    def _process_import(self, imported_data, rearrange=True):
        layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Nucleoid", "Classes", "center_lines", "Localisations", ]]

        append_mode = self.gui.import_append_mode.currentIndex()

        if append_mode == 0:
            # removes all layers (except segmentation layer)
            for layer_name in layer_names:
                self.viewer.layers.remove(self.viewer.layers[layer_name])
            # reset segmentation and class layers
            self.segLayer.data = np.zeros((1, 100, 100), dtype=np.uint16)
            self.nucLayer.data = np.zeros((1, 100, 100), dtype=np.uint16)
            self.classLayer.data = np.zeros((1, 100, 100), dtype=np.uint16)

        imported_images = imported_data["imported_images"]

        self.viewer.dims.set_current_step(0, 0)

        for layer_name, layer_data in imported_images.items():
            images = layer_data["images"]
            masks = layer_data["masks"]
            classes = layer_data["classes"]
            metadata = layer_data["metadata"]

            # for i in range(len(images)):
            #     print(metadata[i]["image_name"])

            if "nmasks" in layer_data.keys():
                nmasks = layer_data["nmasks"]
            else:
                nmasks = []

            new_image_stack, new_metadata = self.stack_images(images, metadata)
            new_mask_stack, new_metadata = self.stack_images(masks, metadata)
            new_nmask_stack, new_metadata = self.stack_images(nmasks, metadata)
            new_class_stack, new_metadata = self.stack_images(classes, metadata)

            if len(new_mask_stack) == 0:
                new_mask_stack = np.zeros(new_image_stack.shape, dtype=np.uint16)

            if len(new_nmask_stack) == 0:
                new_nmask_stack = np.zeros(new_image_stack.shape, dtype=np.uint16)

            if len(new_class_stack) == 0:
                new_class_stack = np.zeros(new_image_stack.shape, dtype=np.uint16)

            colormap = "gray"

            if layer_name == "405":
                colormap = "green"
            if layer_name == "532":
                colormap = "red"
            if layer_name == "Cy3":
                colormap = "red"
            if layer_name == "DAPI":
                colormap = "green"

            if append_mode == 1 and layer_name in layer_names:
                current_image_stack = self.viewer.layers[layer_name].data
                current_metadata = self.viewer.layers[layer_name].metadata
                current_mask_stack = self.segLayer.data
                current_nmask_stack = self.nucLayer.data
                current_class_stack = self.classLayer.data

                if len(current_image_stack) == 0:
                    setattr(self, layer_name, self.viewer.add_image(new_image_stack, name=layer_name, colormap=colormap, gamma=0.8, metadata=new_metadata, ), )

                    image_layer = getattr(self, layer_name)
                    image_layer.mouse_drag_callbacks.append(self._segmentationEvents)
                    image_layer.mouse_double_click_callbacks.append(self._doubeClickEvents)

                    self.segLayer.data = new_mask_stack
                    self.nucLayer.data = new_nmask_stack
                    self.classLayer.data = new_class_stack
                    self.segLayer.metadata = new_metadata
                    self.nucLayer.metadata = new_metadata
                    self.classLayer.metadata = new_metadata

                else:

                    (appended_image_stack, appended_metadata,) = self.append_image_stacks(current_metadata, new_metadata, current_image_stack, new_image_stack, )
                    (appended_mask_stack, appended_metadata,) = self.append_image_stacks(current_metadata, new_metadata, current_mask_stack, new_mask_stack, )
                    (appended_nmask_stack, appended_metadata,) = self.append_image_stacks(current_metadata, new_metadata, current_nmask_stack, new_nmask_stack, )
                    (appended_class_stack, appended_metadata,) = self.append_image_stacks(current_metadata, new_metadata, current_class_stack, new_class_stack, )

                    self.viewer.layers.remove(self.viewer.layers[layer_name])

                    setattr(self, layer_name, self.viewer.add_image(appended_image_stack, name=layer_name, colormap=colormap, gamma=0.8, metadata=appended_metadata, ), )

                    image_layer = getattr(self, layer_name)
                    image_layer.mouse_drag_callbacks.append(self._segmentationEvents)
                    image_layer.mouse_double_click_callbacks.append(self._doubeClickEvents)

                    self.segLayer.data = appended_mask_stack
                    self.nucLayer.data = appended_nmask_stack
                    self.classLayer.data = appended_class_stack
                    self.segLayer.metadata = appended_metadata
                    self.nucLayer.metadata = appended_metadata
                    self.classLayer.metadata = new_metadata

            else:
                if append_mode == 2:
                    if layer_name in layer_names:
                        import re

                        channel_indeces = [int(re.findall(r"\[(\d+)\]", layer)[-1]) for layer in layer_names if bool(re.search(r"\[\d+\]", layer)) == True]

                        if len(channel_indeces) == 0:
                            new_index = 1
                        else:
                            new_index = max(channel_indeces) + 1

                        layer_name = layer_name + f" [{new_index}]"

                setattr(self, layer_name, self.viewer.add_image(new_image_stack, name=layer_name, colormap=colormap, gamma=0.8, metadata=new_metadata, ), )

                image_layer = getattr(self, layer_name)
                image_layer.mouse_drag_callbacks.append(self._segmentationEvents)
                image_layer.mouse_double_click_callbacks.append(self._doubeClickEvents)

                self.segLayer.data = new_mask_stack
                self.nucLayer.data = new_nmask_stack
                self.classLayer.data = new_class_stack
                self.segLayer.metadata = new_metadata
                self.nucLayer.metadata = new_metadata
                self.classLayer.metadata = new_metadata

        # sets labels such that only label contours are shown
        self.segLayer.contour = 1
        self.segLayer.opacity = 1
        self.nucLayer.contour = 1
        self.nucLayer.opacity = 1

        self._reorderLayers()
        self._updateFileName()
        self._updateSegmentationCombo()
        self._updateSegChannels()
        self.gui.import_progressbar.reset()
        self.viewer.reset_view()
        self._autoClassify()
        self.align_image_channels()
        self._autoContrast()
        self._updateScaleBar()

    def _reorderLayers(self):
        try:
            layer_names = [layer.name for layer in self.viewer.layers if layer.name in ["Segmentations", "Nucleoid", "Classes", "center_lines"]]

            for layer in ["center_lines", "Localisations", "Classes", "Nucleoid", "Segmentations", ]:
                if layer in layer_names:
                    layer_index = self.viewer.layers.index(layer)
                    self.viewer.layers.move(layer_index, -1)

        except:
            pass

    def _autoClassify(self, reset=False, margin = 3):

        mask_stack = self.segLayer.data.copy()
        label_stack = self.classLayer.data.copy()

        for i in range(len(mask_stack)):
            mask = mask_stack[i, :, :]
            label = label_stack[i, :, :]

            label_ids = np.unique(label)
            mask_ids = np.unique(mask)

            if len(label_ids) == 1 or reset == True:
                label = np.zeros(label.shape, dtype=np.uint16)

                for mask_id in mask_ids:
                    if mask_id != 0:
                        cnt_mask = np.zeros(label.shape, dtype=np.uint8)
                        cnt_mask[mask == mask_id] = 255

                        cnt, _ = cv2.findContours(cnt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, )

                        x, y, w, h = cv2.boundingRect(cnt[0])
                        y1, y2, x1, x2 = y, (y + h), x, (x + w)

                        edge = False

                        if x1 < margin:
                            edge = True
                        if y1 < margin:
                            edge = True
                        if x2 > cnt_mask.shape[1] - margin:
                            edge = True
                        if y2 > cnt_mask.shape[0] - margin:
                            edge = True

                        if edge == True:
                            label[mask == mask_id] = 6
                        else:
                            label[mask == mask_id] = 1
                        #
                        # # appends contour to list if the bounding coordinates are along the edge of the image
                        # if (y1 > 0 and y2 < cnt_mask.shape[0] and x1 > 0 and x2 < cnt_mask.shape[1]):
                        #     label[mask == mask_id] = 1
                        #
                        # else:
                        #     label[mask == mask_id] = 6

            label_stack[i, :, :] = label

        self.classLayer.data = label_stack
