"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/guides.html?#widgets

Replace code below according to your needs.
"""

import os
import shutil
import sys
import tempfile
import time
import traceback
from functools import partial

import cv2
import napari
import numpy as np
import pandas as pd
from napari.utils.notifications import show_info
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from qtpy.QtCore import QObject, QRunnable, QThreadPool
from PyQt5.QtGui import QFont
from qtpy.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QProgressBar, QPushButton, QRadioButton, QSlider, QTabWidget, QVBoxLayout, QWidget, )

import napari_bacseg._utils
from napari_bacseg._utils import align_image_channels, unstack_images
from napari_bacseg._utils_worker import Worker
from napari_bacseg._utils_picasso import _picasso_utils

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"





def external_function(self, arg1=None):
    def event(viewer):
        pass  # print(arg1)

    return event


class ExampleQWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer

        btn = QPushButton("Click me!")
        # btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn)

        self.external_function = partial(external_function, self)

        viewer.bind_key("n", self.on_pressed(arg1=1, arg2=2))
        viewer.bind_key("e", self.external_function(arg1=1))

    def on_pressed(self, arg1, arg2):
        def _on_key_pressed(viewer):
            pass  # print(arg1, arg2)

        return _on_key_pressed


class BacSeg(QWidget, _picasso_utils):
    """Widget allows selection of two labels layers and returns a new layer
    highlighing pixels whose values differ between the two layers."""

    def __init__(self, viewer: napari.Viewer):
        """Initialize widget with two layer combo boxes and a run button"""

        super().__init__()

        # import functions
        from napari_bacseg._utils import _manualImport, stack_images
        from napari_bacseg._utils_cellpose import (_initialise_cellpose_model, _select_cellpose_save_directory, _select_cellpose_save_path, _select_custom_cellpose_model, train_cellpose_model, )
        from napari_bacseg._utils_database import (_create_bacseg_database, _load_bacseg_database, _show_database_controls, populate_upload_combos, update_database_metadata, update_upload_combos, )
        from napari_bacseg._utils_interface_events import (_copymasktoall, _delete_active_image, _deleteallmasks, _doubeClickEvents, _imageControls, _modify_channel_changed, _modifyMode, _segmentationEvents, _viewerControls, )
        from napari_bacseg._utils_oufti import (_update_active_midlines, centre_oufti_midlines, generate_midlines, midline_edit_toggle, update_midlines, )
        from napari_bacseg._utils_statistics import _compute_simple_cell_stats, _filter_cells
        from napari_bacseg._utils_tiler import (fold_images, unfold_images, update_image_folds, )
        from napari_bacseg.bacseg_ui import Ui_tab_widget

        self.populate_upload_combos = self.wrapper(populate_upload_combos)
        self.update_upload_combos = self.wrapper(update_upload_combos)
        self.update_database_metadata = self.wrapper(update_database_metadata)
        self.stack_image = self.wrapper(stack_images)
        self._modifyMode = self.wrapper(_modifyMode)
        self._viewerControls = self.wrapper(_viewerControls)
        self._copymasktoall = self.wrapper(_copymasktoall)
        self._deleteallmasks = self.wrapper(_deleteallmasks)
        self._delete_active_image = self.wrapper(_delete_active_image)
        self._imageControls = self.wrapper(_imageControls)
        self._segmentationEvents = self.wrapper(_segmentationEvents)
        self._modify_channel_changed = self.wrapper(_modify_channel_changed)
        self._manualImport = self.wrapper(_manualImport)
        self.train_cellpose_model = self.wrapper(train_cellpose_model)
        self._initialise_cellpose_model = self.wrapper(_initialise_cellpose_model)
        self._select_custom_cellpose_model = self.wrapper(_select_custom_cellpose_model)
        self._select_cellpose_save_directory = self.wrapper(_select_cellpose_save_directory)
        self._select_cellpose_save_path = self.wrapper(_select_cellpose_save_path)
        self.unfold_images = self.wrapper(unfold_images)
        self.fold_images = self.wrapper(fold_images)
        self.update_image_folds = self.wrapper(update_image_folds)
        self.midline_edit_toggle = self.wrapper(midline_edit_toggle)
        self.centre_oufti_midlines = self.wrapper(centre_oufti_midlines)
        self.generate_midlines = self.wrapper(generate_midlines)
        self.update_midlines = self.wrapper(update_midlines)
        self._update_active_midlines = self.wrapper(_update_active_midlines)
        self._create_bacseg_database = self.wrapper(_create_bacseg_database)
        self._load_bacseg_database = self.wrapper(_load_bacseg_database)
        self._show_database_controls = self.wrapper(_show_database_controls)
        self._doubeClickEvents = self.wrapper(_doubeClickEvents)
        self._compute_simple_cell_stats = self.wrapper(_compute_simple_cell_stats)
        self._filter_cells = self.wrapper(_filter_cells)

        application_path = os.path.dirname(sys.executable)
        self.viewer = viewer
        self.setLayout(QVBoxLayout())

        # ui_path = os.path.abspath(r"C:\napari-bacseg\src\napari_bacseg\bacseg_ui.ui")
        # self.bacseg_ui = uic.loadUi(ui_path)
        # command to refresh ui file: pyuic5 bacseg_ui.ui -o bacseg_ui.py

        self.form = Ui_tab_widget()
        self.bacseg_ui = QTabWidget()
        self.form.setupUi(self.bacseg_ui)

        for child in self.bacseg_ui.findChildren(QWidget):
            child.setFont(QFont("Arial", 10))

        # add widget_gui layout to main layout
        self.layout().addWidget(self.bacseg_ui)

        # general references from Qt Desinger References
        self.tab_widget = self.findChild(QTabWidget, "tab_widget")

        # import controls from Qt Desinger References
        self.path_list = []
        self.active_import_mode = ""
        self.import_mode = self.findChild(QComboBox, "import_mode")
        self.import_filemode = self.findChild(QComboBox, "import_filemode")
        self.import_precision = self.findChild(QComboBox, "import_precision")
        self.import_import = self.findChild(QPushButton, "import_import")
        self.import_limit = self.findChild(QComboBox, "import_limit")
        self.import_limit_label = self.findChild(QLabel, "import_limit_label")
        self.import_append_mode = self.findChild(QComboBox, "import_append_mode")

        # self.clear_previous = self.findChild(QCheckBox, "import_clear_previous")
        self.autocontrast = self.findChild(QCheckBox, "import_auto_contrast")
        self.import_multiframe_mode = self.findChild(QComboBox, "import_multiframe_mode")
        self.import_crop_mode = self.findChild(QComboBox, "import_crop_mode")
        self.channel_mode = self.findChild(QComboBox, "nim_channel_mode")
        self.import_progressbar = self.findChild(QProgressBar, "import_progressbar")
        self.import_align = self.findChild(QCheckBox, "import_align")

        self.img_modality = self.findChild(QComboBox, "img_modality")
        self.img_light_source = self.findChild(QComboBox, "img_light_source")
        self.img_stain = self.findChild(QComboBox, "img_stain")
        self.img_stain_target = self.findChild(QComboBox, "img_stain_target")

        self.label_overwrite = self.findChild(QPushButton, "label_overwrite")

        # view tab controls + variables from Qt Desinger References
        self.unfold_tile_size = self.findChild(QComboBox, "unfold_tile_size")
        self.unfold_tile_overlap = self.findChild(QComboBox, "unfold_tile_overlap")
        self.unfold_mode = self.findChild(QComboBox, "unfold_mode")
        self.fold = self.findChild(QPushButton, "fold")
        self.unfold = self.findChild(QPushButton, "unfold")
        self.unfold_progressbar = self.findChild(QPushButton, "unfold_progressbar")
        self.alignment_channel = self.findChild(QComboBox, "alignment_channel")
        self.align_active_image = self.findChild(QPushButton, "align_active_image")
        self.align_all_images = self.findChild(QPushButton, "align_all_images")
        self.undrift_channel = self.findChild(QComboBox, "undrift_channel")
        self.undrift_images = self.findChild(QPushButton, "undrift_images")
        self.undrift_progressbar = self.findChild(QProgressBar, "undrift_progressbar")

        self.picasso_image_channel = self.findChild(QComboBox, "picasso_image_channel")
        self.picasso_box_size = self.findChild(QComboBox, "picasso_box_size")
        self.picasso_min_net_gradient = self.findChild(QLineEdit, "picasso_min_net_gradient")
        self.picasso_detect = self.findChild(QPushButton, "picasso_detect")
        self.picasso_fit = self.findChild(QPushButton, "picasso_fit")
        self.picasso_vis_mode = self.findChild(QComboBox, "picasso_vis_mode")
        self.picasso_vis_size = self.findChild(QComboBox, "picasso_vis_size")
        self.picasso_vis_opacity = self.findChild(QComboBox, "picasso_vis_opacity")
        self.picasso_vis_edge_width = self.findChild(QComboBox, "picasso_vis_edge_width")
        self.picasso_show_vis = self.findChild(QCheckBox, "picasso_show_vis")
        self.picasso_image_frames = self.findChild(QComboBox, "picasso_image_frames")
        self.picasso_progressbar = self.findChild(QProgressBar, "picasso_progressbar")
        self.picasso_export = self.findChild(QPushButton, "picasso_export")
        self.picasso_export_mode = self.findChild(QComboBox, "picasso_export_mode")
        self.picasso_filter_localisations = self.findChild(QCheckBox, "picasso_filter_localisations")

        self.overlay_filename = self.findChild(QCheckBox, "overlay_filename")
        self.overlay_folder = self.findChild(QCheckBox, "overlay_folder")
        self.overlay_microscope = self.findChild(QCheckBox, "overlay_microscope")
        self.overlay_datemodified = self.findChild(QCheckBox, "overlay_datemodified")
        self.overlay_content = self.findChild(QCheckBox, "overlay_content")
        self.overlay_phenotype = self.findChild(QCheckBox, "overlay_phenotype")
        self.overlay_strain = self.findChild(QCheckBox, "overlay_strain")
        self.overlay_staintarget = self.findChild(QCheckBox, "overlay_staintarget")
        self.overlay_antibiotic = self.findChild(QCheckBox, "overlay_antibiotic")
        self.overlay_stain = self.findChild(QCheckBox, "overlay_stain")
        self.overlay_modality = self.findChild(QCheckBox, "overlay_modality")
        self.overlay_lightsource = self.findChild(QCheckBox, "overlay_lightsource")
        self.overlay_focus = self.findChild(QCheckBox, "overlay_focus")
        self.overlay_debris = self.findChild(QCheckBox, "overlay_debris")
        self.overlay_laplacian = self.findChild(QCheckBox, "overlay_laplacian")
        self.overlay_range = self.findChild(QCheckBox, "overlay_range")

        self.zoom_magnification = self.findChild(QComboBox, "zoom_magnification")
        self.zoom_apply = self.findChild(QPushButton, "zoom_apply")

        # cellpose controls + variables from Qt Desinger References
        self.cellpose_segmentation = False
        self.cellpose_model = None
        self.cellpose_custom_model_path = ""
        self.cellpose_train_model_path = ""
        self.cellpose_log_file = None
        self.cellpose_select_custom_model = self.findChild(QPushButton, "cellpose_select_custom_model")
        self.cellpose_segmodel = self.findChild(QComboBox, "cellpose_segmodel")
        self.cellpose_trainmodel = self.findChild(QComboBox, "cellpose_trainmodel")
        self.cellpose_segchannel = self.findChild(QComboBox, "cellpose_segchannel")
        self.cellpose_flowthresh = self.findChild(QSlider, "cellpose_flowthresh")
        self.cellpose_flowthresh_label = self.findChild(QLabel, "cellpose_flowthresh_label")
        self.cellpose_maskthresh = self.findChild(QSlider, "cellpose_maskthresh")
        self.cellpose_maskthresh_label = self.findChild(QLabel, "cellpose_maskthresh_label")
        self.cellpose_minsize = self.findChild(QSlider, "cellpose_minsize")
        self.cellpose_minsize_label = self.findChild(QLabel, "cellpose_minsize_label")
        self.cellpose_diameter = self.findChild(QSlider, "cellpose_diameter")
        self.cellpose_diameter_label = self.findChild(QLabel, "cellpose_diameter_label")
        self.cellpose_segment_active = self.findChild(QPushButton, "cellpose_segment_active")
        self.cellpose_segment_all = self.findChild(QPushButton, "cellpose_segment_all")
        self.cellpose_clear_previous = self.findChild(QCheckBox, "cellpose_clear_previous")
        self.cellpose_usegpu = self.findChild(QCheckBox, "cellpose_usegpu")
        self.cellpose_resetimage = self.findChild(QCheckBox, "cellpose_resetimage")
        self.cellpose_progressbar = self.findChild(QProgressBar, "cellpose_progressbar")
        self.cellpose_train_model = self.findChild(QPushButton, "cellpose_train_model")
        self.cellpose_save_dir = self.findChild(QPushButton, "cellpose_save_dir")
        self.cellpose_trainchannel = self.findChild(QComboBox, "cellpose_trainchannel")
        self.cellpose_nepochs = self.findChild(QComboBox, "cellpose_nepochs")
        self.cellpose_batchsize = self.findChild(QComboBox, "cellpose_batchsize")
        self.cellpose_seg_batchsize = self.findChild(QComboBox, "cellpose_seg_batchsize")
        self.cellpose_min_seg_size = self.findChild(QComboBox, "cellpose_min_seg_size")
        self.cellpose_seg_mode = self.findChild(QComboBox, "cellpose_seg_mode")
        self.cellpose_invert_images = self.findChild(QCheckBox, "cellpose_invert_images")
        self.cellpose_auto_classify = self.findChild(QCheckBox, "cellpose_auto_classify")

        # modify tab controls + variables from Qt Desinger References
        self.interface_mode = "panzoom"
        self.segmentation_mode = "add"
        self.class_mode = "single"
        self.class_colour = 1
        self.modify_panzoom = self.findChild(QPushButton, "modify_panzoom")
        self.modify_segment = self.findChild(QPushButton, "modify_segment")
        self.modify_classify = self.findChild(QPushButton, "modify_classify")
        self.modify_refine = self.findChild(QPushButton, "modify_refine")
        self.refine_channel = self.findChild(QComboBox, "refine_channel")
        self.refine_all = self.findChild(QPushButton, "refine_all")
        self.modify_copymasktoall = self.findChild(QPushButton, "modify_copymasktoall")
        self.modify_deleteallmasks = self.findChild(QPushButton, "modify_deleteallmasks")
        self.modify_deleteactivemasks = self.findChild(QPushButton, "modify_deleteactivemasks")
        self.modify_deleteactiveimage = self.findChild(QPushButton, "modify_deleteactiveimage")
        self.modify_deleteotherimages = self.findChild(QPushButton, "modify_deleteotherimages")
        self.modify_progressbar = self.findChild(QProgressBar, "modify_progressbar")
        self.modify_channel = self.findChild(QComboBox, "modify_channel")

        self.filter_metric = self.findChild(QComboBox, "filter_metric")
        self.filter_criteria = self.findChild(QComboBox, "filter_criteria")
        self.filter_threshold = self.findChild(QLineEdit, "filter_threshold")
        self.filter_mode = self.findChild(QComboBox, "filter_mode")
        self.filter_report = self.findChild(QPushButton, "filter_report")
        self.filter_remove = self.findChild(QPushButton, "filter_remove")
        self.filter_ignore_edge = self.findChild(QCheckBox, "filter_ignore_edge")

        self.modify_auto_panzoom = self.findChild(QCheckBox, "modify_auto_panzoom")
        self.modify_add = self.findChild(QPushButton, "modify_add")
        self.modify_extend = self.findChild(QPushButton, "modify_extend")
        self.modify_split = self.findChild(QPushButton, "modify_split")
        self.modify_join = self.findChild(QPushButton, "modify_join")
        self.modify_delete = self.findChild(QPushButton, "modify_delete")
        self.classify_single = self.findChild(QPushButton, "classify_single")
        self.classify_dividing = self.findChild(QPushButton, "classify_dividing")
        self.classify_divided = self.findChild(QPushButton, "classify_divided")
        self.classify_vertical = self.findChild(QPushButton, "classify_vertical")
        self.classify_broken = self.findChild(QPushButton, "classify_broken")
        self.classify_edge = self.findChild(QPushButton, "classify_edge")
        self.modify_viewmasks = self.findChild(QCheckBox, "modify_viewmasks")
        self.modify_viewlabels = self.findChild(QCheckBox, "modify_viewlabels")
        self.find_next = self.findChild(QPushButton, "find_next")
        self.find_previous = self.findChild(QPushButton, "find_previous")
        self.find_criterion = self.findChild(QComboBox, "find_criterion")
        self.find_mode = self.findChild(QComboBox, "find_mode")
        self.scalebar_show = self.findChild(QCheckBox, "scalebar_show")
        self.scalebar_resolution = self.findChild(QLineEdit, "scalebar_resolution")
        self.scalebar_units = self.findChild(QComboBox, "scalebar_units")

        self.set_quality_mode = self.findChild(QComboBox, "set_quality_mode")
        self.set_focus_0 = self.findChild(QPushButton, "set_focus_0")
        self.set_focus_1 = self.findChild(QPushButton, "set_focus_1")
        self.set_focus_2 = self.findChild(QPushButton, "set_focus_2")
        self.set_focus_3 = self.findChild(QPushButton, "set_focus_3")
        self.set_focus_4 = self.findChild(QPushButton, "set_focus_4")
        self.set_focus_5 = self.findChild(QPushButton, "set_focus_5")

        self.set_debris_1 = self.findChild(QPushButton, "set_debris_1")
        self.set_debris_2 = self.findChild(QPushButton, "set_debris_2")
        self.set_debris_3 = self.findChild(QPushButton, "set_debris_3")
        self.set_debris_4 = self.findChild(QPushButton, "set_debris_4")
        self.set_debris_5 = self.findChild(QPushButton, "set_debris_5")

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

        self.upload_initial = self.findChild(QComboBox, "upload_initial")
        self.upload_content = self.findChild(QComboBox, "upload_content")
        self.upload_microscope = self.findChild(QComboBox, "upload_microscope")
        self.upload_antibiotic = self.findChild(QComboBox, "upload_antibiotic")
        self.upload_phenotype = self.findChild(QComboBox, "upload_phenotype")
        self.upload_strain = self.findChild(QComboBox, "upload_strain")
        self.upload_abxconcentration = self.findChild(QComboBox, "upload_abxconcentration")
        self.upload_treatmenttime = self.findChild(QComboBox, "upload_treatmenttime")
        self.upload_mount = self.findChild(QComboBox, "upload_mount")
        self.upload_protocol = self.findChild(QComboBox, "upload_protocol")
        self.upload_overwrite_images = self.findChild(QCheckBox, "upload_overwrite_images")
        self.upload_overwrite_masks = self.findChild(QCheckBox, "upload_overwrite_masks")
        self.overwrite_selected_metadata = self.findChild(QCheckBox, "overwrite_selected_metadata")
        self.overwrite_all_metadata = self.findChild(QCheckBox, "overwrite_all_metadata")
        self.upload_all = self.findChild(QPushButton, "upload_all")
        self.upload_active = self.findChild(QPushButton, "upload_active")
        self.database_download = self.findChild(QPushButton, "database_download")
        self.database_download_limit = self.findChild(QComboBox, "database_download_limit")
        self.create_database = self.findChild(QPushButton, "create_database")
        self.load_database = self.findChild(QPushButton, "load_database")
        self.display_database_path = self.findChild(QLineEdit, "display_database_path")
        self.upload_progressbar = self.findChild(QProgressBar, "upload_progressbar")
        self.download_progressbar = self.findChild(QProgressBar, "download_progressbar")
        self.upload_tab = self.findChild(QWidget, "upload_tab")
        self.upload_segmentation_combo = self.findChild(QComboBox, "upload_segmentation_combo")
        self.upload_label_combo = self.findChild(QComboBox, "upload_label_combo")
        self.download_sort_order_1 = self.findChild(QComboBox, "download_sort_order_1")
        self.download_sort_order_2 = self.findChild(QComboBox, "download_sort_order_2")
        self.download_sort_direction_1 = self.findChild(QComboBox, "download_sort_direction_1")
        self.download_sort_direction_2 = self.findChild(QComboBox, "download_sort_direction_2")
        self.store_metadata = self.findChild(QPushButton, "store_metadata")
        self.filter_metadata = self.findChild(QPushButton, "filter_metadata")
        self.reset_metadata = self.findChild(QPushButton, "reset_metadata")
        self.upload_images_setting = self.findChild(QCheckBox, "upload_images_setting")
        self.upload_segmentations_setting = self.findChild(QCheckBox, "upload_segmentations_setting")
        self.upload_metadata_setting = self.findChild(QCheckBox, "upload_metadata_setting")

        self.image_metadata_controls = self.findChild(QFormLayout, "image_metadata_controls")

        self._show_database_controls(False)

        # oufti tab controls
        self.oufti_generate_all_midlines = self.findChild(QPushButton, "oufti_generate_all_midlines")
        self.oufti_generate_active_midlines = self.findChild(QPushButton, "oufti_generate_active_midlines")
        self.oufti_panzoom_mode = self.findChild(QRadioButton, "oufti_panzoom_mode")
        self.oufti_edit_mode = self.findChild(QRadioButton, "oufti_edit_mode")
        self.oufti_midline_vertexes = self.findChild(QComboBox, "oufti_midline_vertexes")
        self.oufti_centre_all_midlines = self.findChild(QPushButton, "oufti_centre_all_midlines")
        self.oufti_centre_active_midlines = self.findChild(QPushButton, "oufti_centre_active_midlines")
        self.oufti_mesh_length = self.findChild(QComboBox, "oufti_mesh_length")
        self.oufti_mesh_dilation = self.findChild(QComboBox, "oufti_mesh_dilation")

        # export tab controls from Qt Desinger References
        self.export_channel = self.findChild(QComboBox, "export_channel")
        self.export_mode = self.findChild(QComboBox, "export_mode")
        self.export_location = self.findChild(QComboBox, "export_location")
        self.export_modifier = self.findChild(QLineEdit, "export_modifier")
        self.export_single = self.findChild(QCheckBox, "export_single")
        self.export_dividing = self.findChild(QCheckBox, "export_dividing")
        self.export_divided = self.findChild(QCheckBox, "export_divided")
        self.export_vertical = self.findChild(QCheckBox, "export_vertical")
        self.export_broken = self.findChild(QCheckBox, "export_broken")
        self.export_edge = self.findChild(QCheckBox, "export_edge")
        self.export_statistics_multithreaded = self.findChild(QCheckBox, "export_statistics_multithreaded")
        self.export_active = self.findChild(QPushButton, "export_active")
        self.export_all = self.findChild(QPushButton, "export_all")
        self.export_normalise = self.findChild(QCheckBox, "export_normalise")
        self.export_invert = self.findChild(QCheckBox, "export_invert")
        self.export_scalebar = self.findChild(QCheckBox, "export_scalebar")
        self.export_scalebar_resolution = self.findChild(QLineEdit, "export_scalebar_resolution")
        self.export_scalebar_resolution_units = self.findChild(QComboBox, "export_scalebar_resolution_units")
        self.export_scalebar_size = self.findChild(QLineEdit, "export_scalebar_size")
        self.export_scalebar_size_units = self.findChild(QComboBox, "export_scalebar_size_units")
        self.export_scalebar_colour = self.findChild(QComboBox, "export_scalebar_colour")
        self.export_scalebar_thickness = self.findChild(QComboBox, "export_scalebar_thickness")
        self.export_cropzoom = self.findChild(QCheckBox, "export_crop_zoom")
        self.export_mask_background = self.findChild(QCheckBox, "export_mask_background")

        self.export_stack_channel = self.findChild(QComboBox, "export_stack_channel")
        self.export_stack_mode = self.findChild(QComboBox, "export_stack_mode")
        self.export_stack_location = self.findChild(QComboBox, "export_stack_location")
        self.export_stack_modifier = self.findChild(QLineEdit, "export_stack_modifier")
        self.export_stack_image_setting = self.findChild(QCheckBox, "export_stack_image_setting")
        self.export_stack_overwrite_setting = self.findChild(QCheckBox, "export_stack_overwrite_setting")
        self.export_stack_active = self.findChild(QPushButton, "export_stack_active")
        self.export_stack_all = self.findChild(QPushButton, "export_stack_all")

        self.export_autocontrast = self.findChild(QCheckBox, "export_autocontrast")
        self.export_statistics_pixelsize = self.findChild(QLineEdit, "export_statistics_pixelsize")
        self.export_statistics_active = self.findChild(QPushButton, "export_statistics_active")
        self.export_statistics_all = self.findChild(QPushButton, "export_statistics_all")
        self.export_colicoords_mode = self.findChild(QComboBox, "export_colicoords_mode")
        self.export_progressbar = self.findChild(QProgressBar, "export_progressbar")
        self.export_image_setting = self.findChild(QCheckBox, "export_image_setting")
        self.export_overwrite_setting = self.findChild(QCheckBox, "export_overwrite_setting")
        self.export_directory = ""

        # import events
        self.autocontrast.stateChanged.connect(self._autoContrast)
        self.import_import.clicked.connect(self._importDialog)
        self.label_overwrite.clicked.connect(self.overwrite_channel_info)
        self.import_mode.currentIndexChanged.connect(self._set_available_multiframe_modes)

        # view events
        self.fold.clicked.connect(self.fold_images)
        self.unfold.clicked.connect(self.unfold_images)
        self.tiler_object = None
        self.tile_dict = {"Segmentations": [], "Classes": [], "Nucleoid": []}
        self.unfolded = False

        self.align_active_image.clicked.connect(partial(self._align_images, mode="active"))
        self.align_all_images.clicked.connect(partial(self._align_images, mode="all"))
        self.undrift_images.clicked.connect(self._undrift_images)

        self.scalebar_show.stateChanged.connect(self._updateScaleBar)
        self.scalebar_resolution.textChanged.connect(self._updateScaleBar)
        self.scalebar_units.currentTextChanged.connect(self._updateScaleBar)
        self.overlay_filename.stateChanged.connect(self._updateFileName)
        self.overlay_folder.stateChanged.connect(self._updateFileName)
        self.overlay_microscope.stateChanged.connect(self._updateFileName)
        self.overlay_datemodified.stateChanged.connect(self._updateFileName)
        self.overlay_content.stateChanged.connect(self._updateFileName)
        self.overlay_phenotype.stateChanged.connect(self._updateFileName)
        self.overlay_strain.stateChanged.connect(self._updateFileName)
        self.overlay_antibiotic.stateChanged.connect(self._updateFileName)
        self.overlay_stain.stateChanged.connect(self._updateFileName)
        self.overlay_staintarget.stateChanged.connect(self._updateFileName)
        self.overlay_modality.stateChanged.connect(self._updateFileName)
        self.overlay_lightsource.stateChanged.connect(self._updateFileName)
        self.overlay_focus.stateChanged.connect(self._updateFileName)
        self.overlay_debris.stateChanged.connect(self._updateFileName)
        self.overlay_laplacian.stateChanged.connect(self._updateFileName)
        self.overlay_range.stateChanged.connect(self._updateFileName)
        self.zoom_apply.clicked.connect(self._applyZoom)

        # cellpose events
        self.cellpose_flowthresh.valueChanged.connect(lambda: self._updateSliderLabel("cellpose_flowthresh", "cellpose_flowthresh_label"))
        self.cellpose_maskthresh.valueChanged.connect(lambda: self._updateSliderLabel("cellpose_maskthresh", "cellpose_maskthresh_label"))
        self.cellpose_minsize.valueChanged.connect(lambda: self._updateSliderLabel("cellpose_minsize", "cellpose_minsize_label"))
        self.cellpose_diameter.valueChanged.connect(lambda: self._updateSliderLabel("cellpose_diameter", "cellpose_diameter_label"))

        self.cellpose_select_custom_model.clicked.connect(self._select_custom_cellpose_model)
        self.cellpose_save_dir.clicked.connect(self._select_cellpose_save_directory)
        self.cellpose_segment_all.clicked.connect(self._segmentAll)
        self.cellpose_segment_active.clicked.connect(self._segmentActive)
        self.cellpose_train_model.clicked.connect(self._trainCellpose)
        self.cellpose_segchannel.currentTextChanged.connect(self._updateSegChannels)

        # modify tab events
        self.modify_panzoom.clicked.connect(self._modifyMode(mode="panzoom"))
        self.modify_segment.clicked.connect(self._modifyMode(mode="segment"))
        self.modify_classify.clicked.connect(self._modifyMode(mode="classify"))
        self.modify_refine.clicked.connect(self._modifyMode(mode="refine"))
        self.modify_add.clicked.connect(self._modifyMode(mode="add"))
        self.modify_extend.clicked.connect(self._modifyMode(mode="extend"))
        self.modify_join.clicked.connect(self._modifyMode(mode="join"))
        self.modify_split.clicked.connect(self._modifyMode(mode="split"))
        self.modify_delete.clicked.connect(self._modifyMode(mode="delete"))
        self.classify_single.clicked.connect(self._modifyMode(mode="single"))
        self.classify_dividing.clicked.connect(self._modifyMode(mode="dividing"))
        self.classify_divided.clicked.connect(self._modifyMode(mode="divided"))
        self.classify_vertical.clicked.connect(self._modifyMode(mode="vertical"))
        self.classify_broken.clicked.connect(self._modifyMode(mode="broken"))
        self.classify_edge.clicked.connect(self._modifyMode(mode="edge"))

        self.viewer.bind_key(key="b", func=self.set_blurred, overwrite=True)
        self.set_focus_1.clicked.connect(partial(self.set_image_quality, mode="focus", value=1))
        self.set_focus_2.clicked.connect(partial(self.set_image_quality, mode="focus", value=2))
        self.set_focus_3.clicked.connect(partial(self.set_image_quality, mode="focus", value=3))
        self.set_focus_4.clicked.connect(partial(self.set_image_quality, mode="focus", value=4))
        self.set_focus_5.clicked.connect(partial(self.set_image_quality, mode="focus", value=5))
        self.viewer.bind_key(key="f", func=self.set_focused, overwrite=True)

        self.set_debris_1.clicked.connect(partial(self.set_image_quality, mode="debris", value=1))
        self.set_debris_2.clicked.connect(partial(self.set_image_quality, mode="debris", value=2))
        self.set_debris_3.clicked.connect(partial(self.set_image_quality, mode="debris", value=3))
        self.set_debris_4.clicked.connect(partial(self.set_image_quality, mode="debris", value=4))
        self.set_debris_5.clicked.connect(partial(self.set_image_quality, mode="debris", value=5))

        self.modify_viewmasks.stateChanged.connect(self._viewerControls("viewmasks"))
        self.modify_viewlabels.stateChanged.connect(self._viewerControls("viewlabels"))
        self.refine_all.clicked.connect(self._refine_bacseg)
        self.modify_copymasktoall.clicked.connect(self._copymasktoall)
        self.modify_deleteallmasks.clicked.connect(self._deleteallmasks(mode="all"))
        self.modify_deleteactivemasks.clicked.connect(self._deleteallmasks(mode="active"))
        self.modify_deleteactiveimage.clicked.connect(self._delete_active_image(mode="active"))
        self.modify_deleteotherimages.clicked.connect(self._delete_active_image(mode="other"))
        self.find_next.clicked.connect(self._sort_cells("next"))
        self.find_previous.clicked.connect(self._sort_cells("previous"))
        self.modify_channel.currentTextChanged.connect(self._modify_channel_changed)

        self.filter_report.clicked.connect(partial(self._filter_segmentations, remove=False))
        self.filter_remove.clicked.connect(partial(self._filter_segmentations, remove=True))

        # export events
        self.export_active.clicked.connect(self._export("active"))
        self.export_all.clicked.connect(self._export("all"))
        self.export_stack_active.clicked.connect(self._export_stack("active"))
        self.export_stack_all.clicked.connect(self._export_stack("all"))
        self.export_statistics_active.clicked.connect(self._export_statistics("active"))
        self.export_statistics_all.clicked.connect(self._export_statistics("all"))

        # oufti events
        self.oufti_generate_all_midlines.clicked.connect(self.generate_midlines(mode="all"))
        self.oufti_generate_active_midlines.clicked.connect(self.generate_midlines(mode="active"))
        self.viewer.bind_key(key="m", func=self.midline_edit_toggle, overwrite=True)
        self.oufti_edit_mode.clicked.connect(self.midline_edit_toggle)
        self.oufti_panzoom_mode.clicked.connect(self.midline_edit_toggle)
        self.oufti_centre_all_midlines.clicked.connect(self.centre_oufti_midlines(mode="all"))
        self.oufti_centre_active_midlines.clicked.connect(self.centre_oufti_midlines(mode="active"))

        # upload tab events
        self.upload_all.clicked.connect(self._uploadDatabase(mode="all"))
        self.upload_active.clicked.connect(self._uploadDatabase(mode="active"))
        self.database_download.clicked.connect(self._downloadDatabase)
        self.create_database.clicked.connect(self._create_bacseg_database)
        self.load_database.clicked.connect(self._load_bacseg_database)

        self.picasso_detect.clicked.connect(self.detect_picasso_localisations)
        self.picasso_fit.clicked.connect(self.fit_picasso_localisations)
        self.picasso_vis_size.currentIndexChanged.connect(self.update_localisation_visualisation)
        self.picasso_vis_mode.currentIndexChanged.connect(self.update_localisation_visualisation)
        self.picasso_vis_opacity.currentIndexChanged.connect(self.update_localisation_visualisation)
        self.picasso_vis_edge_width.currentIndexChanged.connect(self.update_localisation_visualisation)
        self.picasso_export.clicked.connect(self.export_picasso_localisations)

        self.upload_initial.currentTextChanged.connect(self.populate_upload_combos)

        self.filter_metadata.clicked.connect(self.update_upload_combos)
        self.reset_metadata.clicked.connect(self.populate_upload_combos)

        self.updating_combos = False

        self.store_metadata.clicked.connect(self.update_database_metadata)

        # viewer event that call updateFileName when the slider is modified
        self.contours = []
        self.viewer.dims.events.current_step.connect(self._sliderEvent)

        # self.segImage = self.viewer.add_image(np.zeros((1,100,100),dtype=np.uint16),name="Image")
        self.class_colours = {1: (255 / 255, 255 / 255, 255 / 255, 1), 2: (0 / 255, 255 / 255, 0 / 255, 1), 3: (0 / 255, 170 / 255, 255 / 255, 1), 4: (170 / 255, 0 / 255, 255 / 255, 1), 5: (
        255 / 255, 170 / 255, 0 / 255, 1), 6: (255 / 255, 0 / 255, 0 / 255, 1), }

        self.classLayer = self.viewer.add_labels(np.zeros((1, 100, 100), dtype=np.uint16), opacity=0.25, name="Classes", color=self.class_colours, metadata={0: {"image_name": ""}}, visible=False, )
        self.nucLayer = self.viewer.add_labels(np.zeros((1, 100, 100), dtype=np.uint16), opacity=1, name="Nucleoid", metadata={0: {"image_name": ""}}, )
        self.segLayer = self.viewer.add_labels(np.zeros((1, 100, 100), dtype=np.uint16), opacity=1, name="Segmentations", metadata={0: {"image_name": ""}}, )

        self.segLayer.contour = 1

        # keyboard events, only triggered when viewer is not empty (an image is loaded/active)
        self.viewer.bind_key(key="a", func=self._modifyMode(mode="add"), overwrite=True)
        self.viewer.bind_key(key="e", func=self._modifyMode(mode="extend"), overwrite=True)
        self.viewer.bind_key(key="j", func=self._modifyMode(mode="join"), overwrite=True)
        self.viewer.bind_key(key="s", func=self._modifyMode(mode="split"), overwrite=True)
        self.viewer.bind_key(key="d", func=self._modifyMode(mode="delete"), overwrite=True)
        self.viewer.bind_key(key="r", func=self._modifyMode(mode="refine"), overwrite=True)
        self.viewer.bind_key(key="k", func=self._modifyMode(mode="clicktozoom"), overwrite=True)
        self.viewer.bind_key(key="Control-1", func=self._modifyMode(mode="single"), overwrite=True, )
        self.viewer.bind_key(key="Control-2", func=self._modifyMode(mode="dividing"), overwrite=True, )
        self.viewer.bind_key(key="Control-3", func=self._modifyMode(mode="divided"), overwrite=True, )
        self.viewer.bind_key(key="Control-4", func=self._modifyMode(mode="vertical"), overwrite=True, )
        self.viewer.bind_key(key="Control-5", func=self._modifyMode(mode="broken"), overwrite=True, )
        self.viewer.bind_key(key="Control-6", func=self._modifyMode(mode="edge"), overwrite=True)
        self.viewer.bind_key(key="F1", func=self._modifyMode(mode="panzoom"), overwrite=True)
        self.viewer.bind_key(key="F2", func=self._modifyMode(mode="segment"), overwrite=True)
        self.viewer.bind_key(key="F3", func=self._modifyMode(mode="classify"), overwrite=True)
        self.viewer.bind_key(key="h", func=self._viewerControls("h"), overwrite=True)
        self.viewer.bind_key(key="i", func=self._viewerControls("i"), overwrite=True)
        self.viewer.bind_key(key="o", func=self._viewerControls("o"), overwrite=True)
        self.viewer.bind_key(key="x", func=self._viewerControls("x"), overwrite=True)
        self.viewer.bind_key(key="z", func=self._viewerControls("z"), overwrite=True)
        self.viewer.bind_key(key="c", func=self._viewerControls("c"), overwrite=True)
        self.viewer.bind_key(key="Right", func=self._imageControls("Right"), overwrite=True)
        self.viewer.bind_key(key="Left", func=self._imageControls("Left"), overwrite=True)
        self.viewer.bind_key(key="u", func=self._imageControls("Upload"), overwrite=True)
        self.viewer.bind_key(key="Control-d", func=self._deleteallmasks(mode="active"), overwrite=True, )
        self.viewer.bind_key(key="Control-Shift-d", func=self._deleteallmasks(mode="all"), overwrite=True, )
        self.viewer.bind_key(key="Control-i", func=self._delete_active_image(mode="active"), overwrite=True, )
        self.viewer.bind_key(key="Control-Shift-i", func=self._delete_active_image(mode="other"), overwrite=True, )

        self.viewer.bind_key(key="Control-l", func=self._downloadDatabase(), overwrite=True)
        self.viewer.bind_key(key="Control-u", func=self._uploadDatabase(mode="active"), overwrite=True, )
        self.viewer.bind_key(key="Control-Shift-u", func=self._uploadDatabase(mode="all"), overwrite=True, )
        #
        self.viewer.bind_key(key="Control-Left", func=self._manual_align_channels("left", mode="active"), overwrite=True, )
        self.viewer.bind_key(key="Control-Right", func=self._manual_align_channels("right", mode="active"), overwrite=True, )
        self.viewer.bind_key(key="Control-Up", func=self._manual_align_channels("up", mode="active"), overwrite=True, )
        self.viewer.bind_key(key="Control-Down", func=self._manual_align_channels("down", mode="active"), overwrite=True, )

        self.viewer.bind_key(key="Alt-Left", func=self._manual_align_channels("left", mode="all"), overwrite=True, )
        self.viewer.bind_key(key="Alt-Right", func=self._manual_align_channels("right", mode="all"), overwrite=True, )
        self.viewer.bind_key(key="Alt-Up", func=self._manual_align_channels("up", mode="all"), overwrite=True, )
        self.viewer.bind_key(key="Alt-Down", func=self._manual_align_channels("down", mode="all"), overwrite=True, )

        self.import_filemode.currentIndexChanged.connect(self.update_import_limit)
        self.import_mode.currentIndexChanged.connect(self.update_import_limit)
        self.update_import_limit()

        # mouse events
        self.segLayer.mouse_drag_callbacks.append(self._segmentationEvents)
        self.nucLayer.mouse_drag_callbacks.append(self._segmentationEvents)

        # self.segLayer.mouse_move_callbac1ks.append(self._zoomEvents)
        self.segLayer.mouse_double_click_callbacks.append(self._doubeClickEvents)

        # viewer events
        self.viewer.layers.events.inserted.connect(self._manualImport)
        self.viewer.layers.events.removed.connect(self._updateSegmentationCombo)
        self.viewer.layers.selection.events.changed.connect(self._updateFileName)

        self.threadpool = QThreadPool()  # self.load_dev_data()

        self.widget_notifications = True



    def _check_number_string(self, string):

        try:
            float(string)
            return True
        except:
            return False

    def _filter_segmentations(self,viewer = None, remove = True):

        try:

            metric = self.filter_metric.currentText()
            criteria = self.filter_criteria.currentText()
            threshold = self.filter_threshold.text()
            fov_mode = self.filter_mode.currentText()
            ignore_edge = self.filter_ignore_edge.isChecked()

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


    def _undrift_images(self):
        worker = Worker(self._undrift)
        worker.signals.progress.connect(partial(self._Progresbar, progressbar="undrift"))
        worker.signals.finished.connect(self._undrift_postprocesing)
        self.threadpool.start(worker)

    @staticmethod
    def _undrift_preprocesing(img):
        from skimage import exposure

        if np.max(img) > 0:
            img = img.copy()
            v_min, v_max = np.percentile(img[img != 0], (0.1, 99.9))
            img = exposure.rescale_intensity(img, in_range=(v_min, v_max))

        img = cv2.GaussianBlur(img, (5, 5), 0)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return img

    def _undrift_postprocesing(self):
        try:
            # remove borders
            undrift_channel = self.undrift_channel.currentText()

            boundary_image = np.min(self.viewer.layers[undrift_channel].data.copy(), axis=0)
            boundary, _ = self._find_shifted_boundary(boundary_image)

            layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["center_lines"]]

            for layer in layer_names:
                self.viewer.layers[layer].data = self.viewer.layers[layer].data[:, boundary[1]: boundary[3], boundary[0]: boundary[2]]
                image_shape = self.viewer.layers[layer].data.shape

                for i in range(image_shape[0]):
                    if layer not in ["center_lines"]:
                        try:
                            self.viewer.layers[layer].metadata[i]["dims"] = [image_shape[-1], image_shape[-2], ]
                            self.viewer.layers[layer].metadata[i]["crop"] = [0, image_shape[-2], 0, image_shape[-1], ]
                        except:
                            pass

            # refresh active layer
            self.viewer.layers[self.undrift_channel.currentText()].refresh()

            # reset viewer
            self.viewer.reset_view()

        except:
            print(traceback.format_exc())

    def _find_shifted_boundary(self, image):
        x0, y0, x1, y1 = 0, 0, image.shape[1], image.shape[0]

        # Find non-black pixels
        coords = np.column_stack(np.where(image > 0))

        if coords.size != 0:
            y_centre_slice = image[:, int(image.shape[1] / 2)]
            x_centre_slice = image[int(image.shape[0] / 2), :]

            y0 = np.where(y_centre_slice > 0)[0][0]
            y1 = np.where(y_centre_slice > 0)[0][-1]
            x0 = np.where(x_centre_slice > 0)[0][0]
            x1 = np.where(x_centre_slice > 0)[0][-1]

            if y1 - y0 > 10 and x1 - x0 > 10:
                if y0 < 0:
                    y0 = 0
                if y1 > image.shape[0]:
                    y1 = image.shape[0]
                if x0 < 0:
                    x0 = 0
                if x1 > image.shape[1]:
                    x1 = image.shape[1]

        image_shape = y1 - y0, x1 - x0

        return [x0, y0, x1, y1], image_shape

    def _undrift(self, progress_callback):
        try:
            import scipy
            from skimage.registration import phase_cross_correlation

            layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Nucleoid", "Classes", "center_lines", "Localisations", ]]

            if layer_names != []:
                shift_list = []

                undrift_channel = self.undrift_channel.currentText()

                image_shape = self.viewer.layers[undrift_channel].data.shape

                if len(image_shape) == 3:
                    if image_shape[0] > 1:
                        anchor_binary = self._undrift_preprocesing(self.viewer.layers[undrift_channel].data[0])

                        for i in range(image_shape[0] - 1):
                            progress = int((i + 1) / (image_shape[0] - 1) * 100)
                            progress_callback.emit(progress)

                            target_image = self.viewer.layers[undrift_channel].data[i + 1]
                            target_binary = self._undrift_preprocesing(target_image)

                            shift, error, diffphase = phase_cross_correlation(anchor_binary, target_binary, upsample_factor=100, )

                            shift_list.append(shift)

                        for layer in layer_names:
                            for i in range(image_shape[0] - 1):
                                shifted_image = scipy.ndimage.shift(self.viewer.layers[layer].data[i + 1], shift_list[i], cval=-1, )

                                self.viewer.layers[layer].data[i + 1] = shifted_image

                        # boundary_image = np.min(self.viewer.layers[undrift_channel].data.copy(), axis=0)  # boundary, _ = self._find_shifted_boundary(boundary_image)  #  # layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["center_lines"]]  #  # for layer in layer_names:  #     self.viewer.layers[layer].data = self.viewer.layers[layer].data[:, boundary[1]:boundary[3], boundary[0]:boundary[2]]  #     frame = self.viewer.layers[layer].data[0].copy()  #  #     for i in range(image_shape[0] - 1):  #         if layer not in ["Segmentations", "Nucleoid", "Classes"]:  #             self.viewer.layers[layer].metadata[i]["dims"] = [frame.shape[-1], frame.shape[-2]]  #             self.viewer.layers[layer].metadata[i]["crop"] = [0, frame.shape[-2], 0, frame.shape[-1]]

                return

        except:
            print(traceback.format_exc())

    def _set_available_multiframe_modes(self):
        multiframe_items = [self.import_multiframe_mode.itemText(i) for i in range(self.import_multiframe_mode.count())]
        mode = self.import_mode.currentText()

        if mode in ["Images", "NanoImager Data", "ImageJ files(s)"]:
            if "Keep All Frames (BETA)" not in multiframe_items:
                self.import_multiframe_mode.addItem("Keep All Frames (BETA)")
        else:
            if "Keep All Frames (BETA)" in multiframe_items:
                self.import_multiframe_mode.removeItem(multiframe_items.index("Keep All Frames (BETA)"))

    def _applyZoom(self):
        try:
            import re

            magnification = self.zoom_magnification.currentText()
            pixel_resolution = float(self.scalebar_resolution.text())
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

                alignment_channel = self.alignment_channel.currentText()
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

            update_mode = self.set_quality_mode.currentIndex()

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

                img_modality = self.img_modality.currentText()
                img_light_source = self.img_light_source.currentText()
                img_stain = self.img_stain.currentText()
                img_stain_target = self.img_stain_target.currentText()

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
            multithreaded = self.export_statistics_multithreaded.isChecked()

            if self.unfolded == True:
                self.fold_images()

            pixel_size = float(self.export_statistics_pixelsize.text())

            colicoords_channel = self.export_colicoords_mode.currentText()
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

                from napari_bacseg._utils_statistics import (get_cell_statistics, process_cell_statistics, )

                self.get_cell_statistics = self.wrapper(get_cell_statistics)
                self.process_cell_statistics = self.wrapper(process_cell_statistics)

                worker = Worker(self.get_cell_statistics, mode=mode, pixel_size=pixel_size, colicoords_dir=colicoords_dir, )
                worker.signals.progress.connect(partial(self._Progresbar, progressbar="export"))
                worker.signals.result.connect(partial(self.process_cell_statistics, path=path))
                self.threadpool.start(worker)
                cell_data = worker.result()

                if self.export_colicoords_mode.currentIndex() != 0:
                    from napari_bacseg._utils_colicoords import run_colicoords

                    self.run_colicoords = self.wrapper(run_colicoords)

                    worker = Worker(self.run_colicoords, cell_data=cell_data, colicoords_channel=colicoords_channel, pixel_size=pixel_size, statistics=True, multithreaded=multithreaded, )

                    worker.signals.progress.connect(partial(self._Progresbar, progressbar="export"))
                    worker.signals.result.connect(partial(self.process_cell_statistics, path=path))
                    self.threadpool.start(worker)

        return _event

    def update_import_limit(self):
        if (self.import_filemode.currentIndex() == 1 or self.import_mode.currentText() == "Zeiss (.czi) Data"):
            self.import_limit.setEnabled(True)
            self.import_limit.setCurrentIndex(0)
            self.import_limit.show()
            self.import_limit_label.show()

            if self.import_mode.currentText() == "Zeiss (.czi) Data":
                self.import_limit_label.setText("Import Limit (FOV)")
            else:
                self.import_limit_label.setText("Import Limit (Files)")

        else:
            self.import_limit.setEnabled(False)
            self.import_limit.setCurrentIndex(6)
            self.import_limit.hide()
            self.import_limit_label.hide()

    def _sort_cells(self, order):
        def _event(viewer):
            if self.unfolded == True:
                self.fold_images()

            try:
                current_fov = self.viewer.dims.current_step[0]

                meta = self.segLayer.metadata[current_fov]

                self._compute_simple_cell_stats()

                find_criterion = self.find_criterion.currentText()
                find_mode = self.find_mode.currentText()

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

        pixel_size = float(self.export_statistics_pixelsize.text())

        if pixel_size <= 0:
            pixel_size = 1

        current_fov = self.viewer.dims.current_step[0]

        channel = self.refine_channel.currentText()
        colicoords_channel = channel.replace("Mask + ", "")

        mask_stack = self.segLayer.data
        mask = mask_stack[current_fov, :, :].copy()

        from napari_bacseg._utils_colicoords import (process_colicoords, run_colicoords, )
        from napari_bacseg._utils_statistics import get_cell_statistics

        self.get_cell_statistics = self.wrapper(get_cell_statistics)
        self.run_colicoords = self.wrapper(run_colicoords)
        self.process_colicoords = self.wrapper(process_colicoords)

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
            try:
                if (self.database_path != "" and os.path.exists(self.database_path) == True):
                    if self.unfolded == True:
                        self.fold_images()

                    if self.upload_initial.currentText() in ["", "Required for upload", ]:
                        show_info("Please select the user initial.")
                    else:
                        from napari_bacseg._utils_database_IO import (_upload_bacseg_database, )

                        self._upload_bacseg_database = self.wrapper(_upload_bacseg_database)

                        worker = Worker(self._upload_bacseg_database, mode=mode)
                        worker.signals.progress.connect(partial(self._Progresbar, progressbar="database_upload"))
                        self.threadpool.start(worker)
            except:
                pass

        return _event

    def _downloadDatabase(self, viewer=None):
        try:
            if (self.database_path != "" and os.path.exists(self.database_path) == True):
                if self.unfolded == True:
                    self.fold_images()

                if self.upload_initial.currentText() in ["", "Required for upload", ]:
                    show_info("Please select the user initial.")

                else:
                    from napari_bacseg._utils_database_IO import (get_filtered_database_metadata, read_bacseg_images, )

                    self.get_filtered_database_metadata = self.wrapper(get_filtered_database_metadata)
                    self.read_bacseg_images = self.wrapper(read_bacseg_images)

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

        segChannel = self.cellpose_segchannel.currentText()

        self.export_channel.setCurrentText(segChannel)


    def _Progresbar(self, progress, progressbar):
        if progressbar == "picasso":
            self.picasso_progressbar.setValue(progress)
        if progressbar == "import":
            self.import_progressbar.setValue(progress)
        if progressbar == "export":
            self.export_progressbar.setValue(progress)
        if progressbar == "cellpose":
            self.cellpose_progressbar.setValue(progress)
        if progressbar == "database_upload":
            self.upload_progressbar.setValue(progress)
        if progressbar == "database_download":
            self.download_progressbar.setValue(progress)
        if progressbar == "modify":
            self.modify_progressbar.setValue(progress)
        if progressbar == "undrift":
            self.undrift_progressbar.setValue(progress)

        if progress == 100:
            time.sleep(1)
            self.import_progressbar.setValue(0)
            self.export_progressbar.setValue(0)
            self.cellpose_progressbar.setValue(0)
            self.modify_progressbar.setValue(0)
            self.undrift_progressbar.setValue(0)
            self.download_progressbar.setValue(0)
            self.upload_progressbar.setValue(0)

            # self.picasso_progressbar.setValue(0)

    def _importDialog(self, paths=None):
        if self.unfolded == True:
            self.fold_images()

        import_mode = self.import_mode.currentText()
        import_filemode = self.import_filemode.currentText()

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
                self.import_images = self.wrapper(napari_bacseg._utils.import_images)

                worker = Worker(self.import_images, file_paths=paths)
                worker.signals.result.connect(self._process_import)
                worker.signals.progress.connect(partial(self._Progresbar, progressbar="import"))
                self.threadpool.start(worker)

            if import_mode == "NanoImager Data":
                self.read_nim_directory = self.wrapper(napari_bacseg._utils.read_nim_directory)
                self.read_nim_images = self.wrapper(napari_bacseg._utils.read_nim_images)

                measurements, file_paths, channels = self.read_nim_directory(paths)

                worker = Worker(self.read_nim_images, measurements=measurements, channels=channels, )
                worker.signals.result.connect(self._process_import)
                worker.signals.progress.connect(partial(self._Progresbar, progressbar="import"))
                self.threadpool.start(worker)

            if import_mode == "Mask (.tif) Segmentation(s)":
                self.import_masks = self.wrapper(napari_bacseg._utils.import_masks)
                self.import_masks(paths, file_extension=".tif")
                self._autoClassify()

            if import_mode == "Cellpose (.npy) Segmentation(s)":
                self.import_masks = self.wrapper(napari_bacseg._utils.import_masks)
                self.import_masks(paths, file_extension=".npy")
                self._autoClassify()

            if import_mode == "Oufti (.mat) Segmentation(s)":
                self.import_masks = self.wrapper(napari_bacseg._utils.import_masks)
                self.import_masks(paths, file_extension=".mat")
                self._autoClassify()

            if import_mode == "JSON (.txt) Segmentation(s)":
                self.import_masks = self.wrapper(napari_bacseg._utils.import_masks)
                self.import_masks(paths, file_extension=".txt")
                self._autoClassify()

            if import_mode == "ImageJ files(s)":
                self.import_imagej = self.wrapper(napari_bacseg._utils.import_imagej)

                worker = Worker(self.import_imagej, paths=paths)
                worker.signals.result.connect(self._process_import)
                worker.signals.progress.connect(partial(self._Progresbar, progressbar="import"))
                self.threadpool.start(worker)

            if import_mode == "ScanR Data":
                from napari_bacseg._utils import (read_scanr_directory, read_scanr_images, )

                self.read_scanr_images = self.wrapper(read_scanr_images)

                measurements, file_paths, channels = read_scanr_directory(self, paths)

                worker = Worker(self.read_scanr_images, measurements=measurements, channels=channels, )
                worker.signals.result.connect(self._process_import)
                worker.signals.progress.connect(partial(self._Progresbar, progressbar="import"))
                self.threadpool.start(worker)

            if import_mode == "Zeiss (.czi) Data":
                try:
                    from napari_bacseg._utils_zeiss import (get_zeiss_measurements, read_zeiss_image_files, )

                    self.get_zeiss_measurements = self.wrapper(get_zeiss_measurements)
                    self.read_zeiss_image_files = self.wrapper(read_zeiss_image_files)

                    (zeiss_measurements, channel_names, num_measurements,) = get_zeiss_measurements(self, paths)

                    worker = Worker(self.read_zeiss_image_files, zeiss_measurements=zeiss_measurements, channel_names=channel_names, num_measurements=num_measurements, )
                    worker.signals.result.connect(self._process_import)
                    worker.signals.progress.connect(partial(self._Progresbar, progressbar="import"))
                    self.threadpool.start(worker)

                except:
                    print(traceback.format_exc())

    def _export_stack(self, mode, viewer=None):
        def _event(viewer):
            execute_export = True

            if self.export_location.currentIndex() == 1:
                desktop = os.path.expanduser("~/Desktop")
                self.export_directory = QFileDialog.getExistingDirectory(self, "Select Directory", desktop)

                if self.export_directory == "":
                    execute_export = False

            if execute_export == True:
                self.export_stacks = self.wrapper(napari_bacseg._utils.export_stacks)

                worker = Worker(self.export_stacks, mode=mode)
                worker.signals.progress.connect(partial(self._Progresbar, progressbar="export"))
                self.threadpool.start(worker)

        return _event

    def _export(self, mode, viewer=None):
        def _event(viewer):
            # if self.unfolded == True:
            #     self.fold_images()

            execute_export = True

            if self.export_location.currentIndex() == 1:
                desktop = os.path.expanduser("~/Desktop")
                self.export_directory = QFileDialog.getExistingDirectory(self, "Select Directory", desktop)

                if self.export_directory == "":
                    execute_export = False

            if execute_export == True:
                self.export_files = self.wrapper(napari_bacseg._utils.export_files)

                worker = Worker(self.export_files, mode=mode)
                worker.signals.progress.connect(partial(self._Progresbar, progressbar="export"))
                self.threadpool.start(worker)

        return _event

    def _trainCellpose(self):
        if self.unfolded == True:
            self.fold_images()

        from napari_bacseg._utils_cellpose import train_cellpose_model

        self.train_cellpose_model = self.wrapper(train_cellpose_model)

        worker = Worker(self.train_cellpose_model)
        worker.signals.progress.connect(partial(self._Progresbar, progressbar="cellpose_train"))
        self.threadpool.start(worker)

    def _segmentActive(self):
        if self.unfolded == True:
            self.fold_images()

        from napari_bacseg._utils_cellpose import (_process_cellpose, _run_cellpose, )

        self._run_cellpose = self.wrapper(_run_cellpose)
        self._process_cellpose = self.wrapper(_process_cellpose)

        current_fov = self.viewer.dims.current_step[0]
        chanel = self.cellpose_segchannel.currentText()

        images = self.viewer.layers[chanel].data.copy()

        image = [images[current_fov, :, :]]

        worker = Worker(self._run_cellpose, images=image)
        worker.signals.result.connect(self._process_cellpose)
        worker.signals.progress.connect(partial(self._Progresbar, progressbar="cellpose"))
        self.threadpool.start(worker)

    def _segmentAll(self):
        if self.unfolded == True:
            self.fold_images()

        from napari_bacseg._utils_cellpose import (_process_cellpose, _run_cellpose, )

        self._run_cellpose = self.wrapper(_run_cellpose)
        self._process_cellpose = self.wrapper(_process_cellpose)

        channel = self.cellpose_segchannel.currentText()

        images = self.viewer.layers[channel].data.copy()

        images = unstack_images(images)

        worker = Worker(self._run_cellpose, images=images)
        worker.signals.result.connect(self._process_cellpose)
        worker.signals.progress.connect(partial(self._Progresbar, progressbar="cellpose"))
        self.threadpool.start(worker)


    def _updateSliderLabel(self, slider_name, label_name):
        self.slider = self.findChild(QSlider, slider_name)
        self.label = self.findChild(QLabel, label_name)

        slider_value = self.slider.value()

        if (slider_name == "cellpose_flowthresh" or slider_name == "cellpose_maskthresh"):
            self.label.setText(str(slider_value / 100))
        else:
            self.label.setText(str(slider_value))

    def _updateSegmentationCombo(self):
        layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Nucleoid", "Classes", "center_lines", "Localisations", ]]

        self.cellpose_segchannel.clear()
        self.cellpose_segchannel.addItems(layer_names)

        self.cellpose_trainchannel.clear()
        self.cellpose_trainchannel.addItems(layer_names)

        self.cellpose_trainchannel.clear()
        self.cellpose_trainchannel.addItems(layer_names)

        self.alignment_channel.clear()
        self.alignment_channel.addItems(layer_names)

        self.undrift_channel.clear()
        self.undrift_channel.addItems(layer_names)

        self.export_stack_channel.clear()
        self.export_stack_channel.addItems(layer_names)

        self.picasso_image_channel.clear()
        self.picasso_image_channel.addItems(layer_names)

        self.export_channel.clear()
        export_layers = layer_names
        export_layers.extend(["All Channels (Stack)", "All Channels (Horizontal Stack)", "All Channels (Vertical Stack)", "First Three Channels (RGB)", ])
        self.export_channel.addItems(export_layers)

        self.refine_channel.clear()
        refine_layers = ["Mask + " + layer for layer in layer_names]
        self.refine_channel.addItems(["Mask"] + refine_layers)

        self.export_colicoords_mode.clear()
        refine_layers = ["Mask + " + layer for layer in layer_names]
        self.export_colicoords_mode.addItems(["None (OpenCV Stats)", "Mask"] + refine_layers)

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
            if self.scalebar_show.isChecked() and len(layer_names) > 0:
                pixel_resolution = float(self.scalebar_resolution.text())
                scalebar_units = self.scalebar_units.currentText()

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
            if self.autocontrast.isChecked():
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

            if (self.overlay_filename.isChecked() and "image_name" in metadata.keys()):
                viewer_text = f"File Name: {metadata['image_name']}"
            if self.overlay_folder.isChecked() and "folder" in metadata.keys():
                viewer_text = viewer_text + f"\nFolder: {metadata['folder']}"
            if (self.overlay_microscope.isChecked() and "microscope" in metadata.keys()):
                viewer_text = (viewer_text + f"\nMicroscope: {metadata['microscope']}")
            if (self.overlay_datemodified.isChecked() and "date_modified" in metadata.keys()):
                viewer_text = (viewer_text + f"\nDate Modified: {metadata['date_modified']}")
            if (self.overlay_content.isChecked() and "content" in metadata.keys()):
                viewer_text = viewer_text + f"\nContent: {metadata['content']}"
            if self.overlay_strain.isChecked() and "strain" in metadata.keys():
                viewer_text = viewer_text + f"\nStrain: {metadata['strain']}"
            if (self.overlay_phenotype.isChecked() and "phenotype" in metadata.keys()):
                viewer_text = (viewer_text + f"\nPhenotype: {metadata['phenotype']}")
            if (self.overlay_antibiotic.isChecked() and "antibiotic" in metadata.keys()):
                viewer_text = (viewer_text + f"\nAntibiotic: {metadata['antibiotic']}")
            if self.overlay_stain.isChecked() and "stain" in metadata.keys():
                viewer_text = viewer_text + f"\nStain: {metadata['stain']}"
            if (self.overlay_staintarget.isChecked() and "stain_target" in metadata.keys()):
                viewer_text = (viewer_text + f"\nStain Target: {metadata['stain_target']}")
            if (self.overlay_modality.isChecked() and "modality" in metadata.keys()):
                viewer_text = (viewer_text + f"\nModality: {metadata['modality']}")
            if (self.overlay_lightsource.isChecked() and "source" in metadata.keys()):
                viewer_text = (viewer_text + f"\nLight Source: {metadata['source']}")
            if (self.overlay_focus.isChecked() and "image_focus" in metadata.keys()):
                viewer_text = (viewer_text + f"\nImage Focus: {metadata['image_focus']}")
            if (self.overlay_debris.isChecked() and "image_debris" in metadata.keys()):
                viewer_text = (viewer_text + f"\nImage Debris: {metadata['image_debris']}")
            if self.overlay_laplacian.isChecked():
                image_laplacian = np.mean(cv2.Laplacian(image, cv2.CV_64F))
                viewer_text = viewer_text + f"\nLaplacian: {image_laplacian}"
            if self.overlay_range.isChecked():
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

        append_mode = self.import_append_mode.currentIndex()

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

            from napari_bacseg._utils import stack_images

            new_image_stack, new_metadata = stack_images(images, metadata)
            new_mask_stack, new_metadata = stack_images(masks, metadata)
            new_nmask_stack, new_metadata = stack_images(nmasks, metadata)
            new_class_stack, new_metadata = stack_images(classes, metadata)

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
                    from napari_bacseg._utils import append_image_stacks

                    (appended_image_stack, appended_metadata,) = append_image_stacks(current_metadata, new_metadata, current_image_stack, new_image_stack, )

                    (appended_mask_stack, appended_metadata,) = append_image_stacks(current_metadata, new_metadata, current_mask_stack, new_mask_stack, )

                    (appended_nmask_stack, appended_metadata,) = append_image_stacks(current_metadata, new_metadata, current_nmask_stack, new_nmask_stack, )

                    (appended_class_stack, appended_metadata,) = append_image_stacks(current_metadata, new_metadata, current_class_stack, new_class_stack, )

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
        self.import_progressbar.reset()
        self.viewer.reset_view()
        self._autoClassify()
        align_image_channels(self)
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

    def _autoClassify(self, reset=False):
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

                        # appends contour to list if the bounding coordinates are along the edge of the image
                        if (y1 > 0 and y2 < cnt_mask.shape[0] and x1 > 0 and x2 < cnt_mask.shape[1]):
                            label[mask == mask_id] = 1

                        else:
                            label[mask == mask_id] = 6

            label_stack[i, :, :] = label

        self.classLayer.data = label_stack
