import os
import pathlib

# from napari_bacseg._utils_json import import_coco_json, export_coco_json
import traceback

import numpy as np
import pandas as pd
from glob2 import glob
from napari.utils.notifications import show_info
from qtpy.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QLabel, QProgressBar, QPushButton, QTabWidget, )


def _create_bacseg_database(self, viewer=None, database_name="BacSeg", path=None):
    if type(path) != str:
        desktop = os.path.expanduser("~/Desktop")
        path = QFileDialog.getExistingDirectory(self, "Select Directory", desktop)

    if os.path.isdir(path):
        database_directory = str(pathlib.PurePath(path, f"{database_name}_Database"))

        num_user_keys = self.user_metadata_keys

        if os.path.exists(database_directory) == True:
            if self.widget_notifications:
                show_info(f"{database_name} Database already exists at location {database_directory}")

        else:
            if self.widget_notifications:
                show_info(f"Creating {database_name} Database at location {database_directory}")

            if os.path.isdir(database_directory) is False:
                os.mkdir(database_directory)

            folders = ["Images", "Metadata", "Models"]

            folder_paths = [str(pathlib.PurePath(database_directory, folder)) for folder in folders]

            for folder_path in folder_paths:
                if os.path.exists(folder_path) == False:
                    os.mkdir(folder_path)

            database_metadata_list = ["abxconcentration", "antibiotic", "content", "microscope", "modality", "mount", "protocol", "source", "stain", "treatment_time", "user_initial", ]

            user_metadata_list = ["example_user"]

            for meta_item in database_metadata_list:
                txt_meta = f"# {database_name} Database Metadata: {meta_item} (Add new entries below):"

                txt_meta_path = pathlib.PurePath(database_directory, "Metadata", f"{database_name} Database Metadata [{meta_item}].txt", )

                with open(txt_meta_path, "w") as f:
                    f.write(txt_meta)

            for user in user_metadata_list:
                txt_meta = f"# {database_name} User Metadata: {user}\n"
                txt_meta += "# Replace 'example_user' with your intial\n"

                txt_meta_path = pathlib.PurePath(database_directory, "Metadata", f"{database_name} User Metadata [{user}].txt", )

                for i in range(1, num_user_keys + 1):
                    txt_meta += f"\n# User Meta [{i}] (add new entries below):"
                    txt_meta += "\nexample_item1"
                    txt_meta += "\nexample_item2"
                    txt_meta += "\nexample_item3"
                    txt_meta += "\n"

                with open(txt_meta_path, "w") as f:
                    f.write(txt_meta)

    return database_directory


def _load_bacseg_database(self, path=""):
    if os.path.isdir(path) == False:
        desktop = os.path.expanduser("~/Desktop")
        path = QFileDialog.getExistingDirectory(self, "Select Directory", desktop)

    if os.path.isdir(path):
        if "AKSEG" in path or "BacSeg" in path:
            target_database_folders = ["Images", "Metadata", "Models"]
            active_database_folders = [os.path.basename(path) for path in glob(path + "/*", recursive=True)]

            if set(target_database_folders).issubset(active_database_folders):
                self.database_path = os.path.abspath(path)
                from napari_bacseg._utils_database import (populate_upload_combos, )

                populate_upload_combos(self)
                self._populateUSERMETA

                self.display_database_path.setText(path)
                self._show_database_controls(True)


def _show_database_controls(self, visible=True):
    all_database_controls = self.upload_tab.findChildren((QCheckBox, QComboBox, QLabel, QPushButton, QProgressBar, QTabWidget))

    load_database_controls = ["create_database", "load_database", "display_database_path", "display_database_label", "database_io_title", ]

    [item.setVisible(visible) for item in all_database_controls if item.objectName() not in load_database_controls]


def generate_txt_metadata(self, database_directory):
    database_name = (pathlib.Path(database_directory).parts[-1].replace("_Database", ""))

    path = pathlib.PurePath(database_directory, "Metadata", f"{database_name} Metadata.xlsx")

    if os.path.exists:
        akmeta = pd.read_excel(path, usecols="B:M", header=2)

        akmeta = dict(user_initial=akmeta["User Initial"].dropna().astype(str).tolist(), content=akmeta["Image Content"].dropna().astype(str).tolist(), microscope=akmeta[
            "Microscope"].dropna().astype(str).tolist(), modality=akmeta["Modality"].dropna().astype(str).tolist(), source=akmeta["Light Source"].dropna().astype(str).tolist(), antibiotic=akmeta[
            "Antibiotic"].dropna().astype(str).tolist(), abxconcentration=akmeta["Antibiotic Concentration"].dropna().astype(str).tolist(), treatment_time=akmeta[
            "Treatment Time (mins)"].dropna().astype(str).tolist(), stain=akmeta["Stains"].dropna().astype(str).tolist(), stain_target=akmeta["Stain Target"].dropna().astype(str).tolist(), mount=
        akmeta["Mounting Method"].dropna().astype(str).tolist(), protocol=akmeta["Protocol"].dropna().astype(str).tolist(), )

        # generate file metadata

        for key, value in akmeta.items():
            txt_meta = f"# {database_name} Database Metadata: {key} (Add new entries below):"

            txt_meta_path = pathlib.PurePath(database_directory, "Metadata", f"{database_name} Database Metadata [{key}].txt", )

            for item in value:
                txt_meta += f"\n{item.lstrip().rstrip()}"

            with open(txt_meta_path, "w") as f:
                f.write(txt_meta)

        # generate user metadata

        user_metadata = pd.read_excel(path, sheet_name="User Metadata", header=2)
        users = (user_metadata[user_metadata["User Initial"] != "example"]["User Initial"].unique().tolist())

        for user in users:
            txt_meta = f"# {database_name} User Metadata: {user}\n"

            txt_meta_path = pathlib.PurePath(database_directory, "Metadata", f"{database_name} User Metadata [{user}].txt", )

            num_keys = self.user_metadata_keys
            for i in range(1, num_keys + 1):

                txt_meta += f"\n# User Meta [{i}] (add new entries below):"
                txt_meta_key = f"User Meta #{i}"

                if txt_meta_key in user_metadata.columns:
                    item_list = (user_metadata[(user_metadata["User Initial"] == user)][txt_meta_key].dropna().tolist())
                else:
                    item_list = []

                if len(item_list) == 0:
                    txt_meta += "\n"
                else:
                    for item in item_list:
                        if item not in ["", " ", None]:
                            txt_meta += f"\n{item.lstrip().rstrip()}"
                txt_meta += "\n"

            with open(txt_meta_path, "w") as f:
                f.write(txt_meta)


def read_txt_metadata(self, database_directory):

    database_name = (pathlib.Path(database_directory).parts[-1].replace("_Database", ""))

    metadata_directory = str(pathlib.PurePath(database_directory, "Metadata"))

    metadata_files = glob(metadata_directory + r"\*.txt")

    if len(metadata_files) == 0:
        generate_txt_metadata(self, database_directory)
        metadata_files = glob(metadata_directory + r"\*.txt")

    image_metadata_files = [path for path in metadata_files if f"{database_name} Database Metadata" in path]
    user_metadata_fies = [path for path in metadata_files if f"{database_name} User Metadata" in path]

    image_metadata = {}

    for file in image_metadata_files:
        key = strip_brackets(file)

        with open(file) as f:
            lines = f.readlines()

            lines = [line.replace("\n", "") for line in lines if line[0] != "#"]

        image_metadata[key] = lines

    user_metadata = {}

    for file in user_metadata_fies:
        user = strip_brackets(file)

        with open(file) as f:
            lines = f.readlines()

        metakey = None

        user_dict = {"User Initial": user}

        for i, line in enumerate(lines):
            line = line.lstrip().rstrip().replace("\n", "")

            if "User Meta" in line and i != 0:
                metakey = f"meta{strip_brackets(line)}"

                if metakey not in user_dict.keys():
                    user_dict[metakey] = []

            else:
                if metakey is not None and line.strip() not in ["", ",", " ", None, ]:
                    user_dict[metakey].append(line)

        loaded_keys = list(user_dict.keys())
        expected_keys = [f"meta{i}" for i in range(1, self.user_metadata_keys + 1)]

        for key in expected_keys:
            if key not in loaded_keys:
                user_dict[key] = []

        user_metadata[user] = user_dict

    return image_metadata, user_metadata


def strip_brackets(string):
    value = string[string.find("[") + 1: string.find("]")]

    return value


def populate_upload_combos(self):
    try:
        akmeta, _ = read_txt_metadata(self, self.database_path)

        self.upload_initial.clear()
        self.upload_initial.addItems(["Required for upload"] + akmeta["user_initial"])
        self.upload_content.clear()
        self.upload_content.addItems(["Required for upload"] + akmeta["content"])
        self.upload_microscope.clear()
        self.upload_microscope.addItems(["Required for upload"] + akmeta["microscope"])
        self.upload_antibiotic.clear()
        self.upload_antibiotic.addItems([""] + akmeta["antibiotic"])
        self.upload_abxconcentration.clear()
        self.upload_abxconcentration.addItems([""] + akmeta["abxconcentration"])
        self.upload_treatmenttime.clear()
        self.upload_treatmenttime.addItems([""] + akmeta["treatment_time"])
        self.upload_mount.clear()
        self.upload_mount.addItems([""] + akmeta["mount"])
        self.upload_protocol.clear()
        self.upload_protocol.addItems([""] + akmeta["protocol"])

    except:
        print(traceback.format_exc())


def update_database_metadata(self, control=None):

    new_user = False
    new_user_initial = ""

    database_directory = self.database_path

    if os.path.exists(database_directory):
        user_intial_index = self.upload_initial.currentIndex()

        database_name = (pathlib.Path(database_directory).parts[-1].replace("_Database", ""))

        dbmeta, usermeta = read_txt_metadata(self, self.database_path)

        control_dict = {"abxconcentration": "upload_abxconcentration", "antibiotic": "upload_antibiotic", "content": "upload_content",
                        "microscope": "upload_microscope", "modality": "label_modality", "mount": "upload_mount",
                        "protocol": "upload_protocol", "source": "label_light_source", "stain": "label_stain", "stain_target":
                        "label_stain_target", "treatment_time": "upload_treatmenttime", "user_initial": "upload_initial"}

        num_keys = self.user_metadata_keys

        for i in range(1, num_keys + 1):
            control_dict[f"meta{i}"] = f"upload_usermeta{i}"

        upload_control_dict = {}

        for meta_key, meta_values in dbmeta.items():
            if meta_key in control_dict.keys():
                control_name = control_dict[meta_key]

                try:
                    combo_box = getattr(self, control_name)

                    upload_control_dict[control_name] = combo_box.currentText()

                    setattr(combo_box, "allItems", lambda: [combo_box.itemText(i) for i in range(combo_box.count())], )
                    combo_box_items = combo_box.allItems()

                    combo_box_items = [str(item).lstrip().rstrip() for item in combo_box_items]
                    combo_box_items.append(str(combo_box.currentText()))
                    combo_box_items = [item for item in combo_box_items if item not in ["", " ", "Required for upload"]]
                    meta_values = [item for item in meta_values if item not in ["", " ", "Required for upload"]]

                    combo_box_items = np.unique(combo_box_items).tolist()

                    if (len(combo_box_items) > 0 and combo_box_items != meta_values):
                        dbmeta[meta_key] = combo_box_items

                        txt_meta_path = pathlib.PurePath(database_directory, "Metadata", f"{database_name} Database Metadata [{meta_key}].txt", )

                        if os.path.exists(txt_meta_path):
                            if meta_key == "user_initial":
                                new_user = True
                                new_user_initial = str(combo_box.currentText())

                            txt_meta = f"# {database_name} Database Metadata: {meta_key} (Add new entries below):"

                            txt_meta_path = pathlib.PurePath(database_directory, "Metadata", f"{database_name} Database Metadata [{meta_key}].txt", )

                            for item in combo_box_items:
                                txt_meta += f"\n{item.lstrip().rstrip()}"

                            with open(txt_meta_path, "w") as f:
                                f.write(txt_meta)

                            show_info(f"Updated {meta_key.upper()} metadata items in {database_name} database")

                        else:
                            print(f"Metadata file not found: {txt_meta_path}")

                except:
                    print(traceback.format_exc())

        user_initial = self.upload_initial.currentText()

        if user_initial in usermeta.keys():
            txt_meta = f"# {database_name} User Metadata: {user_initial}\n"

            txt_meta_path = pathlib.PurePath(database_directory, "Metadata", f"{database_name} User Metadata [{user_initial}].txt", )

            for meta_key, meta_values in usermeta[user_initial].items():
                try:
                    if meta_key != "User Initial":
                        control_name = control_dict[meta_key]

                        combo_box = getattr(self, control_name)
                        upload_control_dict[control_name] = combo_box.currentText()

                        setattr(combo_box, "allItems", lambda: [combo_box.itemText(i) for i in range(combo_box.count())], )
                        combo_box_items = combo_box.allItems()

                        combo_box_items = [str(item).lstrip().rstrip() for item in combo_box_items]
                        combo_box_items.append(str(combo_box.currentText()))
                        combo_box_items = [item for item in combo_box_items if item not in ["", " ", "Required for upload",
                                                                                    'example_item1', 'example_item2', 'example_item3']]
                        meta_values = [item for item in meta_values if item not in ["", " ", "Required for upload",
                                                                                    'example_item1', 'example_item2', 'example_item3']]

                        combo_box_items = np.unique(combo_box_items).tolist()
                        meta_values = np.unique(meta_values).tolist()

                        txt_meta += f"\n# User Meta [{meta_key.replace('meta', '')}] (add new entries below):"

                        if len(combo_box_items) > 0:
                            for item in combo_box_items:
                                txt_meta += f"\n{item}"
                        else:
                            txt_meta += f"\n{item}"

                        txt_meta += "\n"

                        with open(txt_meta_path, "w") as f:
                            f.write(txt_meta)

                        if (len(combo_box_items) > 0 and combo_box_items != meta_values):
                            show_info(f"Updated {meta_key.upper()} metadata items in {database_name} database")
                except:
                    pass

        if new_user:
            txt_meta = f"# {database_name} User Metadata: {new_user_initial}\n"
            txt_meta += "# Replace 'example_user' with your intial\n"

            txt_meta_path = pathlib.PurePath(database_directory, "Metadata", f"{database_name} User Metadata [{new_user_initial}].txt", )

            for i in range(1, num_keys + 1):
                txt_meta += f"\n# User Meta [{i}] (add new entries below):"
                txt_meta += "\nexample_item1"
                txt_meta += "\nexample_item2"
                txt_meta += "\nexample_item3"
                txt_meta += "\n"

            with open(txt_meta_path, "w") as f:
                f.write(txt_meta)

            show_info(f"Added new user metadata files for {new_user_initial} in {database_name} database")

        self.populate_upload_combos()
        self._populateUSERMETA
        self.upload_initial.setCurrentIndex(user_intial_index)

        for control_name, control_text in upload_control_dict.items():
            combo_box = getattr(self, control_name)
            combo_box.setCurrentText(control_text)


def read_file_metadata(self):

    # control_dict = {"abxconcentration": "upload_abxconcentration", "antibiotic": "upload_antibiotic",
    #                 "content": "upload_content", "microscope": "upload_microscope",
    #                 "modality": "label_modality", "mount": "upload_mount",
    #                 "protocol": "upload_protocol", "source": "label_light_source",
    #                 "stain": "label_stain", "stain_target": "label_stain_target",
    #                 "treatment_time": "upload_treatmenttime"}

    control_dict = {}

    user_file_meta = {}

    user_key_list = np.arange(1, self.user_metadata_keys + 1).tolist()
    user_key_list.reverse()

    for key in user_key_list:
        user_key = f"meta{key}"
        control_dict[user_key] = f"user_meta{key}"

    database_path = self.database_path
    user_initial = self.upload_initial.currentText()

    if database_path != "" and user_initial != "":

        user_metadata_path = os.path.join(database_path, "Images", user_initial, f"{user_initial}_file_metadata.txt", )

        if os.path.exists(user_metadata_path):

            user_metadata = pd.read_csv(user_metadata_path, sep=",", low_memory=False)

            for key, dfkey in control_dict.items():

                values = user_metadata[dfkey].unique().tolist()

                user_file_meta[key] = values

    return user_file_meta





def _populateUSERMETA(self):

    try:
        _, usermeta = read_txt_metadata(self, self.database_path)
        file_usermeta = read_file_metadata(self)

        user_initial = self.upload_initial.currentText()

        num_keys = self.user_metadata_keys

        for key in range(1, num_keys + 1 ):

            control_name = f"upload_usermeta{key}"
            combo_box = getattr(self, control_name)
            combo_box.clear()

            if user_initial in usermeta.keys():
                user_meta_values = usermeta[user_initial][f"meta{key}"]
                file_meta_values = file_usermeta[f"meta{key}"]

                meta_values = np.unique(user_meta_values + file_meta_values).tolist()

                meta_values = [value for value in meta_values if value not in ["", " ", "nan", "Required for upload",
                                                                               'example_item1', 'example_item2', 'example_item3',
                                                                               None, np.nan]]
                combo_box.addItems([""] + meta_values)
            else:
                combo_box.setCurrentText("")

    except:
        # print(traceback.format_exc())
        pass
