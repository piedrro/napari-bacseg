import cellpose.io
import pandas as pd
import traceback
import numpy as np
from functools import partial
from bactfit.preprocess import data_to_cells
from bactfit.cell import CellList, ModelCell
from bactfit.postprocess import remove_locs_outside_cell
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pyqtgraph as pg
from io import BytesIO
from picasso.render import render
from PyQt5.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QFormLayout, QVBoxLayout, QWidget, QMainWindow, QSpinBox
from PyQt5.QtWidgets import QFileDialog
import os
import cv2
from shapely.geometry import Polygon
from matplotlib.colors import ListedColormap
from napari.utils.notifications import show_info
from napari_bacseg.funcs.threading_utils import Worker
from napari.utils.notifications import show_info
import h5py
import yaml

class CustomPyQTGraphWidget(pg.GraphicsLayoutWidget):

    def __init__(self, parent):
        super().__init__()

        self.parent = parent
        self.frame_position_memory = {}
        self.frame_position = None

class _heatmap_utils:

    def update_render_length_range(self):

        try:

            if hasattr(self, "celllist") == False:
                return

            if self.celllist is None:
                return

            cell_lengths = self.celllist.get_cell_lengths()

            if len(cell_lengths) > 0:
                min_length = min(cell_lengths)
                max_length = max(cell_lengths)
                self.gui.heatmap_min_length.setRange(min_length, max_length)
                self.gui.heatmap_max_length.setRange(min_length, max_length)
                self.gui.heatmap_min_length.setValue(min_length)
                self.gui.heatmap_max_length.setValue(max_length)

                self.gui.heatmap_min_length.setSingleStep(0.01)
                self.gui.heatmap_max_length.setSingleStep(0.01)

        except:
            print(traceback.format_exc())
            pass



    def cell_heatmap_compute_finished(self):

        try:

            self.update_render_length_range()
            self.heatmap_canvas.clear()
            self.plot_heatmap()
            self.update_ui()
            show_info("Cell heatmap computed.")

        except:
            print(traceback.format_exc())
            pass

    def plot_heatmap(self):

        try:

            if hasattr(self, "celllist") == False:
                show_info("Heatmap requires transformed localisations")
                return
            if self.celllist is None:
                show_info("Heatmap requires transformed localisations")
                return

            self.update_ui(init=True)

            heatmap_mode = self.gui.heatmap_mode.currentText()
            colourmap_name = self.gui.heatmap_colourmap.currentText()
            draw_outline = self.gui.render_draw_outline.isChecked()
            min_length = self.gui.heatmap_min_length.value()
            max_length = self.gui.heatmap_max_length.value()
            symmetry = self.gui.render_symmetry.isChecked()
            bins = self.heatmap_binning.value()
            blur_method = self.heatmap_blur_method.currentText()
            min_blur_width = self.heatmap_min_blur_width.value()
            oversampling = self.heatmap_oversampling.value()

            self.heatmap_canvas.clear()

            celllist = self.celllist

            celllist = celllist.filter_by_length(min_length, max_length)

            if len(celllist.data) == 0:
                self.update_ui()
                return

            polygon = celllist.data[0].cell_polygon
            polygon_coords = np.array(polygon.exterior.coords)

            celllocs = celllist.get_locs(symmetry=symmetry)

            if celllocs is None:
                self.update_ui()
                return

            if len(celllocs) == 0:
                self.update_ui()
                return

            n_cells = len(celllist.data)
            n_locs = len(celllocs)
            if symmetry:
                n_locs = int(n_locs/4)

            self.heatmap_locs = celllocs
            self.heatmap_polygon = polygon_coords

            show_info(f"Generating Cell {heatmap_mode.lower()} with {n_locs} localisations from {n_cells} cells")

            if heatmap_mode == "Heatmap":

                heatmap = celllist.plot_heatmap(symmetry=symmetry, bins=bins,
                    cmap=colourmap_name, draw_outline=draw_outline,
                    show=False, save=False, path=None, dpi=500)

                self.heatmap_image = heatmap
                self.show_heatmap(heatmap)

            elif heatmap_mode == "Render":

                render = celllist.plot_render(
                    symmetry=symmetry, oversampling=oversampling,
                    blur_method=blur_method, min_blur_width=min_blur_width,
                    cmap=colourmap_name, draw_outline=draw_outline,
                    show=False, save=False, path=None, dpi=500)

                self.heatmap_image = render
                self.show_heatmap(render)

            else:
                pass

            self.update_ui()

        except:
            self.update_ui()
            print(traceback.format_exc())
            pass

    def show_heatmap(self, image):

        try:

            image = np.rot90(image, k=3)
            image = np.fliplr(image)

            self.heatmap_canvas.clear()
            self.heatmap_canvas.setImage(image)

            self.heatmap_canvas.ui.histogram.hide()
            self.heatmap_canvas.ui.roiBtn.hide()
            self.heatmap_canvas.ui.menuBtn.hide()

        except:
            print(traceback.format_exc())
            pass


    def plot_cell_heatmap(self, celllocs, polygon_coords,
            colourmap_name="inferno", draw_outline=True):

        try:

            bins = self.heatmap_binning.value()

            heatmap, xedges, yedges = np.histogram2d(celllocs["x"], celllocs["y"], bins=bins)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

            cmap = self.get_custom_cmap(colour=colourmap_name)

            plt.rcParams["axes.grid"] = False
            fig, ax = plt.subplots()
            im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap=cmap)
            if draw_outline:
                ax.plot(*polygon_coords.T, color='white', linewidth=1)
            ax.axis('off')

            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            plt.colorbar(im, cax=cax)
            cax.set_facecolor('black')

            buf = BytesIO()
            plt.savefig(buf, format='png',
                bbox_inches='tight',pad_inches=0.1,
                facecolor='black', dpi=500)
            buf.seek(0)
            heatmap = plt.imread(buf)

            # Close the figure
            plt.close(fig)

            self.heatmap_image = heatmap

            heatmap = np.rot90(heatmap, k=3)
            heatmap = np.fliplr(heatmap)

            self.heatmap_canvas.clear()
            self.heatmap_canvas.setImage(heatmap)

            self.heatmap_canvas.ui.histogram.hide()
            self.heatmap_canvas.ui.roiBtn.hide()
            self.heatmap_canvas.ui.menuBtn.hide()

        except:
            print(traceback.format_exc())
            pass


    def get_custom_cmap(self, colour = "jet"):

        cmap = plt.get_cmap(colour.lower())
        new_cmap = cmap(np.arange(cmap.N))

        new_cmap[0] = [0, 0, 0, 1]

        new_cmap = ListedColormap(new_cmap)

        return new_cmap


    def plot_cell_render(self, celllocs, polygon_coords,
            colourmap_name="inferno", draw_outline=True):

        try:

            blur_method = self.heatmap_blur_method.currentText()
            min_blur_width = self.heatmap_min_blur_width.value()
            oversampling = self.heatmap_oversampling.value()

            celllocs = pd.DataFrame(celllocs)

            picasso_columns = ["frame",
                               "y", "x",
                               "photons", "sx", "sy", "bg",
                               "lpx", "lpy",
                               "ellipticity", "net_gradient",
                               "group", "iterations", ]

            column_filter = [col for col in picasso_columns if col in celllocs.columns]
            celllocs = celllocs[column_filter]
            celllocs = celllocs.to_records(index=False)

            xmin, xmax = polygon_coords[:, 0].min(), polygon_coords[:, 0].max()
            ymin, ymax = polygon_coords[:, 1].min(), polygon_coords[:, 1].max()

            h,w = int(ymax-ymin)+3, int(xmax-xmin)+3

            viewport = [(float(0), float(0)), (float(h), float(w))]

            if blur_method == "One-Pixel-Blur":
                blur_method = "smooth"
            elif blur_method == "Global Localisation Precision":
                blur_method = "convolve"
            elif blur_method == "Individual Localisation Precision, iso":
                blur_method = "gaussian_iso"
            elif blur_method == "Individual Localisation Precision":
                blur_method = "gaussian"
            else:
                blur_method = None

            n_rendered_locs, image = render(celllocs,
                viewport=viewport,
                blur_method=blur_method,
                min_blur_width=min_blur_width,
                oversampling=oversampling, ang=0, )

            #stretch polygon to image size
            polygon_coords = np.array(polygon_coords)
            polygon_coords = polygon_coords * oversampling

            cmap = self.get_custom_cmap(colour=colourmap_name)

            plt.rcParams["axes.grid"] = False
            fig, ax = plt.subplots()
            ax.imshow(image, cmap=cmap)
            if draw_outline:
                ax.plot(*polygon_coords.T, color='white')
            ax.axis('off')

            buf = BytesIO()
            plt.savefig(buf, format='png',
                bbox_inches='tight', pad_inches=0.1,
                facecolor='black', dpi=500)
            buf.seek(0)
            image = plt.imread(buf)
            plt.close(fig)

            self.heatmap_image = image

            #rotate and flip
            image = np.rot90(image, k=3)
            image = np.fliplr(image)

            self.heatmap_canvas.clear()
            self.heatmap_canvas.setImage(image)

            self.heatmap_canvas.ui.histogram.hide()
            self.heatmap_canvas.ui.roiBtn.hide()
            self.heatmap_canvas.ui.menuBtn.hide()

        except:
            print(traceback.format_exc())
            pass



    def export_cell_heatmap(self):

        try:

            if hasattr(self, "heatmap_image") == False:
                return

            if self.heatmap_image is None:
                return

            image = self.heatmap_image.copy()
            mode = self.gui.heatmap_mode.currentText()


            image_channel = self.gui.picasso_image_channel.currentText()
            path = self.viewer.layers[image_channel].metadata[0]["image_path"]

            directory = os.path.dirname(path)
            file_name = os.path.basename(path)
            base, ext = os.path.splitext(file_name)
            path = os.path.join(directory, base + f"_cell_{mode.lower()}" + ".png")
            path = QFileDialog.getSaveFileName(self, "Save Image", path, "PNG (*.png,*.tif)")[0]

            if path == "":
                return

            plt.imsave(path, image, cmap='inferno')
            show_info(f"Exported cell {mode.lower()} to {path}")

        except:
            print(traceback.format_exc())



    def get_heatmap_locs(self):

        try:

            heatmap_datset = self.gui.heatmap_dataset.currentText()
            heatmap_channel = self.gui.heatmap_channel.currentText()
            min_length = self.gui.heatmap_min_length.value()
            max_length = self.gui.heatmap_max_length.value()
            min_msd = self.gui.heatmap_min_msd.value()
            max_msd = self.gui.heatmap_max_msd.value()

            if hasattr(self, "celllist") == False:
                return

            if self.celllist is None:
                return

            celllist = self.celllist
            celllist = celllist.filter_by_length(min_length, max_length)

            if len(celllist.data) == 0:
                return

            celllocs = celllist.get_locs()
            celllocs = pd.DataFrame(celllocs)

            if "dataset" in celllocs.columns:
                if heatmap_datset != "All Datasets":
                    celllocs = celllocs[celllocs["dataset"] == heatmap_datset]
            if "channel" in celllocs.columns:
                if heatmap_channel != "All Channels":
                    celllocs = celllocs[celllocs["channel"] == heatmap_channel]

            celllocs = celllocs.to_records(index=False)

            if len(celllocs) == 0:
                return

            if "msd" in celllocs.dtype.names:
                celllocs = celllocs[celllocs["msd"] > min_msd]
                celllocs = celllocs[celllocs["msd"] < max_msd]

            return celllocs

        except:
            print(traceback.format_exc())
            return None


    def export_heatmap_locs(self):

        try:

            self.update_ui(init=True)

            if hasattr(self, "heatmap_locs") == False:
                show_info("No heatmap localisations to export")
                return

            locs = self.heatmap_locs
            polygon_coords = self.heatmap_polygon

            if len(locs) == 0:
                show_info("No heatmap localisations to export")
                return

            image_channel = self.gui.picasso_image_channel.currentText()
            path = self.viewer.layers[image_channel].metadata[0]["image_path"]

            if type(path) == list:
                path = path[0]

            directory = os.path.dirname(path)
            file_name = os.path.basename(path)
            base, ext = os.path.splitext(file_name)
            path = os.path.join(directory, base + "_heatmap_locs.csv")
            options = QFileDialog.Options()
            file_filter = "CSV (*.csv);;Picasso HDF5 (*.hdf5);; POS.OUT (*.pos.out)"
            path, filter = QFileDialog.getSaveFileName(self, "Save Image", path, file_filter, options=options)

            if path == "":
                return None

            if filter == "CSV (*.csv)":

                locs = pd.DataFrame(locs)
                locs.to_csv(path, index=False)

                show_info(f"Exported heatmap CSV localisations")

            elif filter == "Picasso HDF5 (*.hdf5)":

                xmin, xmax = polygon_coords[:, 0].min(), polygon_coords[:, 0].max()
                ymin, ymax = polygon_coords[:, 1].min(), polygon_coords[:, 1].max()

                h,w = int(ymax-ymin)+3, int(xmax-xmin)+3

                image_shape = (0,h,w)

                locs = pd.DataFrame(locs)

                picasso_columns = ["frame", "y", "x", "photons",
                                   "sx", "sy", "bg", "lpx", "lpy",
                                   "ellipticity", "net_gradient", "group", "iterations", ]

                for column in locs.columns:
                    if column not in picasso_columns:
                        locs.drop(column, axis=1, inplace=True)

                locs = locs.to_records(index=False)

                box_size = int(self.gui.picasso_box_size.currentText())
                picasso_info = self.get_picasso_info(path, image_shape, box_size)

                info_path = path.replace(".hdf5", ".yaml")

                with h5py.File(path, "w") as hdf_file:
                    hdf_file.create_dataset("locs", data=locs)

                # Save to temporary YAML file
                with open(info_path, "w") as file:
                    yaml.dump_all(picasso_info, file, default_flow_style=False)

                show_info(f"Exported heatmap HDF5 localisations")

            elif filter == "POS.OUT (*.pos.out)":

                localisation_data = pd.DataFrame(locs)

                pos_locs = localisation_data[["frame", "x", "y", "photons", "bg", "sx", "sy", ]].copy()

                pos_locs.dropna(axis=0, inplace=True)

                pos_locs.columns = ["FRAME", "XCENTER", "YCENTER", "BRIGHTNESS", "BG", "S_X", "S_Y", ]

                pos_locs.loc[:, "I0"] = 0
                pos_locs.loc[:, "THETA"] = 0
                pos_locs.loc[:, "ECC"] = pos_locs["S_X"] / pos_locs["S_Y"]
                pos_locs.loc[:, "FRAME"] = pos_locs["FRAME"] + 1

                pos_locs = pos_locs[["FRAME", "XCENTER", "YCENTER", "BRIGHTNESS", "BG", "I0", "S_X", "S_Y", "THETA", "ECC", ]]

                pos_locs.to_csv(path, sep="\t", index=False)

                show_info(f"Exported heatmap POS.OUT localisations")

            else:
                print("File format not supported")

        except:
            print(traceback.format_exc())
            self.update_ui()
            pass

        self.update_ui()