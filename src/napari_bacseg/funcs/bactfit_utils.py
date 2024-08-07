import numpy as np
import traceback
import pandas as pd
from napari_bacseg.funcs.threading_utils import Worker
from bactfit.preprocess import data_to_cells, mask_to_cells
from functools import partial
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
from bactfit.cell import CellList, Cell, ModelCell
from napari.utils.notifications import show_info

class _bactfit_utils:

    def run_bactfit_finished(self):

        self.update_ui()
        # self.update_segmentation_combos()

    def run_bactfit_results(self, cell_list):

        if cell_list is None:
            return

        data = cell_list.get_cell_polygons(ndim=3,flipxy=True)

        cell_names = data["names"]
        cell_polygons = data["polygons"]
        cell_radii = data["cell_radii"]
        cell_params = data["poly_params"]
        cell_poles = data["cell_poles"]
        cell_midlines = data["midlines"]

        layer_names = [layer.name for layer in self.viewer.layers]

        if "Cells" in layer_names:
            self.viewer.layers.remove("Cells")

        shapes = []
        shape_types = []
        properties = {"name": [], "cell": []}

        for name, polygon, radius, midline, params, poles in zip(cell_names, cell_polygons,
                cell_radii, cell_midlines, cell_params, cell_poles):

            try:

                fit_params = {"name": name, "radius": radius,
                              "poly_params": params, "cell_poles": poles}

                shapes.append(polygon)
                shape_types.append("polygon")
                properties["name"].append(name)
                properties["cell"].append(fit_params)

                shapes.append(midline)
                shape_types.append("path")
                properties["name"].append(name)
                properties["cell"].append(fit_params)

            except:
                pass

        show_info(f"Initialsing Napari Cells Layer")

        self.cellLayer = self.initialise_cellLayer(shapes=shapes,
            shape_types=shape_types, properties=properties)

        self.store_cell_shapes()


    def run_bactfit(self, segmentations, progress_callback=None):

        try:

            min_radius = float(self.gui.fit_min_radius.value())
            max_radius = float(self.gui.fit_max_radius.value())
            max_error = float(self.gui.fit_max_error.value())

            show_info(f"Building CellList")

            celllist = mask_to_cells(segmentations)

            n_cells = len(celllist.data)

            if n_cells == 0:
                return None

            show_info(f"BactFit Fitting {n_cells} cells")

            celllist.optimise(max_radius=max_radius, min_radius=min_radius,
                max_error=max_error, progress_callback=progress_callback)

            error_list = [cell.fit_error for cell in celllist.data]
            error_list = [error for error in error_list if error is not None]
            print(f"Max error: {max(error_list)}")

        except:
            print(traceback.format_exc())
            return None

        return celllist

    def initialise_bactfit(self):

        if hasattr(self, "segLayer"):

            segmentations = self.segLayer.data.copy()

            segmentations = [seg for seg in segmentations if seg.shape[0] > 2]

            if len(segmentations) == 0:
                return

            self.update_ui(init=True)

            worker = Worker(self.run_bactfit, segmentations)
            worker.signals.progress.connect(partial(self._Progresbar,
                    progressbar="bactfit"))
            worker.signals.result.connect(self.run_bactfit_results)
            worker.signals.finished.connect(self.run_bactfit_finished)
            worker.signals.error.connect(self.update_ui)
            self.threadpool.start(worker)

    def transform_coordinates(self, celllist,
            model, method, progress_callback=None):

        try:

            celllist.transform_cells(model,
                progress_callback=progress_callback)

        except:
            print(traceback.format_exc())
            pass

        return celllist

    def init_transform_coordinates(self, progress_callback=None, method = "angular"):

        try:
            self.update_ui(init=True)

            if hasattr(self, "cellLayer") == False:
                show_info("Coordinate transformation requires fitted cells")
                self.update_ui()
                return

            if hasattr(self, "fitted_locs") == False:
                show_info("Coordinate transformation requires fitted localisations")
                self.update_ui()
                return

            locs = self.fitted_locs.copy()
            locs = pd.DataFrame(locs)

            if len(locs) == 0:
                show_info("Coordinate transformation requires fitted localisations")
                self.update_ui()
                return

            cell_data = self.cellLayer.data.copy()

            if len(cell_data) == 0:
                show_info("Coordinate transformation requires fitted cells")
                self.update_ui()
                return

            cell_length = int(self.gui.model_cell_length.value())
            cell_radius = int(self.gui.model_cell_radius.value())
            pixel_size = float(self.gui.transform_pixel_size.value())

            show_info("Building CellList with localisations")

            cell_list = self.populate_cell_list(locs, pixel_size)

            if len(cell_list) == 0:
                show_info("No cells found")
                self.update_ui()
                return

            celllist = CellList(cell_list)

            model = ModelCell(
                length=cell_length,
                radius=cell_radius,)

            if hasattr(self, "celllist"):
                del self.celllist

            n_cells = len(celllist.data)
            n_locs = len(celllist.get_locs())

            show_info(f"Transforming {n_locs} localisations within {n_cells} cells")

            try:
                worker = Worker(self.transform_coordinates, celllist, model, method)
                worker.signals.progress.connect(partial(self._Progresbar,
                        progressbar="transform_coordinates"))
                worker.signals.result.connect(self.transform_coordinates_results)
                worker.signals.finished.connect(self.transform_coordinates_finished)
                worker.signals.error.connect(self.update_ui)
                self.threadpool.start(worker)
            except:
                self.update_ui()
                print(traceback.format_exc())


        except:
            print(traceback.format_exc())
            pass


    def transform_coordinates_results(self, celllist):

        try:


            self.celllist = celllist

            celllocs = celllist.get_locs(symmetry=False)
            celllocs = pd.DataFrame(celllocs)

            if len(celllocs) == 0:
                return

            self.transformed_locs = celllocs

            n_transformed = len(celllocs)

            show_info(f"Transformed {n_transformed} localisations")

        except:
            print(traceback.format_exc())
            pass

    def transform_coordinates_finished(self):

        self.update_render_length_range()
        self.heatmap_canvas.clear()
        self.plot_heatmap()
        self.update_ui()

    def populate_cell_list(self, locs=None, pixel_size=None):

        try:

            if isinstance(locs, pd.DataFrame):
                locs = locs.to_records(index=False)

            name_list = self.cellLayer.properties["name"].copy()
            name_list = list(set(name_list))
            cell_list = []

            for name in name_list:

                cell = self.get_cell(name, bactfit=True,
                    pixel_size=pixel_size)

                if cell is None:
                    continue

                frame_index = int(cell.frame_index)

                if frame_index is None:
                    continue

                if locs is not None:

                    frame_locs = locs[locs["frame"] == frame_index].copy()
                    frame_locs = frame_locs.to_records(index=False)

                    if len(frame_locs) == 0:
                        continue

                    cell.remove_locs_outside_cell(frame_locs)
                    cell_locs = cell.locs
                else:
                    cell_locs = None

                if cell_locs is None:
                    continue

                cell_list.append(cell)

            if len(cell_list) == 0:
                show_info("No cells found")
                return []

        except:
            print(traceback.format_exc())
            pass

        return cell_list