import numpy as np
import traceback
from napari_bacseg.funcs.threading_utils import Worker
from napari_bacseg.bactfit.preprocess import data_to_cells, mask_to_cells
from functools import partial
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
from napari_bacseg.bactfit.fit import BactFit
from napari_bacseg.bactfit.cell import CellList
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
        cell_widths = data["widths"]
        cell_params = data["poly_params"]
        cell_poles = data["cell_poles"]
        cell_midlines = data["midlines"]

        layer_names = [layer.name for layer in self.viewer.layers]

        if "Cells" in layer_names:
            self.viewer.layers.remove("Cells")

        shapes = []
        shape_types = []
        properties = {"name": [], "cell": []}

        for name, polygon, width, midline, params, poles in zip(cell_names, cell_polygons,
                cell_widths, cell_midlines, cell_params, cell_poles):

            try:

                fit_params = {"name": name, "width": width,
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

        self.cellLayer = self.initialise_cellLayer(shapes=shapes,
            shape_types=shape_types, properties=properties)

        self.store_cell_shapes()


    def run_bactfit(self, segmentations, progress_callback=None):

        try:

            min_radius = float(self.gui.fit_min_radius.value())
            max_radius = float(self.gui.fit_max_radius.value())
            max_error = float(self.gui.fit_max_error.value())

            celllist = mask_to_cells(segmentations)

            n_cells = len(celllist.data)

            if n_cells == 0:
                return None

            show_info(f"BactFit Fitting {n_cells} cells")

            bf = BactFit(celllist=celllist,
                max_radius=max_radius, min_radius=min_radius, max_error=max_error,
                progress_callback=progress_callback, parallel=True,)

            celllist = bf.fit()

            self.celllist = celllist

            max_error = [cell.fit_error for cell in self.celllist.data]
            print(f"Max error: {max(max_error)}")

        except:
            print(traceback.format_exc())
            return None

        return self.celllist

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






