import tifffile 
import numpy as np
import cv2
import os
import traceback
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, medial_axis, binary_opening, disk
from scipy.interpolate import CubicSpline, BSpline
from skimage.draw import polygon
from shapely.geometry import LineString, Point, LinearRing, Polygon
from shapely.affinity import rotate
import matplotlib.path as mpltPath
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import copy
from scipy.interpolate import interp1d
import math
from scipy.interpolate import splrep, splev
import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
import shapely
from scipy.spatial.distance import directed_hausdorff
import shapely
from scipy.spatial import distance
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Event
from functools import partial
import warnings

from napari_bacseg.bactfit.utils import (resize_line, moving_average, rotate_polygon,
    rotate_linestring, fit_poly)

class BactFit(object):

    def __init__(self,
            cell = None,
            celllist = None,
            refine_fit = True,
            min_radius = -1,
            max_radius = -1,
            fit_mode = "directed_hausdorff",
            progress_callback=None,
            parallel=True,
            silence_tqdm=False,
            max_workers=None,
            max_error = -1,
            **kwargs):

        self.cell = cell
        self.celllist = celllist
        self.refine_fit = refine_fit
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.fit_mode = fit_mode
        self.progress_callback = progress_callback
        self.parallel = parallel
        self.max_workers = max_workers
        self.silence_tqdm = silence_tqdm
        self.max_error = max_error
        self.kwargs = kwargs


    def fit(self):

        if self.cell is not None:

            fitted_cell = BactFit.fit_cell(
                self.cell,
                refine_fit=self.refine_fit,
                fit_mode=self.fit_mode,
                min_radius=self.min_radius,
                max_radius=self.max_radius,
                silence_tqdm=self.silence_tqdm,
                max_error=self.max_error,)

            return fitted_cell

        if self.celllist is not None:

            fitted_cells = BactFit.fit_celllist(
                self.celllist,
                refine_fit=self.refine_fit,
                fit_mode=self.fit_mode,
                min_radius=self.min_radius,
                max_radius=self.max_radius,
                progress_callback=self.progress_callback,
                parallel=self.parallel,
                max_workers=self.max_workers,
                silence_tqdm=self.silence_tqdm,
                max_error=self.max_error,)

            return fitted_cells


    @staticmethod
    def get_polygon_medial_axis(outline, refine=True):

        if type(outline) == LineString:
            polygon_outline = outline
            polygon = Polygon(outline.coords)
        elif type(outline) == Polygon:
            polygon = outline
            polygon_outline = LineString(polygon.exterior.coords)
        else:
            return None, None

        if len(polygon_outline.coords) < 200:
            polygon_outline = resize_line(polygon_outline, 200)
            polygon = Polygon(polygon_outline.coords)

        # Extract the exterior coordinates of the polygon
        exterior_coords = np.array(polygon.exterior.coords)

        exterior_coords = moving_average(exterior_coords, padding=10, iterations=2)

        # Compute the Voronoi diagram of the exterior coordinates
        vor = Voronoi(exterior_coords)

        # Function to check if a point is inside the polygon
        def point_in_polygon(point, polygon):
            return polygon.contains(Point(point))

        # Extract the medial axis points from the Voronoi vertices

        coords = []

        for i, region in enumerate(vor.regions):
            if -1 not in region:
                try:
                    coords.append(vor.vertices[i].tolist())
                except:
                    pass

        coords = [point for point in coords if point_in_polygon(point, polygon)]

        centroid = polygon.centroid
        cell_radius = polygon_outline.distance(centroid)

        if refine:
            coords = [p for p in coords if polygon_outline.distance(Point(p)) > cell_radius * 0.8]
            coords = np.array(coords)

        return np.array(coords), cell_radius

    @staticmethod
    def bactfit_result(cell, params, cell_polygon, poly_params,
            fit_mode = "directed_hausdorff"):

        x_min = params[0]
        x_max = params[1]
        cell_width = params[2]
        x_offset = params[3]
        y_offset = params[4]
        angle = params[5]

        p = np.poly1d(poly_params)
        x_fitted = np.linspace(x_min, x_max, num=20)
        y_fitted = p(x_fitted)

        x_fitted += x_offset
        y_fitted += y_offset

        midline_coords = np.column_stack((x_fitted, y_fitted))
        midline = LineString(midline_coords)

        midline = rotate_linestring(midline, angle=angle)

        midline_coords = np.array(midline.coords)
        cell_poles = [midline_coords[0], midline_coords[-1]]

        fitted_polygon = midline.buffer(cell_width)

        distance = BactFit.compute_bacfit_distance(fitted_polygon, cell_polygon, fit_mode)

        cell.cell_midline = midline
        cell.cell_polygon = fitted_polygon
        cell.cell_poles = cell_poles
        cell.polynomial_params = poly_params
        cell.fit_error = distance
        cell.cell_width = cell_width
        
        return cell


    @staticmethod
    def refine_function(params, cell_polygon, poly_params,
            fit_mode="directed_hausdorff"):

        """
        Objective function to minimize: the Hausdorff distance between the buffered spline and the target contour.
        """

        try:
            params = list(params)

            x_min = params[0]
            x_max = params[1]
            cell_width = params[2]
            x_offset = params[3]
            y_offset = params[4]
            angle = params[5]

            p = np.poly1d(poly_params)
            x_fitted = np.linspace(x_min, x_max, num=10)
            y_fitted = p(x_fitted)

            x_fitted += x_offset
            y_fitted += y_offset

            midline_coords = np.column_stack((x_fitted, y_fitted))

            midline = LineString(midline_coords)
            midline = rotate_linestring(midline, angle=angle)

            midline_buffer = midline.buffer(cell_width)

            distance = BactFit.compute_bacfit_distance(midline_buffer,
                cell_polygon, fit_mode)

        except:
            distance = np.inf

        return distance

    @staticmethod
    def compute_bacfit_distance(midline_buffer, cell_polygon,
            fit_mode = "directed_hausdorff"):

        try:

            midline_buffer_centroid = midline_buffer.centroid
            cell_polygon_centroid = cell_polygon.centroid

            if fit_mode == "hausdorff":
                # Calculate the Hausdorff distance between the buffered spline and the target contour
                distance = midline_buffer.hausdorff_distance(cell_polygon)
            elif fit_mode == "directed_hausdorff":
                # Calculate directed Hausdorff distance in both directions
                buffer_points = np.array(midline_buffer.exterior.coords)
                contour_points = np.array(cell_polygon.exterior.coords)
                dist1 = directed_hausdorff(buffer_points, contour_points)[0]
                dist2 = directed_hausdorff(contour_points, buffer_points)[0]
                distance = dist1 + dist2

            centroid_distance = midline_buffer_centroid.distance(cell_polygon_centroid)
            distance += centroid_distance**0.5

        except:
            # print(traceback.format_exc())
            distance = np.inf

        return distance

    @staticmethod
    def get_poly_coords(x, y, coefficients, margin=0, n_points=10):
        x1 = np.min(x) - margin
        x2 = np.max(x) + margin

        p = np.poly1d(coefficients)
        x_fitted = np.linspace(x1, x2, num=n_points)
        y_fitted = p(x_fitted)

        return np.column_stack((x_fitted, y_fitted))

    @staticmethod
    def register_fit_data(cell, cell_centre=[], vertical=False):

        if vertical:

            cell_polygon = cell.cell_polygon
            cell_midline = cell.cell_midline

            cell_polygon = rotate_polygon(cell_polygon, angle=-90)
            cell_midline = rotate_linestring(cell_midline, angle=-90)

            midline_coords = np.array(cell_midline.coords)
            cell_poles = [midline_coords[0], midline_coords[-1]]

            cell.cell_polygon = cell_polygon
            cell.cell_midline = cell_midline
            cell.cell_poles = cell_poles

        return cell


    @staticmethod
    def fit_cell(cell, refine_fit=True, fit_mode="directed_hausdorff",
            min_radius=-1, max_radius=-1, polygon_length = 100, max_error = -1, **kwargs):

        try:

            cell_polygon = cell.cell_polygon
            cell_outline = LineString(cell_polygon.exterior.coords)
            vertical = cell.vertical

            if len(cell_outline.coords) < polygon_length:
                cell_outline = resize_line(cell_outline, polygon_length)
                cell_polygon = Polygon(cell_outline.coords)

            if vertical:
                cell_polygon = rotate_polygon(cell_polygon)

            medial_axis_coords, radius = BactFit.get_polygon_medial_axis(cell_polygon)

            if min_radius > 0:
                radius = max(radius, min_radius)
            if max_radius > 0:
                radius = min(radius, max_radius)

            medial_axis_fit, poly_params = fit_poly(medial_axis_coords,
                degree=[1, 2, 3], maxiter=100, minimise_curvature=False)

            x_min = np.min(medial_axis_fit[:, 0])
            x_max = np.max(medial_axis_fit[:, 0])
            x_offset = 0
            y_offset = 0
            rotation = 0
            params = [x_min, x_max, radius, x_offset, y_offset, rotation]

            if refine_fit:

                warnings.filterwarnings("ignore", category=RuntimeWarning)

                bounds = [(None, None),  # x_min
                          (None, None),  # x_max
                          (None, None),  # radius
                          (None, None),  # x_offset
                          (None, None),  # y_offset
                          (None, None)]  # rotation

                if min_radius > 0:
                    bounds[2] = (min_radius, None)
                if max_radius > 0:
                    bounds[2] = (None, max_radius)

                result = minimize(BactFit.refine_function, params,
                    args=(cell_polygon, poly_params, fit_mode),
                    tol=1e-1, options={'maxiter': 500}, bounds=bounds)

                params = result.x

            cell = BactFit.bactfit_result(cell, params, cell_polygon,
                poly_params, fit_mode)

            cell = BactFit.register_fit_data(cell, vertical=vertical)

            if max_error > 0 and hasattr(cell, "fit_error"):
                if cell.fit_error > max_error:
                    cell = None

        except:
            pass

        return cell

    @staticmethod
    def fit_celllist(celllist, refine_fit=True, fit_mode="directed_hausdorff",
            min_radius=-1, max_radius=-1, progress_callback=None, parallel=True,
            max_workers=None, silence_tqdm=False, max_error=-1, **kwargs):

        num_cells = len(celllist.data)

        if parallel:

            if max_workers == None:
                max_workers = os.cpu_count()

            with ProcessPoolExecutor(max_workers=max_workers) as executor:

                futures = {executor.submit(BactFit.fit_cell, cell, refine_fit=refine_fit,
                    fit_mode=fit_mode, min_radius=min_radius, max_radius=max_radius, max_error=max_error): cell
                    for cell in celllist.data}

                completed = 0
                for future in tqdm(as_completed(futures), total=num_cells,
                        desc="Optimising Cells", disable=silence_tqdm):

                    cell = future.result()
                    if cell is not None:
                        if cell.fit_error is not None:
                            cell_index  = cell.cell_index
                            celllist.data[cell_index] = cell

                    completed += 1
                    if progress_callback is not None:
                        progress = (completed / num_cells) * 100
                        progress_callback.emit(progress)

        else:

            iter = 0

            for cell in tqdm(celllist, total=num_cells,
                    desc="Optimising Cells", disable=silence_tqdm):

                cell = BactFit.fit_cell(cell, refine_fit=refine_fit,
                    fit_mode=fit_mode, min_radius=min_radius,
                    max_radius=max_radius)

                if cell.fit_error is not None:
                    cell_index = cell.cell_index
                    celllist.data[cell_index] = cell

                if progress_callback is not None:
                    progress = ((iter + 1) / num_cells)*100
                    progress_callback.emit(progress)

                iter += 1

        #remove cells with no fit error
        celllist.data = [cell for cell in celllist.data if cell.fit_error is not None]
        celllist.assign_cell_indices(reindex=True)

        num_fitted_cells = len(celllist.data)
        print(f"Fit completed for {num_fitted_cells}/{num_cells} cells")

        return celllist
