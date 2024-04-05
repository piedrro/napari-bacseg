import traceback

import numpy as np
import cv2
from shapely.geometry import LineString, LinearRing, Point
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.ops import nearest_points, split
import math
from skimage.morphology import medial_axis, skeletonize, thin
from scipy.spatial.distance import cdist
import warnings
import os
import scipy.io
import matplotlib.path as mpltPath
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt



class _oufti_utils:
    
    def centre_oufti_midlines(self, mode="all"):
        def _event(viewer):
            layer_names = [layer.name for layer in self.viewer.layers]
    
            if "center_lines" in layer_names:

                meta_stack = self.segLayer.metadata.copy()
                mask_stack = self.segLayer.data.copy()
    
                if mode == "active":
                    current_step = self.viewer.dims.current_step[0]
    
                    dim_range = [current_step]
                else:
                    dim_range = np.arange(mask_stack.shape[0])
    
                for i in dim_range:
                    mask = mask_stack[i]
                    meta = meta_stack[i]
    
                    polygons, contours, contour_ids = _oufti_utils.get_mask_polygons(mask)
    
                    if "midlines" in meta.keys():
                        mid_lines = meta["midlines"]
    
                        for j in range(len(mid_lines)):
                            mid_line = mid_lines[j]
    
                            mid_line = np.flip(mid_line)
    
                            index = _oufti_utils.get_contour_index(mid_line, polygons, contour_ids)
    
                            cnt = contours[index]
    
                            left_line, right_line, _, _, _ = _oufti_utils.get_midline_boundary_lines(mid_line, cnt, smooth=False)
    
                            mid_line = _oufti_utils.centre_midline(left_line, right_line, mid_line, export_segments=len(mid_line))
    
                            mid_line = _oufti_utils.interpolate_data(mid_line, export_segments=len(mid_line))
    
                            mid_line = np.flip(mid_line)
    
                            mid_lines[j] = mid_line
    
                        meta["midlines"] = mid_lines
    
                        meta_stack[i] = meta
    
                self.segLayer.metadata = meta_stack
    
                current_fov = self.viewer.dims.current_step[0]
                self._sliderEvent(current_fov)
    
        return _event

    def update_midlines(self):
        layer_names = [layer.name for layer in self.viewer.layers]
    
        if "center_lines" in layer_names:
            polygons = self.shapeLayer.data
    
            current_fov = self.viewer.dims.current_step[0]
    
            self.segLayer.metadata[current_fov]["midlines"] = polygons
            
    def midline_edit_toggle(self, viewer=None):
        layer_names = [layer.name for layer in self.viewer.layers]
    
        if "center_lines" in layer_names:
            self.viewer.layers.selection.select_only(self.shapeLayer)
    
            if self.shapeLayer.mode == "pan_zoom":
                self.shapeLayer.mode = "select"
                self.shapeLayer.selected_data = set(np.arange(len(self.shapeLayer.data)))
                self.shapeLayer.mode = "direct"
                self.oufti_panzoom_mode.setChecked(False)
                self.oufti_edit_mode.setChecked(True)
    
            else:
                self.shapeLayer.mode = "pan_zoom"
                self.oufti_panzoom_mode.setChecked(True)
                self.oufti_edit_mode.setChecked(False)
    
    def generate_midlines(self, mode="all", viewer=None):

        def _event(viewer):

            vertexes = int(self.oufti_midline_vertexes.currentText())
    
            layer_names = [layer.name for layer in self.viewer.layers]
    
            meta = self.segLayer.metadata.copy()
    
            if "center_lines" not in layer_names:
                self.shapeLayer = self.viewer.add_shapes(shape_type='path', edge_width=0.5, opacity=0.5, edge_color='red', face_color='black', name="center_lines")
    
                self.shapeLayer.mouse_drag_callbacks.append(self._segmentationEvents)
    
                self.shapeLayer.events.data.connect(self.update_midlines)
    
            mask_stack = self.segLayer.data.copy()
            polygon_ids = []
    
            if mode == "active":
                current_step = self.viewer.dims.current_step[0]
                dim_range = [current_step]
            else:
                dim_range = np.arange(mask_stack.shape[0])
    
            for i in dim_range:
                mask = mask_stack[i]
    
                mask_ids = np.unique(mask)
    
                polygons = []
    
                for id in mask_ids:
                    if id != 0:
                        try:
                            cell_mask = np.zeros(mask.shape, dtype=np.uint8)
                            cell_mask[mask == id] = 255
    
                            contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
                            cnt = contours[0]
    
                            x, y, w, h = cv2.boundingRect(cnt)
    
                            if h > w:
                                cnt90, shift_xy = _oufti_utils.rotate_contour(cnt, angle=90)
                                mid_line = _oufti_utils.get_voronoi_midline(cnt90, extend=False, vertices=vertexes)
    
                                if mid_line is not None:
                                    mid_line = _oufti_utils.rotate_model(mid_line, shift_xy, angle=-90)
                            else:
                                mid_line = _oufti_utils.get_voronoi_midline(cnt, extend=False, vertices=vertexes)
    
                            if mid_line is not None:
                                mid_line = _oufti_utils.interpolate_data(mid_line, export_segments=vertexes)
    
                                mid_line = _oufti_utils.snap_midline_to_contour(cnt, mid_line)
    
                                mid_line = np.flip(mid_line)
    
                                polygon_ids.append(id)
                                polygons.append(mid_line)

                                print(mid_line)
    
                        except:
                            print(traceback.format_exc())
                            pass
    
                meta[i]["midlines"] = polygons

            self.segLayer.metadata = meta
            self._sliderEvent(0)
            self._update_active_midlines()
    
        return _event
    
    def _update_active_midlines(self):
        try:
            layer_names = [layer.name for layer in self.viewer.layers]
    
            if "center_lines" in layer_names:
                current_fov = self.viewer.dims.current_step[0]
    
                meta = self.segLayer.metadata[current_fov].copy()
    
                if "midlines" in meta.keys():
                    polygons = meta["midlines"]
    
                    self.shapeLayer.data = polygons
                    self.shapeLayer.shape_type = ["path"] * len(polygons)
    
                else:
                    self.shapeLayer.data = []
                    self.shapeLayer.shape_type = []
    
        except:
            pass
    
    @staticmethod
    def find_contours(img):
        # finds contours of shapes, only returns the external contours of the shapes
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
        return contours
    
    @staticmethod
    def moving_average(line, padding=5, iterations=1):
        x, y = line[:, 0], line[:, 1]
    
        x = np.concatenate((x[-padding:], x, x[:padding]))
        y = np.concatenate((y[-padding:], y, y[:padding]))
    
        for i in range(iterations):
            y = np.convolve(y, np.ones(padding), 'same') / padding
            x = np.convolve(x, np.ones(padding), 'same') / padding
    
            x = np.array(x)
            y = np.array(y)
    
        x = x[padding:-padding]
        y = y[padding:-padding]
    
        line = np.stack([x, y]).T
    
        return line
    
    @staticmethod
    def get_voronoi_midline(cnt, smooth=True, voronoi_distance=2, poly_margin=10, poly_limits=[None, 1], extend=True, vertices=100):
        try:
            polygon = cnt.reshape(-1, 2)
    
            if smooth:
                polygon = _oufti_utils.moving_average(polygon)
    
            path = mpltPath.Path(polygon)
    
            vor = Voronoi(cnt.reshape(-1, 2))
    
            vX = vor.vertices.T[0]
            vZ = vor.vertices.T[1]
            vorMask = (vX >= polygon.T[0].min()) & (vX <= polygon.T[0].max()) & (vZ >= polygon.T[1].min()) & (vZ <= polygon.T[1].max())
            verts = vor.vertices[vorMask]
    
            insideMask = path.contains_points(verts)
    
            verts = verts[insideMask]
    
            polygon_distances = np.min(cdist(verts, polygon), axis=1)
    
            verts = verts[polygon_distances > voronoi_distance]
    
            mid_line = _oufti_utils.fit_polyline(verts[:, 0], verts[:, 1], polys=[2, 3], margin=poly_margin, poly_limits=poly_limits)
    
            if extend == False:
                _, _, mid_line, _ = _oufti_utils.get_boundary_lines(mid_line, cnt)
    
            mid_line = LineString(mid_line)
            mid_line = _oufti_utils.resize_line(mid_line, vertices)
            mid_line = np.array(mid_line.xy).T
        except:
            mid_line = None
    
        return mid_line
    
    @staticmethod
    def get_boundary_lines(mid_line, cnt, smooth=True, n_segments=100):
        cnt_array = cnt.reshape(-1, 2)
    
        if smooth:
            cnt_array = _oufti_utils.moving_average(cnt_array)
    
        polygon = Polygon(cnt_array)
        midline_linestring = LineString(mid_line)
    
        intersect_splitter = midline_linestring.intersection(polygon)
        geomcollect = split(polygon, midline_linestring)
        left_line, right_line = geomcollect.geoms[0], geomcollect.geoms[1]
    
        left_line = _oufti_utils.remove_intersecting(left_line, midline_linestring)
        right_line = _oufti_utils.remove_intersecting(right_line, midline_linestring)
    
        distances = np.min(cdist(left_line, right_line), axis=0)
        distances_flip = np.min(cdist(left_line, np.flip(right_line)), axis=0)
    
        distances = np.sum(np.take(distances, [0, -1]))
        distances_flip = np.sum(np.take(distances_flip, [0, -1]))
    
        if distances_flip > distances:
            right_line = np.flip(right_line, axis=0)
    
        p1 = (right_line[0] + left_line[0]) / 2
        p2 = (right_line[-1] + left_line[-1]) / 2
    
        p1 = _oufti_utils.find_closest_point(p1, midline_linestring)
        p2 = _oufti_utils.find_closest_point(p2, midline_linestring)
    
        left_line = np.concatenate(([p1], left_line, [p2]))
        right_line = np.concatenate(([p1], right_line, [p2]))
    
        right_line = np.flip(right_line, axis=0)
    
        left_lineString = LineString(left_line)
        right_lineString = LineString(right_line)
    
        left_lineString = _oufti_utils.resize_line(left_lineString, n_segments)
        right_lineString = _oufti_utils.resize_line(right_lineString, n_segments)
    
        left_line = np.array(left_lineString.xy).T
        right_line = np.array(right_lineString.xy).T
        mid_line = (left_line + np.flipud(right_line)) / 2
    
        return left_line, right_line, mid_line, cnt_array
    
    @staticmethod
    def fit_polyline(x, y, polys=[2, 3], margin=50, poly_limits=None):
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
    
            residual_list = []
            param_list = []
    
            for poly in polys:
                params, residuals, rank, singular_values, rcond = np.polyfit(x.copy(), y.copy(), poly, full=True)
    
                residual_list.append(residuals[0])
                param_list.append(params)
    
            residual_list, param_list = zip(*sorted(zip(residual_list, param_list), reverse=True))
    
            params = param_list[0]
    
            p = np.poly1d(params)
            poly_params = p.c[:2]
    
            if poly_limits != None:
                if len(poly_limits) == len(poly_params):
                    for i, limit in enumerate(poly_limits):
                        if limit != None:
                            if abs(poly_params[i]) >= limit:
                                params = np.polyfit(x.copy(), y.copy(), 1)
                                p = np.poly1d(params)
    
            x1 = np.min(x) - margin
            x2 = np.max(x) + margin
    
            x = np.linspace(x1, x2, 100)
            y = p(x)
    
            fit_line = np.stack([x, y]).T
    
        return fit_line
    
    @staticmethod
    def resize_line(mesh, mesh_length):
        distances = np.linspace(0, mesh.length, mesh_length)
        mesh = LineString([mesh.interpolate(distance) for distance in distances])
    
        return mesh
    
    @staticmethod
    def line_to_array(mesh):
        mesh = np.array([mesh.xy[0][:], mesh.xy[1][:]]).T.reshape(-1, 1, 2)
    
        return mesh
    
    @staticmethod
    def euclidian_distance(x1, y1, x2, y2):
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    
        return distance
    
    @staticmethod
    def polyarea(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    @staticmethod
    def compute_line_metrics(mesh):
        steplength = _oufti_utils.euclidian_distance(mesh[1:, 0] + mesh[1:, 2], mesh[1:, 1] + mesh[1:, 3], mesh[:-1, 0] + mesh[:-1, 2], mesh[:-1, 1] + mesh[:-1, 3]) / 2
    
        steparea = []
        for i in range(len(mesh) - 1):
            steparea.append(_oufti_utils.polyarea([*mesh[i:i + 2, 0], *mesh[i:i + 2, 2][::-1]], [*mesh[i:i + 2, 1], *mesh[i:i + 2, 3][::-1]]))
    
        steparea = np.array(steparea)
    
        d = _oufti_utils.euclidian_distance(mesh[:, 0], mesh[:, 1], mesh[:, 2], mesh[:, 3])
        stepvolume = (d[:-1] * d[1:] + (d[:-1] - d[1:]) ** 2 / 3) * steplength * math.pi / 4
    
        return steplength, steparea, stepvolume
    
    @staticmethod
    def get_mesh(oufti_dict, bisector_length=100, n_segments=100):
        
        left_line = oufti_dict["left_line"]
        right_line = oufti_dict["right_line"]
        mid_line = oufti_dict["mid_line"]
        cnt_array = oufti_dict["cnt_array"]
        end_intersections = oufti_dict["end_intersections"]
        n_segments = oufti_dict["mesh_length"]
    
        left_lineString = LineString(left_line)
        right_lineString = LineString(right_line)
    
        left_lineString = _oufti_utils.resize_line(left_lineString, n_segments)
        right_lineString = _oufti_utils.resize_line(right_lineString, n_segments)
    
        midline_lineString = LineString(mid_line)
        midline_lineString = _oufti_utils.resize_line(midline_lineString, n_segments)
    
        distances = np.linspace(0, midline_lineString.length, n_segments)[1:]
    
        mid_line_segments = [LineString([midline_lineString.interpolate(distance - 0.01), midline_lineString.interpolate(distance + 0.01)]) for distance in distances]
    
        right_line_data = [end_intersections[0].tolist()]
        left_line_data = [end_intersections[0].tolist()]
    
        for segment in mid_line_segments:
            left_bisector = segment.parallel_offset(bisector_length, 'left')
            right_bisector = segment.parallel_offset(bisector_length, 'right')
    
            left_bisector = left_bisector.boundary.geoms[1]
            right_bisector = right_bisector.boundary.geoms[0]
    
            bisector = LineString([left_bisector, right_bisector])
    
            left_intersection = bisector.intersection(left_lineString)
            right_intersection = bisector.intersection(right_lineString)
    
            if left_intersection.type == "Point" and right_intersection.type == "Point":
                right_line_data.append(np.array(left_intersection.xy).reshape(2).tolist())
                left_line_data.append(np.array(right_intersection.xy).reshape(2).tolist())
    
        right_line_data.append(end_intersections[-1].tolist())
        left_line_data.append(end_intersections[-1].tolist())
    
        left_line_data = np.array(left_line_data)
        right_line_data = np.array(right_line_data)
    
        mesh = np.hstack((left_line_data, right_line_data))
        model = np.vstack((left_line_data, np.flipud(right_line_data)))
    
        mesh = mesh + 1
        model = model + 1
    
        steplength, steparea, stepvolume = _oufti_utils.compute_line_metrics(mesh)
    
        polygon = Polygon(model)
        polygon = orient(polygon)
    
        boundingbox = np.asarray(polygon.bounds)
    
        boundingbox[0:2] = np.floor(boundingbox[0:2])
        boundingbox[2:4] = np.ceil(boundingbox[2:4])
        boundingbox[2:4] = boundingbox[2:4] - boundingbox[0:2]
        boundingbox = boundingbox.astype(float)
    
        return mesh, model, steplength, steparea, stepvolume, boundingbox
    
    @staticmethod
    def get_contour_index(midline, polygons, contour_ids):
        intersection_index = None
    
        try:
            intersection_lengths = []
            intersection_ids = []
    
            midline_points = [Point(point) for point in midline]
    
            for i, poly in enumerate(polygons):
                inside_points = [poly.contains(point) for point in midline_points if poly.contains(point) == True]
    
                if len(inside_points) > 0:
                    intersection_lengths.append(len(inside_points))
                    intersection_ids.append(contour_ids[i])
    
            if len(intersection_lengths) != 0:
                intersection_lengths, intersection_ids = zip(*sorted(zip(intersection_lengths, intersection_ids), reverse=True))
    
                intersection_index = contour_ids.index(intersection_ids[0])
    
        except:
            pass
    
        return intersection_index
    
    @staticmethod
    def check_midlines_intersecting(cnt, midline):
        cnt_LinearRing = LinearRing(cnt.reshape(-1, 2))
    
        midline_start_point = Point(midline[0])
        midline_end_point = Point(midline[-1])
    
        if cnt_LinearRing.intersection(midline_start_point).length == 0:
            start_point = np.array(nearest_points(cnt_LinearRing, midline_start_point)[0].xy).T
    
            midline = np.vstack((start_point, midline))
    
        if cnt_LinearRing.intersection(midline_end_point).length == 0:
            end_point = np.array(nearest_points(cnt_LinearRing, midline_end_point)[-1].xy).T
    
            midline = np.vstack((midline, end_point))
    
        return midline
    
    @staticmethod
    def get_midline_boundary_lines(midline, cnt, smooth=True, n_segments=100):
        cnt_array = cnt.reshape(-1, 2)
    
        if smooth:
            cnt_array = _oufti_utils.moving_average(cnt_array)
    
        cnt_LinearRing = LinearRing(cnt.reshape(-1, 2))
    
        midline_start_point = Point(midline[0])
        midline_end_point = Point(midline[-1])
    
        if cnt_LinearRing.intersection(midline_start_point).length == 0:
            start_intersection = np.array(nearest_points(cnt_LinearRing, midline_start_point)[-1].xy).T[0]
    
        if cnt_LinearRing.intersection(midline_end_point).length == 0:
            end_intersection = np.array(nearest_points(cnt_LinearRing, midline_end_point)[0].xy).T[0]
    
        intersection = np.array([start_intersection, end_intersection])
    
        distance = cdist(cnt_array, intersection)
    
        cnt_end_indexes = sorted([np.argmin(dist).tolist() for dist in distance.T])
    
        cnt_array = np.roll(cnt_array, -cnt_end_indexes[0], 0)
        cnt_array = np.append(cnt_array, [cnt_array[0]], 0)
    
        left_line = cnt_array[:cnt_end_indexes[-1] - cnt_end_indexes[0] + 1]
        right_line = cnt_array[cnt_end_indexes[-1] - cnt_end_indexes[0]:]
    
        left_lineString = LineString(left_line)
        right_lineString = LineString(right_line)
    
        left_lineString = _oufti_utils.resize_line(left_lineString, n_segments)
        right_lineString = _oufti_utils.resize_line(right_lineString, n_segments)
    
        left_line = np.array(left_lineString.xy).T
        right_line = np.array(right_lineString.xy).T
        mid_line = (left_line + np.flipud(right_line)) / 2
    
        end_intersections = np.array([mid_line[0], mid_line[-1]])
        end_indicies = np.argmin(cdist(end_intersections, midline), axis=1)
        end_indicies, end_intersections = zip(*sorted(zip(end_indicies, end_intersections), reverse=False))
        end_intersections = list(end_intersections)
    
        return left_line, right_line, mid_line, cnt_array, end_intersections
    
    @staticmethod
    def trim_midline(left_line, right_line, mid_line, margin=10):
        try:
            start_point = left_line[0]
            end_point = left_line[-1]
    
            start_index = np.argmin(cdist([start_point], mid_line))
            end_index = np.argmin(cdist([end_point], mid_line))
    
            if start_index > end_index:
                start_index, end_index = end_index, start_index
    
            end_intersections = [mid_line[start_index], mid_line[end_index]]
    
            margin = 10
    
            if start_index >= margin:
                start_index -= margin
            if end_index <= len(mid_line) + margin:
                end_index += margin
    
            mid_line = mid_line[start_index:end_index]
    
        except:
            pass
    
        return mid_line, end_intersections
    
    @staticmethod
    def remove_intersecting(line, intersecting_line):
        line = LineString(line.exterior)
    
        intersection = line.intersection(intersecting_line)
        intersection = np.array([[geom.xy[0][0], geom.xy[1][0]] for geom in intersection.geoms])
    
        line = np.array(line.xy).T
    
        distance = cdist(line, intersection)
        end_indexes = sorted([np.argmin(dist).tolist() for dist in distance.T])
    
        end_indexes = np.unique(end_indexes).tolist()
    
        if end_indexes[1] - end_indexes[0] > 1:
            line = np.roll(line, -end_indexes[1], 0)
            distance = cdist(line, intersection)
            end_indexes = sorted([np.argmin(dist).tolist() for dist in distance.T])
    
        overlap_length = abs(end_indexes[0] - end_indexes[-1])
    
        line = np.roll(line, -end_indexes[0], 0)
        line = line[overlap_length:]
    
        distances = cdist(line, np.array(intersecting_line.xy).T)
        distances = np.min(distances, axis=1)
        del_indexes = np.argwhere(distances < 0.5).flatten()
    
        line = np.delete(line, del_indexes, axis=0)
    
        return line
    
    @staticmethod
    def dilate_contour(cnt, dilation=0.2):
        mesh_length = len(cnt)
    
        cnt = Polygon(cnt.reshape(-1, 2))
    
        cnt = cnt.buffer(dilation, join_style=2)
    
        cnt = np.array([list(point) for point in cnt.exterior.coords])
    
        cnt = LineString(cnt)
        cnt = _oufti_utils.resize_line(cnt, mesh_length)
    
        cnt = np.array(cnt.xy).T
    
        cnt = cnt.reshape(-1, 1, 2)
    
        return cnt
    
    @staticmethod
    def find_closest_point(point, line):
        point = Point(point)
    
        pol_ext = LinearRing(line)
        d = pol_ext.project(point)
        p = pol_ext.interpolate(d)
        closet_point = list(p.coords)[0]
    
        return closet_point
    
    @staticmethod
    def rotate_contour(cnt, angle=90, units="DEGREES"):
        
        cnt = cnt.copy()
    
        x = cnt[:, :, 1]
        y = cnt[:, :, 0]
    
        x_shift, y_shift = sum(x) / len(x), sum(y) / len(y)
    
        # Shift to origin (0,0)
        x = x - int(x_shift)
        y = y - int(y_shift)
    
        # Convert degrees to radians
        if units == "DEGREES":
            angle = math.radians(angle)
    
        # Rotation matrix multiplication to get rotated x & y
        xr = (x * math.cos(angle)) - (y * math.sin(angle)) + x_shift
        yr = (x * math.sin(angle)) + (y * math.cos(angle)) + y_shift
    
        cnt[:, :, 0] = yr
        cnt[:, :, 1] = xr
    
        shift_xy = [x_shift[0], y_shift[0]]
    
        return cnt, shift_xy
    
    @staticmethod
    def rotate_image(image, shift_xy, angle=90):
        
        x_shift, y_shift = shift_xy
    
        (h, w) = image.shape[:2]
    
        # Perform image rotation
        M = cv2.getRotationMatrix2D((y_shift, x_shift), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
    
        return image, shift_xy
    
    @staticmethod
    def rotate_model(model, shift_xy, angle=-90, units="DEGREES"):
        x = model[:, 1].copy()
        y = model[:, 0].copy()
    
        x_shift, y_shift = shift_xy[0], shift_xy[1]
    
        # Shift to origin (0,0)
        x = x - x_shift
        y = y - y_shift
    
        # Convert degrees to radians
        if units == "DEGREES":
            angle = math.radians(angle)
    
        # Rotation matrix multiplication to get rotated x & y
        xr = (x * math.cos(angle)) - (y * math.sin(angle)) + x_shift
        yr = (x * math.sin(angle)) + (y * math.cos(angle)) + y_shift
    
        model[:, 0] = yr
        model[:, 1] = xr
    
        return model
    
    @staticmethod
    def snap_midline_to_contour(cnt, midline, fit=False):
        try:
            cnt_array = cnt.reshape(-1, 2)
            cnt_poly = Polygon(cnt_array)
    
            if fit == True:
                midline = _oufti_utils.fit_polyline(midline[:, 0], midline[:, 1], polys=[2], margin=5, poly_limits=[None, None])
    
                del_indexes = []
    
                for i, point in enumerate(midline):
                    if cnt_poly.contains(Point(point)) == False:
                        del_indexes.append(i)
    
                midline = np.delete(midline, del_indexes, axis=0)
    
            start_index = np.argmin(cdist(cnt_array, [midline[0]]), axis=0)[0]
            end_index = np.argmin(cdist(cnt_array, [midline[-1]]), axis=0)[0]
    
            midline[0] = cnt_array[start_index]
            midline[-1] = cnt_array[end_index]
    
        except:
            pass
    
        return midline
    

    def get_oufti_data(self, image, mask, midlines=None):
    
        mesh_length = int(self.oufti_mesh_length.currentText())
        mesh_dilation = float(self.oufti_mesh_dilation.currentText())
    
        mask_ids = np.unique(mask)
    
        polygons = []
        contours = []
        contour_ids = []
    
        for i in range(len(mask_ids)):
            try:
                id = mask_ids[i]
    
                if id != 0:
                    cell_mask = np.zeros(mask.shape, dtype=np.uint8)
                    cell_mask[mask == id] = 255
    
                    cnt, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
                    cnt = cnt[0]
    
                    polygon = Polygon(cnt.reshape(-1, 2))
    
                    polygons.append(polygon)
                    contours.append(cnt)
                    contour_ids.append(id)
    
            except:
                pass
    
        oufti_data = []
    
        if midlines is None:
            for i in range(len(contours)):
                try:
                    cnt = contours[i]
    
                    x, y, w, h = cv2.boundingRect(cnt)
    
                    cnt = _oufti_utils.dilate_contour(cnt, mesh_dilation)
    
                    if h > w:
                        cnt90 = cnt.copy()
                        cnt90, shift_xy = _oufti_utils.rotate_contour(cnt90, angle=90)
                        mid_line = _oufti_utils.get_voronoi_midline(cnt90)
                        mid_line = _oufti_utils.rotate_model(mid_line, shift_xy, angle=-90)
    
                    else:
                        mid_line = _oufti_utils.get_voronoi_midline(cnt)
    
                    left_line, right_line, _, cnt_array = _oufti_utils.get_boundary_lines(mid_line, cnt, smooth=True, n_segments=100)
    
                    mid_line, end_intersections = _oufti_utils.trim_midline(left_line, right_line, mid_line, margin=10)
    
                    oufti_dict = {"cnt": cnt, "end_intersections": end_intersections, "cnt_array": cnt_array, "mid_line": mid_line, "left_line": left_line, "right_line": right_line, "mask_shape": mask.shape, "mesh_length": mesh_length}
    
                    oufti_data.append(oufti_dict)
    
                except:
                    # print(traceback.format_exc())
                    pass
    
        else:
            for i in range(len(midlines)):
                try:
                    midline = midlines[i].copy()
                    midline = np.flip(midline)
    
                    index = _oufti_utils.get_contour_index(midline, polygons, contour_ids)
    
                    if index != None:
                        cnt = contours[index]
    
                        if mesh_dilation == 0:
                            midline = _oufti_utils.snap_midline_to_contour(cnt, midline, fit=False)
                        else:
                            cnt = _oufti_utils.dilate_contour(cnt, mesh_dilation)
                            midline = _oufti_utils.snap_midline_to_contour(cnt, midline, fit=True)
    
                        left_line, right_line, _, cnt_array = _oufti_utils.get_boundary_lines(midline, cnt, smooth=True, n_segments=100)
    
                        _, end_intersections = _oufti_utils.trim_midline(left_line, right_line, midline, margin=10)
    
                        oufti_dict = {"cnt": cnt, "end_intersections": end_intersections, "cnt_array": cnt_array, "mid_line": midline, 
                                      "left_line": left_line, "right_line": right_line, "mask_shape": mask.shape, "mesh_length": mesh_length}
    
                        oufti_data.append(oufti_dict)
    
                except:
                    pass
    
        return oufti_data
    
    @staticmethod
    def centre_midline(left_line, right_line, midline, fit_segments=100, export_segments=6):
        left_LineString = LineString(left_line)
        right_LineString = LineString(right_line)
    
        midline_LineString = LineString(midline)
        midline_LineString = _oufti_utils.resize_line(midline_LineString, fit_segments)
    
        midline = np.array(midline_LineString.xy).T
    
        for i in range(1, len(midline) - 1):
            point = Point(midline[i])
    
            left_intersection = np.array(nearest_points(left_LineString, point)[0].xy).T[0]
            right_intersection = np.array(nearest_points(right_LineString, point)[0].xy).T[0]
    
            new_point = np.mean([left_intersection, right_intersection], axis=0)
    
            midline[i] = new_point
    
        midline_LineString = LineString(midline)
        midline_LineString = _oufti_utils.resize_line(midline_LineString, export_segments)
    
        midline = np.array(midline_LineString.xy).T
    
        return midline
    
    @staticmethod
    def get_mask_polygons(mask):
        mask_ids = np.unique(mask)
    
        polygons = []
        contours = []
        contour_ids = []
    
        for i in range(len(mask_ids)):
            id = mask_ids[i]
    
            if id != 0:
                try:
                    cell_mask = np.zeros(mask.shape, dtype=np.uint8)
                    cell_mask[mask == id] = 255
    
                    cnt, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
                    cnt = cnt[0]
    
                    polygon = Polygon(cnt.reshape(-1, 2))
    
                    polygons.append(polygon)
                    contours.append(cnt)
                    contour_ids.append(id)
    
                except:
                    pass
    
        return polygons, contours, contour_ids
    
    @staticmethod
    def interpolate_data(line, export_segments=100):
        from scipy import interpolate
        from scipy.interpolate import splev
    
        interpolations_methods = ['slinear', 'quadratic', 'cubic']
    
        x = line[:, 0]
        y = line[:, 1]
    
        # Linear length along the line:
        distance = np.cumsum(np.sqrt(np.sum(np.diff(line, axis=0) ** 2, axis=1)))
        distance = np.insert(distance, 0, 0)
    
        alpha = np.linspace(distance.min(), int(distance.max()), export_segments)
        interpolator = interpolate.interp1d(distance, line, kind='quadratic', axis=0)
    
        interpolated_points = interpolator(alpha)
    
        out_x = interpolated_points.T[0]
        out_y = interpolated_points.T[1]
    
        out_line = np.stack([out_x, out_y]).T
    
        out_line[0] = line[0]
        out_line[-1] = line[-1]
    
        return out_line
    
    @staticmethod
    def export_oufti(image, oufti_data, file_path):
        file_path = os.path.splitext(file_path)[0] + ".mat"
    
        cell_data = []
    
        for i in range(len(oufti_data)):
            try:
                mesh, model, steplength, steparea, stepvolume, boundingbox = _oufti_utils.get_mesh(oufti_data[i])
    
                cell_struct = {'mesh': mesh, 'model': model, 'birthframe': 1, 'divisions': [], 'ancestors': [], 'descendants': [], 'timelapse': False, 'algorithm': 5, 'polarity': 0, 'stage': 1, 'box': boundingbox, 'steplength': steplength, 'length': np.sum(steplength), 'lengthvector': steplength, 'steparea': steparea, 'area': np.sum(steparea), 'stepvolume': stepvolume.T, 'volume': np.sum(stepvolume)}
    
                cell_data.append(cell_struct)
    
            except:
                print(traceback.format_exc())
                pass
    
        cellListN = len(cell_data)
        cellList = np.zeros((1,), dtype=object)
        cellList_items = np.zeros((1, cellListN), dtype=object)
    
        microbeTrackerParamsString = "% This file contains MicrobeTracker settings optimized for wildtype E. coli cells at 0.114 um/pixel resolution (using algorithm 4)\n\nalgorithm = 4\n\n% Pixel-based parameters\nareaMin = 120\nareaMax = 2200\nthresFactorM = 1\nthresFactorF = 1\nsplitregions = 1\nedgemode = logvalley\nedgeSigmaL = 3\nedveSigmaV = 1\nvalleythresh1 = 0\nvalleythresh2 = 1\nerodeNum = 1\nopennum = 0\nthreshminlevel = 0.02\n\n% Constraint parameters\nfmeshstep = 1\ncellwidth =6.5\nfsmooth = 18\nimageforce = 4\nwspringconst = 0.3\nrigidityRange = 2.5\nrigidity = 1\nrigidityRangeB = 8\nrigidityB = 5\nattrCoeff = 0.1\nrepCoeff = 0.3\nattrRegion = 4\nhoralign = 0.2\neqaldist = 2.5\n\n% Image force parameters\nfitqualitymax = 0.5\nforceWeights = 0.25 0.5 0.25\ndmapThres = 2\ndmapPower = 2\ngradSmoothArea = 0.5\nrepArea = 0.9\nattrPower = 4\nneighRep = 0.15\n\n% Mesh creation parameters\nroiBorder = 20.5\nnoCellBorder = 5\nmaxmesh = 1000\nmaxCellNumber = 2000\nmaxRegNumber = 10000\nmeshStep = 1\nmeshTolerance = 0.01\n\n% Fitting parameters\nfitConvLevel = 0.0001\nfitMaxIter = 500\nmoveall = 0.1\nfitStep = 0.2\nfitStepM = 0.6\n\n% Joining and splitting\nsplitThreshold = 0.35\njoindist = 5\njoinangle = 0.8\njoinWhenReuse = 0\nsplit1 = 0\n\n% Other\nbgrErodeNum = 5\nsgnResize = 1\naligndepth = 1"
    
        for i in range(len(cell_data)):
            cellList_items[0, i] = cell_data[i]
    
        cellList[0] = cellList_items
    
        p = [];
        paramString = np.empty((len(microbeTrackerParamsString.split('\n')), 1), dtype=object)
        paramSplit = microbeTrackerParamsString.split('\n')
        for p_index in range(len(microbeTrackerParamsString.split('\n'))):
            paramString[p_index] = paramSplit[p_index]
    
        outdict = {'cellList': cellList, 'cellListN': cellListN, 'coefPCA': [], 'mCell': [], 'p': [], 'paramString': paramString, 'rawPhaseFolder': [], 'shiftfluo': np.zeros((
        2, 2)), 'shiftframes': [], 'weights': []}
    
        scipy.io.savemat(file_path, outdict)
