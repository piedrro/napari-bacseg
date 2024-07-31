import numpy as np
import traceback

import pandas as pd
from shapely.geometry import Point, LineString, Polygon
from shapely.strtree import STRtree
from napari_bacseg.bactfit.utils import resize_line, rotate_linestring, fit_poly, get_vertical, get_polygon_midline
import matplotlib.pyplot as plt
import math
from shapely.affinity import rotate, translate
import cv2

def find_centerline(midline, width, smooth=True):

    centerline = None

    try:
        def extract_end_points(line, num_points=2):
            coords = list(line.coords)
            if len(coords) < num_points:
                raise ValueError("The LineString does not have enough points.")
            start_points = coords[:num_points]
            end_points = coords[-num_points:]
            return start_points, end_points

        def extend_away(points, distance, ):
            if len(points) < 2:
                raise ValueError("At least two points are required to determine the direction for extension.")

            p1 = Point(points[0])
            p2 = Point(points[1])

            dx = p2.x - p1.x
            dy = p2.y - p1.y
            length = np.hypot(dx, dy)
            factor = distance / length

            # Extend p1 away from p2
            extended_x1 = p1.x - factor * dx
            extended_y1 = p1.y - factor * dy

            # Similarly for the other end
            p3 = Point(points[-1])
            p4 = Point(points[-2])

            dx_end = p4.x - p3.x
            dy_end = p4.y - p3.y
            length_end = np.hypot(dx_end, dy_end)
            factor_end = distance / length_end

            # Extend p3 away from p4
            extended_x2 = p3.x - factor_end * dx_end
            extended_y2 = p3.y - factor_end * dy_end

            return (extended_x1, extended_y1), (extended_x2, extended_y2)

        model = midline.buffer(width)

        centerline = resize_line(midline, 1000)  # High resolution with 1000 points

        start_points, end_points = extract_end_points(centerline)

        extension_distance = width * 3

        extended_start = extend_away(start_points, extension_distance)
        extended_end = extend_away(end_points, extension_distance)

        extended_start_line = LineString([start_points[0], extended_start[0]])
        extended_end_line = LineString([end_points[-1], extended_end[1]])

        outline = LineString(model.exterior.coords)
        intersections_start = outline.intersection(extended_start_line).coords[0]
        intersections_end = outline.intersection(extended_end_line).coords[0]

        centerline_coords = np.array(centerline.coords)
        centerline_coords = np.insert(centerline_coords, 0, intersections_start, axis=0)
        centerline_coords = np.append(centerline_coords, [intersections_end], axis=0)
        centerline = LineString(centerline_coords)

        if smooth:
            vertical = get_vertical(model)

            if vertical:
                centerline = rotate_linestring(centerline, angle=90)

            centerline_coords = np.array(centerline.coords)
            constraining_points = [centerline_coords[0], centerline_coords[-1]]

            centerline_coords, _ = fit_poly(centerline_coords, degree=[1, 2, 3], maxiter=100, constraining_points=constraining_points, constrained=True)

            centerline = LineString(centerline_coords)

            if vertical:
                centerline = rotate_linestring(centerline, angle=-90)

    except:
        print(traceback.format_exc())
        pass

    return centerline



def split_linestring(linestring, num_segments):
    points = [linestring.interpolate(float(i) / num_segments, normalized=True) for i in range(num_segments + 1)]
    return [LineString([points[i], points[i + 1]]) for i in range(num_segments)]


def calculate_angle2(segment: LineString, point: Point, centroid: Point) -> float:
    """
    Calculate the angle between the point and the center of the segment, using the center of the source midline.
    """
    
    line_start = segment.coords[0]
    line_end = segment.coords[1]
    
    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]
    segment_angle = math.atan2(dy, dx)
    point_angle = math.atan2(point.y - line_start[1], point.x - line_start[0])
    angle = point_angle - segment_angle
    
    return math.degrees(angle)

def calculate_new_point2(segment: LineString, centroid: Point,
                         distance: float, angle: float):
    """
    Calculate new point coordinates in the target segment given the distance and angle.
    """

    line_start = segment.coords[0]
    line_end = segment.coords[1]
    
    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]
    segment_angle = math.atan2(dy, dx)
    
    angle_rad = math.radians(angle + math.degrees(segment_angle))
    dx = distance * np.cos(angle_rad)
    dy = distance * np.sin(angle_rad)
    
    new_x = line_start[0] + dx
    new_y = line_start[1] + dy
    
    return new_x, new_y




def calculate_angle(segment, point):
    # Calculate the angle between the segment and the point
    line_start = segment.coords[0]
    line_end = segment.coords[1]
    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]
    segment_angle = math.atan2(dy, dx)
    point_angle = math.atan2(point.y - line_start[1], point.x - line_start[0])
    angle = point_angle - segment_angle
    return math.degrees(angle)

def calculate_new_point(segment, distance, angle):
     line_start = segment.coords[0]
     line_end = segment.coords[1]
     segment_angle = math.atan2(line_end[1] - line_start[1], line_end[0] - line_start[0])
     new_angle = segment_angle + angle
     new_x = line_start[0] + distance * math.cos(new_angle)
     new_y = line_start[1] + distance * math.sin(new_angle)
     return [new_x, new_y]

def reflect_loc_horizontally(loc, centroid):

    center_y = centroid[1]
    centroid_distance = loc.y - center_y

    if centroid_distance < 0:
        loc.y = loc.y + abs(centroid_distance)*2
    else:
        loc.y = loc.y - abs(centroid_distance)*2

    return loc

def reflect_loc_vertically(loc, centroid):

    center_x = centroid[0]
    centroid_distance = loc.x - center_x

    if centroid_distance < 0:
        loc.x = loc.x + abs(centroid_distance)*2
    else:
        loc.x = loc.x - abs(centroid_distance)*2

    return loc


def remove_locs_outside_cell(locs, polygon):

    try:

        if type(locs) != np.recarray:
            return None

        polygon_locs = None

        coords = np.stack([locs["x"], locs["y"]], axis=1)
        points = [Point(coord) for coord in coords]
        spatial_index = STRtree(points)

        possible_points = spatial_index.query(polygon)

        polygon_point_indices = []

        for point_index in possible_points:
            point = points[point_index]

            if polygon.contains(point):
                polygon_point_indices.append(point_index)

        if len(polygon_point_indices) > 0:
            polygon_locs = locs[polygon_point_indices]

    except:
        print(traceback.format_exc())
        polygon_locs = None

    return polygon_locs

def compute_vectors(segments):
    unit_vectors = []
    perpendicular_vectors = []
    for segment in segments:
        segment_start = np.array(segment.coords[0])
        segment_end = np.array(segment.coords[1])
        segment_vector = segment_end - segment_start
        segment_length = np.linalg.norm(segment_vector)
        unit_vector = segment_vector / segment_length
        perpendicular_vector = np.array([-unit_vector[1], unit_vector[0]])
        unit_vectors.append(unit_vector)
        perpendicular_vectors.append(perpendicular_vector)
    return unit_vectors, perpendicular_vectors


def cell_coordinate_transformation(cell, target_cell,
        method = "angular", n_segments=1000, progress_list = [], shape_measurements=True):

    if method == "angular":

        cell = angular_coordinate_transformation(cell, target_cell,
            n_segments, progress_list, shape_measurements)

    elif method == "perpendicular":

        cell = perpendicular_coordinate_transformation(cell, target_cell,
            n_segments, progress_list, shape_measurements)

    return cell


def plot_cell(polygon=None, locs=None, midline=None, title=None):

    try:

        if type(polygon) == Polygon:
            polygon_coords = np.array(polygon.exterior.coords)
            plt.plot(*polygon_coords.T, color='black')
        if type(locs) == np.recarray:
            coords = np.stack([locs["x"], locs["y"]], axis=1)
            plt.scatter(*coords.T, color='red', s=1)
        if type(midline) == LineString:
            plt.plot(*np.array(midline.coords).T, color='blue')
        if title:
            plt.title(title)

        plt.show()

    except:
        print(traceback.format_exc())
        pass

def perpendicular_bisector(segment, width):
    mid_point = segment.interpolate(0.5, normalized=True)
    start, end = segment.coords
    dx, dy = end[0] - start[0], end[1] - start[1]
    bisector = LineString([(-dy, dx), (dy, -dx)])  # Create a perpendicular line
    bisector = translate(bisector, mid_point.x, mid_point.y)  # Translate to midpoint
    scale_factor = width / bisector.length / 2  # Scale factor to get the desired length on each side
    bisector = LineString([mid_point, (mid_point.x - scale_factor * dy, mid_point.y + scale_factor * dx),
                           (mid_point.x + scale_factor * dy, mid_point.y - scale_factor * dx)])
    return bisector

def build_strtree_index(bisectors):
    return STRtree(bisectors)

def find_closest_bisector_with_strtree(strtree, bisectors, point):
    nearest_geom = strtree.nearest(point)
    closest_index = bisectors.index(nearest_geom)
    min_distance = point.distance(nearest_geom)
    return closest_index, min_distance


def compute_vectors(segments):
    unit_vectors = []
    perpendicular_vectors = []
    for segment in segments:
        segment_start = np.array(segment.coords[0])
        segment_end = np.array(segment.coords[1])
        segment_vector = segment_end - segment_start
        segment_length = np.linalg.norm(segment_vector)
        unit_vector = segment_vector / segment_length
        perpendicular_vector = np.array([-unit_vector[1], unit_vector[0]])
        unit_vectors.append(unit_vector)
        perpendicular_vectors.append(perpendicular_vector)
    return unit_vectors, perpendicular_vectors


def perpendicular_coordinate_transformation(cell, target_cell,
        n_segments=1000, progress_list = [], shape_measurements=True):

    transformed_locs = []

    try:

        cell.locs = remove_locs_outside_cell(cell.locs,
            cell.cell_polygon)

        locs = cell.locs

        if type(locs) != np.recarray:
            return cell

        if len(locs) == 0:
            return cell

        source_polygon = cell.cell_polygon
        source_midline = cell.cell_midline
        source_width = cell.width

        if cell.cell_centerline is None:
            cell.cell_centerline = find_centerline(source_midline, source_width)
            source_centerline = cell.cell_centerline

        target_width = target_cell.width
        target_centerline = target_cell.cell_centerline
        target_polygon = target_cell.cell_polygon

        source_segments = split_linestring(source_centerline, n_segments)
        target_segments = split_linestring(target_centerline, n_segments)

        # Precompute vectors for source and target segments
        source_unit_vectors, source_perpendicular_vectors = compute_vectors(source_segments)
        target_unit_vectors, target_perpendicular_vectors = compute_vectors(target_segments)

        # Create STRtree for segments
        tree = STRtree(source_segments)

        for loc in locs:

            try:

                point = Point(loc["x"], loc["y"])

                # Find the nearest segment to each point
                closest_segment_index = tree.nearest(point)

                nearest_segment = source_segments[closest_segment_index]

                segment_start = np.array(nearest_segment.coords[0])
                segment_vector = source_unit_vectors[closest_segment_index]
                point_vector = np.array([loc["x"], loc["y"]]) - segment_start

                # Compute the signed distance from the point to the nearest segment
                distance = point.distance(nearest_segment)
                distance_sign = np.sign(np.cross(segment_vector, point_vector))

                signed_distance = distance*distance_sign
                # Compute the new distance in the target coordinate system
                new_distance = target_width * (signed_distance / source_width)

                # Calculate the new coordinates by moving perpendicular to the target segment at the new distance
                segment_start_target = np.array(target_segments[closest_segment_index].coords[0])
                perp_vector = target_perpendicular_vectors[closest_segment_index]
                new_point_coords = segment_start_target + perp_vector * new_distance

                if target_polygon.contains(Point(new_point_coords)):

                    tloc = loc.copy()
                    tloc["x"] = new_point_coords[0]
                    tloc["y"] = new_point_coords[1]

                    transformed_locs.append(tloc)
            except:
                pass

            progress_list.append(1)

        if len(transformed_locs) > 0:
            transformed_locs = np.hstack(transformed_locs).view(np.recarray).copy()

            cell.locs = transformed_locs
            cell.cell_polygon = target_polygon

            if shape_measurements:
                cell = compute_shape_measurements(cell)

        else:
            cell.locs = None

    except:
        print(traceback.format_exc())
        pass

    return cell



def get_pixel_values(image, points, upscale=1):
    
    image = cv2.resize(image, None, fx=upscale, fy=upscale, 
                        interpolation=cv2.INTER_LINEAR)
    
    points = [(point.x, point.y) for point in points]
    
    pixel_values = []
    
    for point in points:
        x, y = point
        # Get the pixel value at the given point
        pixel_values.append(image[int(y), int(x)])
        
    return pixel_values


def scale_polygon(polygon, scale):
    scaled_coords = [(x * scale, y * scale) for x, y in polygon.exterior.coords]
    return Polygon(scaled_coords)

def scale_midline(midline, scale):
    scaled_coords = [(x * scale, y * scale) for x, y in midline.coords]
    return LineString(scaled_coords)

def angular_pixel_transformation(cell, target_cell, upscale = 2, n_segments=1000):
    
    try:
        
        
        if cell.data == {}:
            return None
        
        if cell.cell_polygon == None:
            return None
        
        if cell.cell_midline == None:
            midline, width = get_polygon_midline(cell.cell_polygon)
            cell.cell_midline = midline
            cell.cell_width = width
            
        if cell.cell_midline == None:
            return None
            
        source_polygon = cell.get_image_polygon()
        source_midline = cell.get_image_midline()
        source_polygon = scale_polygon(source_polygon, upscale)
        source_midline = scale_midline(source_midline, upscale)
        source_centroid = source_midline.centroid
        
        source_width = cell.cell_width
        target_width = target_cell.cell_width
        target_image = target_cell.get_image()


        target_midline = target_cell.cell_midline
        target_polygon = target_cell.cell_polygon
        target_centroid = target_polygon.centroid
        
        source_segments = split_linestring(source_midline, n_segments)
        target_segments = split_linestring(target_midline, n_segments)
        
        # Create STRtree for segments
        tree = STRtree(source_segments)
        
        # Calculate the centroid of the midline
        centroid = source_polygon.centroid
        cx, cy = int(centroid.x), int(centroid.y)
        
        # Get the bounding box of the polygon
        minx, miny, maxx, maxy = map(int, source_polygon.bounds)
        
        # Generate all integer points within the bounding box
        points = [Point((x, y)) for x in range(minx, maxx + 1) for y in range(miny, maxy + 1)]
        
        # Filter points that are inside the polygon
        points = [point for point in points if source_polygon.contains(point) or 
                  source_polygon.touches(point)]
        
        # points_coords = np.array([point.coords for point in points])
        # plt.scatter(*points_coords.T)
        # plt.show()
        
        # image = cell.data["NR"]
        # image = cv2.resize(image, None, fx=upscale, fy=upscale, 
        #                     interpolation=cv2.INTER_LINEAR)
        
        pixel_dict = {channel: get_pixel_values(image, points, upscale) for channel, image in cell.data.items()}
        
        transformed_data = {channel:target_image.copy() for channel in cell.data.keys()}
        
        n_transformed = 0
        
        polygon_coords = np.array(target_polygon.exterior.coords)
        midline_coords = np.array(target_midline.coords)
        
        for point_index, point in enumerate(points):

            try:
                
                if point_index <= 10000:

                    closest_segment_index = tree.nearest(point)
                    nearest_segment = source_segments[closest_segment_index]
    
                    source_distance = point.distance(nearest_segment)
                    angle = calculate_angle2(nearest_segment, point, source_centroid)
                    
                    if source_distance > source_width:
                        source_distance = source_width
                        
                    target_distance = target_width * (source_distance / source_width)
                    
                    # Use the corresponding target segment
                    target_segment = target_segments[closest_segment_index]
    
                    # Calculate the new point in the target segment
                    new_point_coords = calculate_new_point2(target_segment, target_centroid, target_distance, angle)
                    new_point = Point(new_point_coords)
                    
                if target_polygon.contains(new_point) or target_polygon.touches(new_point):
                    
                    n_transformed +=1
                    
                    plt.scatter(new_point_coords[0], new_point_coords[1])
                
                    for channel, image in transformed_data.items():
                        
                        pixel_value = pixel_dict[channel][point_index]
                        iX, iY = new_point_coords
                        
                        image[int(iY),int(iX)] = pixel_value
                        transformed_data[channel] = image
                        
            except:
                print(traceback.format_exc())
                
        plt.show()
                


        # if transformed_data != {}:
        #     target_cell.data = transformed_data

    except:
        print(traceback.format_exc())
        
    return target_cell


def check_point_inside_polygon(polygon, point):
    
    if isinstance(polygon, np.ndarray):
        polygon = Polygon(polygon)

    if type(point) in [np.ndarray, list, tuple]:
        point = Point(point)
        
    if polygon.contains(point) or polygon.touches(point):
        return True
    else:
        return False

        
def angular_coordinate_transformation(cell, target_cell,
        n_segments=1000, shape_measurements=True, progress_list = []):


    try:

        cell.locs = remove_locs_outside_cell(cell.locs,
            cell.cell_polygon)

        locs = cell.locs

        if type(locs) != np.recarray:
            return cell

        if len(locs) == 0:
            return cell

        source_polygon = cell.cell_polygon
        source_midline = cell.cell_midline
        source_centroid = source_midline.centroid

        source_width = cell.cell_width
        target_width = target_cell.cell_width

        target_midline = target_cell.cell_midline
        target_polygon = target_cell.cell_polygon
        target_centroid = target_polygon.centroid

        source_segments = split_linestring(source_midline, n_segments)
        target_segments = split_linestring(target_midline, n_segments)

        # Create STRtree for segments
        tree = STRtree(source_segments)

        transformed_locs = []

        for loc in locs:

            try:

                point = Point(loc["x"], loc["y"])

                closest_segment_index = tree.nearest(point)
                nearest_segment = source_segments[closest_segment_index]

                source_distance = point.distance(nearest_segment)
                angle2 = calculate_angle2(nearest_segment, point, source_centroid)

                target_distance = target_width * (source_distance / source_width)
                
                # Use the corresponding target segment
                target_segment = target_segments[closest_segment_index]

                # Calculate the new point in the target segment
                new_point_coords = calculate_new_point2(target_segment, target_centroid, 
                                                        target_distance, angle2)

                if len(new_point_coords) == 0:
                    continue

                if check_point_inside_polygon(target_polygon, new_point_coords):

                    tloc = loc.copy()
                    tloc["x"] = new_point_coords[0]
                    tloc["y"] = new_point_coords[1]

                    transformed_locs.append(tloc)

            except:
                print(traceback.format_exc())
                pass

        if len(transformed_locs) > 0:

            transformed_locs = np.hstack(transformed_locs).view(np.recarray).copy()

            cell.locs = transformed_locs
            cell.cell_polygon = target_polygon
            cell.cell_midline = target_midline

            if shape_measurements:
                cell = compute_shape_measurements(cell)

        else:
            cell.locs = None

    except:
        print(traceback.format_exc())
        cell.locs = None
        pass

    return cell

def compute_shape_measurements(cell):

    try:

        locs = cell.locs
        midline = cell.cell_midline
        polygon = cell.cell_polygon
        pixel_size = cell.pixel_size

        locs = pd.DataFrame(locs)
        locs["pixel_size (nm)"] = pixel_size
        locs["centroid_distance (nm)"] = None
        locs["membrane_distance (nm)"] = None
        locs["midline_distance (nm)"] = None
        locs["cell_pole_distance (nm)"] = None

        if len(locs) == 0:
            return None

        coords = np.array(locs[["x", "y"]].values)

        try:

            polygon_centroid = polygon.centroid.coords[0]

            centroid_distance = [Point(coord).distance(Point(polygon_centroid)) for coord in coords]

            if len(centroid_distance) == len(coords):
                centroid_distance = np.array(centroid_distance)
                centroid_distance = centroid_distance * pixel_size
                centroid_distance = centroid_distance.tolist()
                locs["centroid_distance (nm)"] = centroid_distance

                print(centroid_distance)

        except:
            pass

        try:

            polygon_coords = np.array(polygon.exterior.coords)
            polygon_outline = LineString(polygon_coords)

            membrane_distance = [Point(coord).distance(polygon_outline) for coord in coords]

            if len(membrane_distance) == len(coords):
                membrane_distance = np.array(membrane_distance)
                membrane_distance = membrane_distance * pixel_size
                membrane_distance = membrane_distance.tolist()
                locs["membrane_distance (nm)"] = membrane_distance

        except:
            pass

        try:

            midline_distance = [Point(coord).distance(midline) for coord in coords]

            if len(midline_distance) == len(coords):
                midline_distance = np.array(midline_distance)
                midline_distance = midline_distance * pixel_size
                midline_distance = midline_distance.tolist()
                locs["midline_distance (nm)"] = midline_distance

        except:
            pass

        try:
            midline_coords = np.array(midline.coords)
            cell_poles = [Point(midline_coords[0]), Point(midline_coords[-1])]

            cell_pole_distance = [min([Point(coord).distance(pole) for pole in cell_poles]) for coord in coords]

            if len(cell_pole_distance) == len(coords):
                cell_pole_distance = np.array(cell_pole_distance)
                cell_pole_distance = cell_pole_distance * pixel_size
                cell_pole_distance = cell_pole_distance.tolist()
                locs["cell_pole_distance (nm)"] = cell_pole_distance
        except:
            pass

        if len(locs) > 0:
            cell.locs = locs.to_records(index=False)

    except:
        print(traceback.format_exc())
        pass

    return cell


