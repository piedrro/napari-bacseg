import multiprocessing
import traceback

import numpy as np
import cv2
from shapely.geometry import LineString, Polygon
from shapely.geometry.polygon import orient
import math
from colicoords import Data, Cell, config
from multiprocessing import Pool
import psutil
import warnings
import pickle
from skimage import exposure
from napari.utils.notifications import show_info


def normalize99(X):
    """ normalize image so 0.0 is 0.01st percentile and 1.0 is 99.99th percentile """

    if np.max(X) > 0:
        X = X.copy()
        v_min, v_max = np.percentile(X[X != 0], (1, 99))
        X = exposure.rescale_intensity(X, in_range=(v_min, v_max))

    return X


def rescale01(x):
    """ normalize image from 0 to 1 """

    if np.max(x) > 0:
        x = (x - np.min(x)) / (np.max(x) - np.min(x))

    return x


def find_contours(img):
    # finds contours of shapes, only returns the external contours of the shapes
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours


def rotate_contour(cnt, angle=90, units="DEGREES"):
    x = cnt[:, :, 1].copy()
    y = cnt[:, :, 0].copy()

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


def rotate_image(image, shift_xy, angle=90):
    x_shift, y_shift = shift_xy

    (h, w) = image.shape[:2]

    # Perform image rotation
    M = cv2.getRotationMatrix2D((y_shift, x_shift), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h))

    return image, shift_xy


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


def get_cell_images(self, image, mask, mask_ids=None):
    if mask_ids == None:
        mask_ids = np.unique(mask)

    cell_data = {}

    for i in range(len(mask_ids)):
        mask_id = mask_ids[i]

        if mask_id != 0:
            cell_image = image.copy()

            cell_mask = np.zeros(mask.shape, dtype=np.uint8)
            cell_mask[mask == mask_id] = 1

            inverted_cell_mask = np.zeros(mask.shape, dtype=np.uint8)
            inverted_cell_mask[mask != 0] = 1
            inverted_cell_mask[mask == mask_id] = 0

            cnt = find_contours(cell_mask)[0]

            x, y, w, h = cv2.boundingRect(cnt)

            if h > w:
                vertical = True
                cell_mask = np.zeros(mask.shape, dtype=np.uint8)
                cnt, shift_xy = rotate_contour(cnt, angle=90)
                cell_image, shift_xy = rotate_image(cell_image, shift_xy, angle=90)
                inverted_cell_mask, shift_xy = rotate_image(inverted_cell_mask, shift_xy, angle=90)
                cv2.drawContours(cell_mask, [cnt], -1, 1, -1)
            else:
                vertical = False
                shift_xy = None

            x, y, w, h = cv2.boundingRect(cnt)
            y1, y2, x1, x2 = y, (y + h), x, (x + w)

            m = 5

            edge = False

            if y1 - 5 > 0:
                y1 = y1 - 5
            else:
                y1 = 0
                edge = True

            if y2 + 5 < cell_mask.shape[0]:
                y2 = y2 + 5
            else:
                y2 = cell_mask.shape[0]
                edge = True

            if x1 - 5 > 0:
                x1 = x1 - 5
            else:
                x1 = 0
                edge = True

            if x2 + 5 < cell_mask.shape[1]:
                x2 = x2 + 5
            else:
                x2 = cell_mask.shape[1]
                edge = True

            h, w = y2 - y1, x2 - x1

            inverted_cell_mask = inverted_cell_mask[y1:y2, x1:x2]
            cell_mask = cell_mask[y1:y2, x1:x2]
            cell_image = cell_image[y1:y2, x1:x2]

            cell_image[inverted_cell_mask == 1] = 0
            cell_image = normalize99(cell_image)

            offset = [y1, x1]
            box = [y1, y2, x1, x2]

            cell_data[i] = dict(cell_image=cell_image, cell_mask=cell_mask, offset=offset, shift_xy=shift_xy, box=box, edge=edge, vertical=vertical, mask_id=mask_id, contour=cnt)

    return cell_data


def resize_line(mesh, mesh_length):
    distances = np.linspace(0, mesh.length, mesh_length)
    mesh = LineString([mesh.interpolate(distance) for distance in distances])

    return mesh


def line_to_array(mesh):
    mesh = np.array([mesh.xy[0][:], mesh.xy[1][:]]).T.reshape(-1, 1, 2)

    return mesh


def euclidian_distance(x1, y1, x2, y2):
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    return distance


def polyarea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def compute_line_metrics(mesh):
    steplength = euclidian_distance(mesh[1:, 0] + mesh[1:, 2], mesh[1:, 1] + mesh[1:, 3], mesh[:-1, 0] + mesh[:-1, 2], mesh[:-1, 1] + mesh[:-1, 3]) / 2

    steparea = []
    for i in range(len(mesh) - 1):
        steparea.append(polyarea([*mesh[i:i + 2, 0], *mesh[i:i + 2, 2][::-1]], [*mesh[i:i + 2, 1], *mesh[i:i + 2, 3][::-1]]))

    steparea = np.array(steparea)

    d = euclidian_distance(mesh[:, 0], mesh[:, 1], mesh[:, 2], mesh[:, 3])
    stepvolume = (d[:-1] * d[1:] + (d[:-1] - d[1:]) ** 2 / 3) * steplength * math.pi / 4

    return steplength, steparea, stepvolume


def get_colicoords_mesh(cell, dat, statistics=False, pixel_size=1):
    offset, vertical, shift_xy = dat["offset"], dat["vertical"], dat["shift_xy"]

    numpoints = 20
    t = np.linspace(cell.coords.xl, cell.coords.xr, num=numpoints)
    a0, a1, a2 = cell.coords.coeff

    x_top = t + cell.coords.r * ((a1 + 2 * a2 * t) / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))
    y_top = a0 + a1 * t + a2 * (t ** 2) - cell.coords.r * (1 / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))

    x_bot = t + - cell.coords.r * ((a1 + 2 * a2 * t) / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))
    y_bot = a0 + a1 * t + a2 * (t ** 2) + cell.coords.r * (1 / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))

    # Left semicirlce
    psi = np.arctan(-cell.coords.p_dx(cell.coords.xl))

    th_l = np.linspace(-0.5 * np.pi + psi, 0.5 * np.pi + psi, num=numpoints)
    cl_dx = cell.coords.r * np.cos(th_l)
    cl_dy = cell.coords.r * np.sin(th_l)

    cl_x = cell.coords.xl - cl_dx
    cl_y = cell.coords.p(cell.coords.xl) + cl_dy

    # Right semicircle
    psi = np.arctan(-cell.coords.p_dx(cell.coords.xr))

    th_r = np.linspace(0.5 * np.pi - psi, -0.5 * np.pi - psi, num=numpoints)
    cr_dx = cell.coords.r * np.cos(th_r)
    cr_dy = cell.coords.r * np.sin(th_r)

    cr_x = cr_dx + cell.coords.xr
    cr_y = cr_dy + cell.coords.p(cell.coords.xr)

    # generate differnet halfs
    x1 = np.concatenate((cl_x[::-1][numpoints // 2:], x_top, cr_x[::-1][:numpoints // 2]))
    y1 = np.concatenate((cl_y[::-1][numpoints // 2:], y_top, cr_y[::-1][:numpoints // 2]))

    mesh1 = np.array([x1, y1])

    x2 = np.concatenate((np.flip(cl_x[::-1][:numpoints // 2 - 1]), x_bot, np.flip(cr_x[::-1][numpoints // 2 - 1:])))
    y2 = np.concatenate((np.flip(cl_y[::-1][:numpoints // 2 - 1]), y_bot, np.flip(cr_y[::-1][numpoints // 2 - 1:])))

    mesh2 = np.array([x2, y2])
    mesh2 = np.fliplr(mesh2)

    length = mesh1.shape[1]

    model = np.hstack([mesh1, mesh2]).T

    model = np.append(model, [model[0]], 0)

    model[:, 0] += offset[1]
    model[:, 1] += offset[0]

    if vertical:
        model = rotate_model(model, shift_xy)

    mesh1 = model[:length + 1, :]
    mesh2 = model[length:, :]

    mesh = np.hstack((mesh1, np.flipud(mesh2))).reshape(-1, 4)

    left_line = mesh[:, :2]
    right_line = mesh[:, 2:]

    mid_line = (left_line + right_line) / 2

    distances, area, volume = compute_line_metrics(mesh)

    # q = np.vstack([left_line, right_line]).reshape(-1,2)
    polygon = Polygon(model)
    polygon = orient(polygon)

    boundingbox = np.asarray(polygon.bounds)

    boundingbox[0:2] = np.floor(boundingbox[0:2])
    boundingbox[2:4] = np.ceil(boundingbox[2:4])
    boundingbox[2:4] = boundingbox[2:4] - boundingbox[0:2]

    dat[
        "statistics"] = dict(colicoords=True, length=cell.length * pixel_size, radius=cell.radius * pixel_size, area=cell.area * pixel_size ** 2, circumference=cell.circumference * pixel_size, aspect_ratio=cell.length / (
                cell.radius * 2))

    dat["mask_id"] = dat["mask_id"]

    dat["oufti"] = dict(mesh=mesh, model=model, boundingbox=boundingbox, distances=distances, area=area, volume=volume)

    dat["refined_cnt"] = model.reshape(-1, 1, 2).astype(int)

    return dat


def colicoords_fit(dat, colicoords_channel='Mask', statistics=False, pixel_size=1):
    config.cfg.IMG_PIXELSIZE = 1000 * pixel_size

    results = {}

    try:
        cell_dat = np.load(dat["cell_images_path"], allow_pickle=True).item()

        dat = {**dat, **cell_dat}

        if dat['edge'] == False:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                cell_image = dat["cell_image"]

                data = Data()
                data.add_data(dat['cell_mask'], 'binary')

                for i in range(len(cell_image)):
                    data.add_data(cell_image[i], 'fluorescence', name=dat["channels"][i])

                cell = Cell(data)

                cell.optimize()

                if colicoords_channel != 'Mask':
                    cell.optimize(colicoords_channel)
                    cell.measure_r(colicoords_channel, mode='mid')

                results = get_colicoords_mesh(cell, dat, statistics, pixel_size)

                channel_list = sorted(list(cell.data.data_dict.keys()))

                results["ldist"] = {}

                for channel in channel_list:
                    if channel != 'binary':
                        ldist = get_cell_ldist(cell, channel)
                        results["ldist"][channel] = ldist

                results["cell"] = cell
                results["colicoords_channel"] = colicoords_channel

                results = {**dat, **results["statistics"]}

        else:
            results["cell"] = None

    except:
        print(traceback.format_exc())
        results["cell"] = None

    return results


def get_cell_ldist(cell, channel):
    nbins = config.cfg.L_DIST_NBINS
    sigma = config.cfg.L_DIST_SIGMA
    sigma_arr = sigma / cell.length

    x_arr, out_arr = cell.l_dist(nbins, data_name=channel, norm_x=True, sigma=sigma_arr)

    max_val = np.max(out_arr)

    if max_val > 0:
        out_arr = out_arr + np.flip(out_arr)

        out_arr -= np.min(out_arr)
        out_arr = out_arr / np.max(out_arr)

    else:
        out_arr = None

    return out_arr


def process_cell_ldist(cell_statistics):
    data = [dat["ldist"] for dat in cell_statistics if dat["cell"] != None]

    channels = np.unique([list(dat.keys()) for dat in data]).tolist()

    ldist_data = {}

    for channel in channels:
        ldist = [dat[channel] for dat in data if dat[channel] is not None]

        ldist = np.stack(ldist)

        ldist_mean = np.nanmean(ldist, axis=0)
        ldist_std = np.std(ldist, axis=0)

        channel_dat = {channel + " mean": ldist_mean, channel + " std": ldist_std}

        ldist_data = {**ldist_data, **channel_dat}

    return ldist_data


def run_colicoords(self, cell_data, colicoords_channel, statistics=False, pixel_size=1, progress_callback=None, multithreaded=True):
    from sys import getsizeof

    processes = multiprocessing.cpu_count() - 1

    free_memory = psutil.virtual_memory().available / 1e6
    cell_data_size = getsizeof(cell_data)

    # assumes spawned python processes use 500mb of RAM
    if processes > free_memory // 500:
        processes = int(free_memory // 500)

    if multithreaded:
        with Pool(processes=processes) as pool:
            iter = []

            def callback(*args):
                iter.append(1)
                progress = (len(iter) / len(cell_data)) * 100

                if progress_callback != None:
                    progress_callback.emit(progress)

                return

            results = [pool.apply_async(colicoords_fit, args=(i,), kwds={'colicoords_channel': colicoords_channel, 'statistics': statistics, 'pixel_size': pixel_size}, callback=callback) for i in
                       cell_data]

            try:
                results[-1].get()
            except:
                print(traceback.format_exc())
            else:
                cell_statistics = [r.get() for r in results]
                pool.close()
                pool.join()

    else:
        iter = []
        cell_statistics = []

        for i in range(len(cell_data)):
            dat = cell_data[i]

            stats = colicoords_fit(dat, colicoords_channel=colicoords_channel, statistics=statistics, pixel_size=pixel_size)

            cell_statistics.append(stats)

            iter.append(1)
            progress = (len(iter) / len(cell_data)) * 100

            if progress_callback != None:
                progress_callback.emit(progress)

    ldist_data = {}

    if statistics is True:
        ldist_data = process_cell_ldist(cell_statistics)

    colicoords_data = dict(cell_statistics=cell_statistics, ldist_data=ldist_data)

    return colicoords_data


def get_l_dist(cell_list, channel, force_symmetry=True):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        nbins = config.cfg.L_DIST_NBINS
        sigma = config.cfg.L_DIST_SIGMA
        sigma_arr = sigma / cell_list.length

        x_arr, out_arr = cell_list.l_dist(nbins, data_name=channel, norm_x=True, sigma=sigma_arr)

        x = x_arr[0]

        maxes = np.max(out_arr, axis=1)
        bools = maxes != 0
        n = np.sum(~bools)

        out_arr = out_arr[bools]

        if force_symmetry == True:
            out_arr = np.array([array + np.flip(array) for array in out_arr])

        a_max = np.max(out_arr, axis=1)
        out_arr = out_arr / a_max[:, np.newaxis]

        ldist_mean = np.nanmean(out_arr, axis=0)
        ldist_std = np.std(out_arr, axis=0)

    return ldist_mean, ldist_std


def process_colicoords(self, colicoords_data):
    colicoords_data = colicoords_data["cell_statistics"]

    current_fov = self.viewer.dims.current_step[0]
    colicoords_channel = self.cellpose_segchannel.currentText()

    image_stack = self.viewer.layers[colicoords_channel].data
    label_stack = self.classLayer.data
    mask_stack = self.segLayer.data

    image = image_stack[current_fov, :, :].copy()
    mask = mask_stack[current_fov, :, :].copy()
    label = label_stack[current_fov, :, :].copy()

    for i in range(len(colicoords_data)):
        dat = colicoords_data[i]

        if dat["cell"] != None:
            mask_id = dat["mask_id"]
            cnt = dat["refined_cnt"]

            new_mask = np.zeros_like(mask)
            cv2.drawContours(new_mask, [cnt], -1, 1, -1)

            current_label = np.unique(label[mask == mask_id])

            label[mask == mask_id] = 0
            mask[mask == mask_id] = 0

            mask[new_mask == 1] = mask_id
            label[new_mask == 1] = current_label

            mask_stack[current_fov, :, :] = mask
            label_stack[current_fov, :, :] = label

            self.segLayer.data = mask_stack
            self.classLayer.data = label_stack
