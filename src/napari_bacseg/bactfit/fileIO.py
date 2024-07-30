import h5py
import numpy as np
from shapely.geometry import Polygon, LineString
from napari_bacseg.bactfit.cell import Cell, CellList
import pickle
import os
import traceback

def save(file_path, cell_data):

    try:

        file_path = os.path.abspath(file_path)
        file_path = os.path.normpath(file_path)
        file_dir = os.path.dirname(file_path)

        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        file, ext = os.path.splitext(file_path)
        file_path = file + ".h5"

        if isinstance(cell_data, Cell):

            with h5py.File(file_path, 'w') as f:
                if hasattr(cell_data, 'cell_index'):
                    save_name = f"cell_{cell_data.cell_index}"
                else:
                    save_name = "cell_0"

                cell_group = f.create_group(save_name)
                write_cell(cell_group, cell_data)

            print(f"Saved cell to {file_path}")

        elif isinstance(cell_data, CellList):

            with h5py.File(file_path, 'w') as f:
                n_cells = len(cell_data.data)

                for cell_index, cell in enumerate(cell_data.data):
                    file_index = str(cell_index).zfill(int(np.ceil(np.log10(n_cells))))
                    save_name = f"cell_{file_index}"
                    cell_group = f.create_group(save_name)
                    write_cell(cell_group, cell)

            print(f"Saved {n_cells} cell(s) to {file_path}")

        else:
            print(f"Invalid data type: type{cell_data}")

    except:
        print(traceback.format_exc())
        print("Error saving cell data")



def write_cell(cell_group, cell):

    attr_grp = cell_group.create_group('attributes')

    attribute_list = ["name","cell_width", "cell_length",
                      "cell_midline", "cell_polygon",
                      "cell_poles", "polynomial_params",
                      "fit_error", "pixel_size", "locs",
                      "cell_index", "vertical",'crop_bounds']

    for attr in attribute_list:
        if hasattr(cell, attr):
            attr_data = cell.__getattribute__(attr)
            if attr_data is not None:
                try:
                    if isinstance(attr_data, (Polygon, LineString)):
                        # Serialize geometric objects
                        attr_data = pickle.dumps(attr_data)
                    elif not isinstance(attr_data, (int, float, str, np.ndarray)):
                        # Serialize other complex objects
                        attr_data = pickle.dumps(attr_data)
                    if isinstance(attr_data, bytes):
                        attr_grp.create_dataset(attr, data=np.void(attr_data))
                    else:
                        attr_grp.create_dataset(attr, data=attr_data)
                except Exception as e:
                    print(f'Could not save attribute {attr}: {e}')

    data_grp = cell_group.create_group('data')
    for channel, image in cell.data.items():
        if type(image) == np.ndarray:
            grp = data_grp.create_group(channel)
            grp.create_dataset(channel, data=image)
            grp.attrs.create('dclass', np.string_(image.dtype))

def load(file_path):

    cell_list = []

    if type(file_path) is str:
        file_path = [file_path]

    for path in file_path:

        with h5py.File(path, 'r') as f:
            cells = [load_cell(f[key]) for key in f.keys()]
            cell_list.extend(cells)

    if len(cell_list) == 1:
        print(f'Loaded cell')
        return cell_list[0]
    else:
        print(f'Loaded {len(cell_list)} cells')
        return CellList(cell_list)

def load_cell(cell_group):

    attr_grp = cell_group['attributes']


    attr_dict = {}

    for attr in list(attr_grp.keys()):
        attr_data = attr_grp[attr][()]

        if isinstance(attr_data, np.void):
            # Deserialize byte data
            attr_data = pickle.loads(bytes(attr_data))
        elif isinstance(attr_data, bytes):
            # Decode bytes to string
            attr_data = attr_data.decode('utf-8')

        attr_dict[attr] = attr_data
        
    cell = Cell(attr_dict)
    
    data_grp = cell_group['data']
    
    data_dict = {}
    
    for channel in list(data_grp.keys()):
        try:
            grp = data_grp[channel]
            data_arr = grp[channel]
            dclass = grp.attrs.get('dclass').decode('UTF-8')
            image = np.array(data_arr, dtype=dclass)
            
            if channel not in data_dict.keys():
                data_dict[channel] = image
            
        except:
            print(traceback.format_exc())
            
    cell.data = data_dict

    return cell




