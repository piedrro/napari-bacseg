import numpy as np
from tiler import Merger
import traceback


class _tiler_utils:

    def unfold_images(self):

        try:

            if self.unfolded == False:
                current_step = list(self.viewer.dims.current_step)
                current_step = [0] * len(current_step)
                self.viewer.dims.current_step = tuple(current_step)

                from tiler import Tiler

                layer_names = [layer.name for layer in self.viewer.layers]

                tile_size = int(self.unfold_tile_size.currentText())
                tile_shape = (tile_size, tile_size)
                overlap = int(self.unfold_tile_overlap.currentText())

                for layer in layer_names:

                    image = self.viewer.layers[layer].data.copy()
                    metadata_stack = self.viewer.layers[layer].metadata.copy()

                    self.tiler_object = Tiler(data_shape=image[0].shape, tile_shape=tile_shape, overlap=overlap, )

                    if self.unfold_mode.currentIndex() == 0:
                        tiled_image = []

                        for i in range(image.shape[0]):
                            tiles = []

                            for tile_id, tile in self.tiler_object.iterate(image[i]):
                                tiles.append(tile)

                            tiles = np.stack(tiles)
                            tiled_image.append(tiles)

                        image = np.stack(tiled_image)

                        self.viewer.layers[layer].data = image
                        self.viewer.layers[layer].ndisplay = 3
                        self.viewer.reset_view()

                        self.unfolded = True
                        self._autoContrast()

                    if self.unfold_mode.currentIndex() == 1:

                        tiled_images = []
                        tiled_metadata = {}

                        for i in range(image.shape[0]):
                            num_image_tiles = 0

                            for tile_id, tile in self.tiler_object.iterate(image[i]):

                                bbox = np.array(self.tiler_object.get_tile_bbox(tile_id=tile_id))
                                bbox = bbox[..., [-2, -1]]
                                y1, x1, y2, x2 = (bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1],)

                                if y2 > image.shape[-2]:
                                    y2 = image.shape[-2]
                                if x2 > image.shape[-1]:
                                    x2 = image.shape[-1]

                                x2 = x2 - x1
                                x1 = 0
                                y2 = y2 - y1
                                y1 = 0

                                if (y2 - y1, x2 - x1) == tile_shape:
                                    num_image_tiles += 1
                                    tiled_images.append(tile)

                                    if layer != "Classes":
                                        meta = dict(metadata_stack[i])
                                        image_name = meta["image_name"]

                                        tile_meta = dict(meta)
                                        tile_name = (str(image_name).split(".")[0] + "_tile" + str(num_image_tiles) + ".tif")
                                        tile_meta["akseg_hash"] = self.get_hash(img=tile)
                                        tile_meta["image_name"] = tile_name
                                        tile_meta["dims"] = [tile.shape[-1], tile.shape[-2], ]
                                        tile_meta["crop"] = [int(y1), int(y2), int(x1), int(x2), ]

                                        tiled_metadata[len(tiled_images) - 1] = tile_meta

                        image = np.stack(tiled_images)
                        self.viewer.layers[layer].data = image
                        self.viewer.layers[layer].ndisplay = 2
                        self.viewer.reset_view()
                        self.unfolded = False
                        self._autoContrast()

                        if layer != "Classes":
                            self.viewer.layers[layer].metadata = tiled_metadata

            self._updateFileName()

        except:
            print(traceback.format_exc())


    def fold_images(self):

        if self.unfolded == True:
            current_step = list(self.viewer.dims.current_step)
            current_step = [0] * len(current_step)
            self.viewer.dims.current_step = tuple(current_step)

            layer_names = [layer.name for layer in self.viewer.layers]

            for layer in layer_names:
                image = self.viewer.layers[layer].data.copy()

                merger = Merger(self.tiler_object)

                merged_image = []

                for i in range(image.shape[0]):
                    merger.reset()

                    for j in range(image.shape[1]):
                        img = image[i, j].copy()

                        merger.add(j, img.data)

                    merged = merger.merge(dtype=img.dtype)
                    merged_image.append(merged)

                image = np.stack(merged_image)

                self.viewer.layers[layer].data = image
                self.viewer.layers[layer].ndisplay = 2
                self.viewer.reset_view()
                self.unfolded = False
                self._autoContrast()


    def update_image_folds(self, mask_ids=None, image_index=None):

        if self.unfolded == True:
            from tiler import Merger

            layer_names = ["Segmentations", "Nucleoid", "Classes"]

            if image_index is not None:
                target_img_id = image_index
            else:
                target_img_id = self.viewer.dims.current_step[0]

            target_tile_id = self.viewer.dims.current_step[1]

            for layer in layer_names:
                image = self.viewer.layers[layer].data.copy()

                frame = image[target_img_id]

                merger = Merger(self.tiler_object)

                merger.reset()

                overwrite_tile_box = []
                overwrite_tile_img = []

                for j in range(frame.shape[0]):
                    img = frame[j].copy()

                    if j == target_tile_id:
                        overwrite_tile_box = np.array(self.tiler_object.get_tile_bbox(target_tile_id))
                        overwrite_tile_box = overwrite_tile_box[..., [-2, -1]]
                        overwrite_tile_img = img

                    merger.add(j, img.data)

                y1, x1, y2, x2 = (overwrite_tile_box[0][0], overwrite_tile_box[0][1], overwrite_tile_box[1][0], overwrite_tile_box[1][1],)

                frame = merger.merge(dtype=img.dtype)

                if y1 < 0:
                    y1 = 0
                if y2 > frame.shape[0]:
                    y2 = frame.shape[0]
                if x1 < 0:
                    x1 = 0
                if x2 > frame.shape[1]:
                    x2 = frame.shape[1]

                frame[y1:y2, x1:x2] = overwrite_tile_img[: y2 - y1, : x2 - x1]

                if mask_ids is not None:
                    for mask_id in mask_ids:
                        frame[frame == mask_id] = 0

                tiles = []

                for tile_id, tile in self.tiler_object.iterate(frame):
                    tiles.append(tile)

                tiles = np.stack(tiles)

                image[target_img_id] = tiles
                self.viewer.layers[layer].data = image
