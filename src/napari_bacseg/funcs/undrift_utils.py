import traceback
from functools import partial
import cv2
import numpy as np
from napari_bacseg.funcs.threading_utils import Worker

class _undrift_utils:

    def _undrift_images(self):
        worker = Worker(self._undrift)
        worker.signals.progress.connect(partial(self._Progresbar, progressbar="undrift"))
        worker.signals.finished.connect(self._undrift_postprocesing)
        self.threadpool.start(worker)

    @staticmethod
    def _undrift_preprocesing(img):
        from skimage import exposure

        if np.max(img) > 0:
            img = img.copy()
            v_min, v_max = np.percentile(img[img != 0], (0.1, 99.9))
            img = exposure.rescale_intensity(img, in_range=(v_min, v_max))

        img = cv2.GaussianBlur(img, (5, 5), 0)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return img

    def _undrift_postprocesing(self):
        try:
            # remove borders
            undrift_channel = self.undrift_channel.currentText()

            boundary_image = np.min(self.viewer.layers[undrift_channel].data.copy(), axis=0)
            boundary, _ = self._find_shifted_boundary(boundary_image)

            layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["center_lines"]]

            for layer in layer_names:
                self.viewer.layers[layer].data = self.viewer.layers[layer].data[:, boundary[1]: boundary[3], boundary[0]: boundary[2]]
                image_shape = self.viewer.layers[layer].data.shape

                for i in range(image_shape[0]):
                    if layer not in ["center_lines"]:
                        try:
                            self.viewer.layers[layer].metadata[i]["dims"] = [image_shape[-1], image_shape[-2], ]
                            self.viewer.layers[layer].metadata[i]["crop"] = [0, image_shape[-2], 0, image_shape[-1], ]
                        except:
                            pass

            # refresh active layer
            self.viewer.layers[self.undrift_channel.currentText()].refresh()

            # reset viewer
            self.viewer.reset_view()

        except:
            print(traceback.format_exc())

    def _find_shifted_boundary(self, image):
        x0, y0, x1, y1 = 0, 0, image.shape[1], image.shape[0]

        # Find non-black pixels
        coords = np.column_stack(np.where(image > 0))

        if coords.size != 0:
            y_centre_slice = image[:, int(image.shape[1] / 2)]
            x_centre_slice = image[int(image.shape[0] / 2), :]

            y0 = np.where(y_centre_slice > 0)[0][0]
            y1 = np.where(y_centre_slice > 0)[0][-1]
            x0 = np.where(x_centre_slice > 0)[0][0]
            x1 = np.where(x_centre_slice > 0)[0][-1]

            if y1 - y0 > 10 and x1 - x0 > 10:
                if y0 < 0:
                    y0 = 0
                if y1 > image.shape[0]:
                    y1 = image.shape[0]
                if x0 < 0:
                    x0 = 0
                if x1 > image.shape[1]:
                    x1 = image.shape[1]

        image_shape = y1 - y0, x1 - x0

        return [x0, y0, x1, y1], image_shape

    def _undrift(self, progress_callback):
        try:
            import scipy
            from skimage.registration import phase_cross_correlation

            layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Nucleoid", "Classes", "center_lines", "Localisations", ]]

            if layer_names != []:
                shift_list = []

                undrift_channel = self.undrift_channel.currentText()

                image_shape = self.viewer.layers[undrift_channel].data.shape

                if len(image_shape) == 3:
                    if image_shape[0] > 1:
                        anchor_binary = self._undrift_preprocesing(self.viewer.layers[undrift_channel].data[0])

                        for i in range(image_shape[0] - 1):
                            progress = int((i + 1) / (image_shape[0] - 1) * 100)
                            progress_callback.emit(progress)

                            target_image = self.viewer.layers[undrift_channel].data[i + 1]
                            target_binary = self._undrift_preprocesing(target_image)

                            shift, error, diffphase = phase_cross_correlation(anchor_binary, target_binary, upsample_factor=100, )

                            shift_list.append(shift)

                        for layer in layer_names:
                            for i in range(image_shape[0] - 1):
                                shifted_image = scipy.ndimage.shift(self.viewer.layers[layer].data[i + 1], shift_list[i], cval=-1, )

                                self.viewer.layers[layer].data[i + 1] = shifted_image

                        # boundary_image = np.min(self.viewer.layers[undrift_channel].data.copy(), axis=0)  # boundary, _ = self._find_shifted_boundary(boundary_image)  #  # layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["center_lines"]]  #  # for layer in layer_names:  #     self.viewer.layers[layer].data = self.viewer.layers[layer].data[:, boundary[1]:boundary[3], boundary[0]:boundary[2]]  #     frame = self.viewer.layers[layer].data[0].copy()  #  #     for i in range(image_shape[0] - 1):  #         if layer not in ["Segmentations", "Nucleoid", "Classes"]:  #             self.viewer.layers[layer].metadata[i]["dims"] = [frame.shape[-1], frame.shape[-2]]  #             self.viewer.layers[layer].metadata[i]["crop"] = [0, frame.shape[-2], 0, frame.shape[-1]]

                return

        except:
            print(traceback.format_exc())