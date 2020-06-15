# Copyright 2020 NXP Semiconductors
# Copyright 2020 Marco Franchi
#
# This file was copied from NXP Semiconductors PyeIQ project respecting its
# rights. All the modified parts below are according to NXP Semiconductors PyeIQ
# project`s LICENSE terms.
#
# Reference: https://source.codeaurora.org/external/imxsupport/pyeiq/
#
# SPDX-License-Identifier: BSD-3-Clause

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import collections
import re
import cv2
import numpy as np
from PIL import Image

from inference import TFLiteInterpreter

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

class BBox(collections.namedtuple(
        'BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal,
    parallel to the x or y axis.
    """
    __slots__ = ()

def make_boxes(i, boxes, class_ids, scores):
    ymin, xmin, ymax, xmax = boxes[i]
    return Object(
        id=int(class_ids[i]),
        score=scores[i],
        bbox=BBox(xmin=np.maximum(0.0, xmin),
                        ymin=np.maximum(0.0, ymin),
                        xmax=np.minimum(1.0, xmax),
                        ymax=np.minimum(1.0, ymax)))

class ObjectsDetectionOpenCV:
    def __init__(self):
        self.video = "../data/video/Home-And-Away.mp4"
        self.model = "../data/model/mobilenet_ssd_v2_coco_quant_postprocess.tflite"
        self.label = "../data/model/coco_labels.txt"

    def set_input(self, image, resample=Image.NEAREST):
        image = image.resize((self.input_image_size()[0:2]), resample)
        self.input_tensor()[:, :] = image

    def input_image_size(self):
        return self.interpreter.input_details[0]['shape'][1:]

    def input_tensor(self):
        return self.tensor(self.interpreter.input_details[0]['index'])()[0]

    def output_tensor(self, i):
        output_data = np.squeeze(self.tensor(
                                 self.interpreter.output_details[i]['index'])())
        if 'quantization' not in self.interpreter.output_details:
            return output_data
        scale, zero_point = self.interpreter.output_details['quantization']
        if scale == 0:
            return output_data - zero_point
        return scale * (output_data - zero_point)

    def load_labels(self, path):
        p = re.compile(r'\s*(\d+)(.+)')
        with open(path, 'r', encoding='utf-8') as f:
            lines = (p.match(line).groups() for line in f.readlines())
            return {int(num): text.strip() for num, text in lines}

    def get_output(self, score_threshold=0.1, top_k=3, image_scale=1.0):
        boxes = self.output_tensor(0)
        class_ids = self.output_tensor(1)
        scores = self.output_tensor(2)
        count = int(self.output_tensor(3))

        return [make_boxes(i, boxes, class_ids, scores) for i in range(top_k) \
                if scores[i] >= score_threshold]

    def append_objs_to_img(self, opencv_im, objs):
        height, width, channels = opencv_im.shape
        for obj in objs:
            x0, y0, x1, y1 = list(obj.bbox)
            x0 = int(x0 * width)
            y0 = int(y0 * height)
            x1 = int(x1 * width)
            y1 = int(y1 * height)

            percent = int(100 * obj.score)
            label = '{}% {}'.format(percent, self.label.get(obj.id, obj.id))

            opencv_im = cv2.rectangle(opencv_im, (x0, y0), (x1, y1),
                                      (0, 255, 0), 2)
            opencv_im = cv2.putText(opencv_im, label, (x0, y0 + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                    (255, 0, 0), 2)
        return opencv_im

    def start(self):
        self.interpreter = TFLiteInterpreter(self.model)
        self.tensor = self.interpreter.interpreter.tensor
        self.label = self.load_labels(self.label)

    def run(self):
        self.start()

        video = cv2.VideoCapture(self.video)
        if (not video) or (not video.isOpened()):
            sys.exit("Your video device could not be found. Exiting...")

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            opencv_im = frame
            opencv_im_rgb = cv2.cvtColor(opencv_im, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(opencv_im_rgb)

            self.set_input(pil_im)
            self.interpreter.run_inference()
            objs = self.get_output()
            opencv_im = self.append_objs_to_img(opencv_im, objs)

            cv2.imshow("OpenCV V4L2", opencv_im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()
