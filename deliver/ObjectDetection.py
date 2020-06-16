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
gi.require_version('Gtk', '3.0')
from gi.repository import GLib, GObject, Gst, Gtk

try:
    import svgwrite
    has_svgwrite = True
except ImportError:
    has_svgwrite = False

import threading
import collections
import os
import re
import sys
import time
import cv2
import numpy as np
from PIL import Image

from inference import TFLiteInterpreter

GObject.threads_init()
Gst.init(None)

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

    def list_average(self):
        return round(sum(self.interpreter.get_time_average()) / len(self.interpreter.get_time_average()), 3)

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
        print("Inf average: {0}".format(self.list_average()))

class ObjectsDetectionV4L2:
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

    def list_average(self):
        return round(sum(self.interpreter.get_time_average()) / len(self.interpreter.get_time_average()), 3)

    def v4l2_video_pipeline(self):
        leaky=" max-size-buffers=1"
        sync="""sync=false drop=True max-buffers=1
             emit-signals=True max-lateness=8000000000"""

        return (("""filesrc location={} ! qtdemux name=demux  demux.video_0
                    ! queue ! decodebin ! queue {} ! videoconvert ! video/x-raw,format=BGR !
                    appsink {}""").format(self.video, leaky, sync))

    def run(self):
        self.start()

        video = cv2.VideoCapture(self.v4l2_video_pipeline())
        print(video)
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
        print("Inf average: {0}".format(self.list_average()))

class GstPipeline:
    def __init__(self, pipeline, user_function, src_size):
        self.user_function = user_function
        self.running = False
        self.gstbuffer = None
        self.sink_size = None
        self.src_size = src_size
        self.box = None
        self.condition = threading.Condition()

        self.pipeline = Gst.parse_launch(pipeline)
        self.overlay = self.pipeline.get_by_name('overlay')
        self.overlaysink = self.pipeline.get_by_name('overlaysink')
        appsink = self.pipeline.get_by_name('appsink')
        appsink.connect('new-sample', self.on_new_sample)

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message', self.on_bus_message)

    def run(self):
        self.running = True
        worker = threading.Thread(target=self.inference_loop)
        worker.start()

        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            Gtk.main()
        except BaseException:
            pass

        self.pipeline.set_state(Gst.State.NULL)
        while GLib.MainContext.default().iteration(False):
            pass
        with self.condition:
            self.running = False
            self.condition.notify_all()
        worker.join()

    def on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            Gtk.main_quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            sys.stderr.write('Warning: %s: %s\n' % (err, debug))
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            sys.stderr.write('Error: %s: %s\n' % (err, debug))
            Gtk.main_quit()
        return True

    def on_new_sample(self, sink):
        sample = sink.emit('pull-sample')
        if not self.sink_size:
            s = sample.get_caps().get_structure(0)
            self.sink_size = (s.get_value('width'), s.get_value('height'))
        with self.condition:
            self.gstbuffer = sample.get_buffer()
            self.condition.notify_all()
        return Gst.FlowReturn.OK

    def get_box(self):
        if not self.box:
            glbox = self.pipeline.get_by_name('glbox')
            if glbox:
                glbox = glbox.get_by_name('filter')
            box = self.pipeline.get_by_name('box')
            assert glbox or box
            assert self.sink_size
            if glbox:
                self.box = (glbox.get_property('x'), glbox.get_property('y'),
                            glbox.get_property('width'), glbox.get_property('height'))
            else:
                self.box = (-box.get_property('left'), -box.get_property('top'),
                            self.sink_size[0] + box.get_property(
                                'left') + box.get_property('right'),
                            self.sink_size[1] + box.get_property('top') + box.get_property('bottom'))
        return self.box

    def inference_loop(self):
        while True:
            with self.condition:
                while not self.gstbuffer and self.running:
                    self.condition.wait()
                if not self.running:
                    break
                gstbuffer = self.gstbuffer
                self.gstbuffer = None

            input_tensor = gstbuffer
            svg = self.user_function(
                input_tensor, self.src_size, self.get_box())
            if svg:
                if self.overlay:
                    self.overlay.set_property('data', svg)
                if self.overlaysink:
                    self.overlaysink.set_property('svg', svg)

def run_pipeline(user_function,
                 src_size,
                 appsink_size,
                 videosrc='None'):
    scale = min(
        appsink_size[0] /
        src_size[0],
        appsink_size[1] /
        src_size[1])
    scale = tuple(int(x * scale) for x in src_size)
    scale_caps = 'video/x-raw,width={width},height={height}'.format(
        width=scale[0], height=scale[1])

    PIPELINE = """filesrc location=%s ! qtdemux name=demux  demux.video_0
                    ! queue ! decodebin  ! videorate
                    ! videoconvert n-threads=4 ! videoscale n-threads=4
                    ! {src_caps} ! {leaky_q} """ % (videosrc)

    PIPELINE += """ ! tee name=t
            t. ! {leaky_q} ! videoconvert ! videoscale ! {scale_caps} ! videobox name=box autocrop=true
               ! {sink_caps} ! {sink_element}
            t. ! queue ! videoconvert
               ! rsvgoverlay name=overlay ! videoconvert ! ximagesink sync=false
            """

    SINK_ELEMENT = 'appsink name=appsink emit-signals=true max-buffers=1 drop=true'
    SINK_CAPS = 'video/x-raw,format=RGB,width={width},height={height}'
    LEAKY_Q = 'queue max-size-buffers=1 leaky=downstream'

    SRC_CAPS = 'video/x-raw,width={width},height={height},framerate=30/1'
    SINK_ELEMENT = 'appsink name=appsink emit-signals=true max-buffers=1 drop=true sync=false'
    SINK_CAPS = 'video/x-raw,format=RGB,width={width},height={height}'
    LEAKY_Q = 'queue max-size-buffers=1 leaky=downstream'

    src_caps = SRC_CAPS.format(width=src_size[0], height=src_size[1])
    sink_caps = SINK_CAPS.format(width=appsink_size[0], height=appsink_size[1])
    pipeline = PIPELINE.format(leaky_q=LEAKY_Q,
                               src_caps=src_caps, sink_caps=sink_caps,
                               sink_element=SINK_ELEMENT, scale_caps=scale_caps)

    print('Gstreamer pipeline:\n', pipeline)

    pipeline = GstPipeline(pipeline, user_function, src_size)
    pipeline.run()

class ObjectsDetectionGStreamer:
    def __init__(self):
        self.videosrc = "../data/video/Home-And-Away.mp4"
        self.model = "../data/model/mobilenet_ssd_v2_coco_quant_postprocess.tflite"
        self.label = "../data/model/coco_labels.txt"

        self.interpreter = None
        self.tensor = None

        self.src_width = 640
        self.src_height = 480

        self.inf_time = []

    def video_config(self):
        if self.args.video_src and self.args.video_src.startswith("/dev/video"):
            self.videosrc = self.args.video_src
        elif self.args.video_src and os.path.exists(self.args.video_src):
            self.videofile = self.args.video_src
            self.src_width = 1920
            self.src_height = 1080

    def input_image_size(self):
        return self.interpreter.input_details[0]['shape'][1:]

    def input_tensor(self):
        return self.tensor(self.interpreter.input_details[0]['index'])()[0]

    def set_input(self, buf):
        result, mapinfo = buf.map(Gst.MapFlags.READ)
        if result:
            np_buffer = np.reshape(np.frombuffer(mapinfo.data, dtype=np.uint8),
                                   self.input_image_size())
            self.input_tensor()[:, :] = np_buffer
            buf.unmap(mapinfo)

    def output_tensor(self, i):
        output_data = np.squeeze(self.tensor(
                                 self.interpreter.output_details[i]['index'])())
        if 'quantization' not in self.interpreter.output_details:
            return output_data
        scale, zero_point = self.interpreter.output_details['quantization']
        if scale == 0:
            return output_data - zero_point
        return scale * (output_data - zero_point)

    def avg_fps_counter(self, window_size):
        window = collections.deque(maxlen=window_size)
        prev = time.monotonic()
        yield 0.0

        while True:
            curr = time.monotonic()
            window.append(curr - prev)
            prev = curr
            yield len(window) / sum(window)

    def load_labels(self, path):
        p = re.compile(r'\s*(\d+)(.+)')
        with open(path, 'r', encoding='utf-8') as f:
            lines = (p.match(line).groups() for line in f.readlines())
            return {int(num): text.strip() for num, text in lines}

    def shadow_text(self, dwg, x, y, text, font_size=20):
        dwg.add(dwg.text(text, insert=(x + 1, y + 1),
                         fill='black', font_size=font_size))
        dwg.add(dwg.text(text, insert=(x, y),
                fill='white', font_size=font_size))

    def generate_svg(self, src_size, inference_size,
                     inference_box, objs, labels, text_lines):
        dwg = svgwrite.Drawing('', size=src_size)
        src_w, src_h = src_size
        inf_w, inf_h = inference_size
        box_x, box_y, box_w, box_h = inference_box
        scale_x, scale_y = src_w / box_w, src_h / box_h

        for y, line in enumerate(text_lines, start=1):
            self.shadow_text(dwg, 10, y * 20, line)
        for obj in objs:
            x0, y0, x1, y1 = list(obj.bbox)
            x, y, w, h = x0, y0, x1 - x0, y1 - y0
            x, y, w, h = int(x * inf_w), int(y * inf_h), \
                         int(w * inf_w), int(h * inf_h)
            x, y = x - box_x, y - box_y
            x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
            percent = int(100 * obj.score)
            label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
            self.shadow_text(dwg, x, y - 5, label)
            dwg.add(dwg.rect(insert=(x, y), size=(w, h), fill='none',
                             stroke='red', stroke_width='2'))
        return dwg.tostring()

    def get_output(self, score_threshold=0.1, top_k=3, image_scale=1.0):
        boxes = self.output_tensor(0)
        category = self.output_tensor(1)
        scores = self.output_tensor(2)
        return [make_boxes(i, boxes, category, scores) \
                for i in range(top_k) if scores[i] >= score_threshold]

    def start(self):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.interpreter = TFLiteInterpreter(self.model)
        self.tensor = self.interpreter.interpreter.tensor

    def run(self):
        self.start()
        labels = self.load_labels(self.label)
        w, h, _ = self.input_image_size()
        inference_size = (w, h)
        fps_counter = self.avg_fps_counter(30)

        def user_callback(input_tensor, src_size, inference_box):
            nonlocal fps_counter
            start_time = time.monotonic()
            self.set_input(input_tensor)
            self.interpreter.run_inference()
            objs = self.get_output()
            end_time = time.monotonic()
            text_lines = ['Inference: {:.2f} ms'.format((end_time-start_time) \
                                                        * 1000),
                          'FPS: {} fps'.format(round(next(fps_counter))),]
            self.inf_time.append(end_time-start_time*1000)
            return self.generate_svg(src_size, inference_size, inference_box,
                                     objs, labels, text_lines)

        result = run_pipeline(user_callback,
                                        src_size=(self.src_width,
                                                  self.src_height),
                                        appsink_size=inference_size,
                                        videosrc=self.videosrc)
