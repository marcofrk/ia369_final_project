# Improving Machine Learning Experience With GStreamer Techniques On Embedded Devices

<h2 class="pm-node nj-subtitle">An overlay algorithms study for AI/ML purpose</h2>

<p class="pm-node nj-authors">Marco Franchi</p>

In this paper you will learn different algorithms for video overlay and its performance when applied to Object Detection algorithms. For this, this paper will use GStreamer framework to compare OpenCV, SVG, and Cairo overlays, when applied to the most common object detection engines, such as SSD, Neo DLR, and TFlite. By applying these different approaches, the user can check the best video performance related to the CPU usage and displayed frame per second (fps), which for low systems as embedded devices, it is something very important in order to keep the maximum of CPU available to the inference process.

# Cairo vs OpenCV

This is the first try on NextJournal to  setting up and test the GObject libs on his notebook.

For this, we will need a video file and some images file:

[car.mpg][nextjournal#file#b5634dbd-c12b-41e9-88aa-8f5cfc31e976]

[cat-20200526T031116Z-001.zip][nextjournal#file#283daaa9-5b53-4f7b-85ad-01de79f033fe]

This code consist in a video with a image overlay.

```python no-exec id=102cd9d4-777d-4330-b6fb-f7a803e62253
import numpy as np
import cv2
import glob

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
from gi.repository import Gst, GObject, GstBase

from .gst_hacks import map_gst_buffer, get_buffer_size
from .utils import draw_image

GST_OVERLAY_OPENCV = 'gstoverlayopencv'

class GstOverlayOpenCv(GstBase.BaseTransform):

    CHANNELS = 3  # RGB 

    __gstmetadata__ = ("An example plugin of GstOverlayOpenCv",
                       "gst-filter/gst_overlay_opencv.py",
                       "gst.Element draw on image",
                       "Taras at LifeStyleTransfer.com")

    __gsttemplates__ = (Gst.PadTemplate.new("src",
                                            Gst.PadDirection.SRC,
                                            Gst.PadPresence.ALWAYS,
                                            Gst.Caps.from_string("video/x-raw,format=RGB")),
                        Gst.PadTemplate.new("sink",
                                            Gst.PadDirection.SINK,
                                            Gst.PadPresence.ALWAYS,
                                            Gst.Caps.from_string("video/x-raw,format=RGB")))
    
    def __init__(self):
        super(GstOverlayOpenCv, self).__init__()  

        # Overlay could be any of your objects as far as it implements __call__
        # and returns numpy.ndarray
        self.overlay = None

    def do_transform_ip(self, inbuffer):
        """
            Implementation of simple filter.
            All changes affected on Inbuffer
            Read more:
            https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer-libs/html/GstBaseTransform.html
        """

        success, (width, height) = get_buffer_size(self.srcpad.get_current_caps())
        if not success:
            return Gst.FlowReturn.ERROR
       
        with map_gst_buffer(inbuffer, Gst.MapFlags.READ) as mapped:
            frame = np.ndarray((height, width, self.CHANNELS), buffer=mapped, dtype=np.uint8)

        overlay = self.overlay()
        x = width - overlay.shape[1] 
        y = height - overlay.shape[0] 
        draw_image(frame, overlay, x, y)

        return Gst.FlowReturn.OK

def register(plugin):
    type_to_register = GObject.type_register(GstOverlayOpenCv)
    return Gst.Element.register(plugin, GST_OVERLAY_OPENCV, 0, type_to_register)

def register_by_name(plugin_name):
    name = plugin_name
    description = "gst.Element draws on image buffer"
    version = '1.12.4'
    gst_license = 'LGPL'
    source_module = 'gstreamer'
    package = 'gstoverlay'
    origin = 'lifestyletransfer.com'
    if not Gst.Plugin.register_static(Gst.VERSION_MAJOR, Gst.VERSION_MINOR,
                                      name, description,
                                      register, version, gst_license,
                                      source_module, package, origin):
        raise ImportError("Plugin {} not registered".format(plugin_name)) 
    return True

register_by_name(GST_OVERLAY_OPENCV)
```

In order to compare overlays, the following code consist in OpenCV example:

```python no-exec id=78b110c5-3c74-4fd9-8add-e5d8eb9c3da2
import numpy as np
import cv2
import glob

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
from gi.repository import Gst, GObject, GstBase

from .gst_hacks import map_gst_buffer, get_buffer_size
from .utils import draw_image

GST_OVERLAY_OPENCV = 'gstoverlayopencv'


# https://lazka.github.io/pgi-docs/GstBase-1.0/classes/BaseTransform.html
class GstOverlayOpenCv(GstBase.BaseTransform):

    CHANNELS = 3  # RGB 

    __gstmetadata__ = ("An example plugin of GstOverlayOpenCv",
                       "gst-filter/gst_overlay_opencv.py",
                       "gst.Element draw on image",
                       "Taras at LifeStyleTransfer.com")

    __gsttemplates__ = (Gst.PadTemplate.new("src",
                                            Gst.PadDirection.SRC,
                                            Gst.PadPresence.ALWAYS,
                                            Gst.Caps.from_string("video/x-raw,format=RGB")),
                        Gst.PadTemplate.new("sink",
                                            Gst.PadDirection.SINK,
                                            Gst.PadPresence.ALWAYS,
                                            Gst.Caps.from_string("video/x-raw,format=RGB")))
    
    def __init__(self):
        super(GstOverlayOpenCv, self).__init__()  

        # Overlay could be any of your objects as far as it implements __call__
        # and returns numpy.ndarray
        self.overlay = None

    def do_transform_ip(self, inbuffer):
        """
            Implementation of simple filter.
            All changes affected on Inbuffer
            Read more:
            https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer-libs/html/GstBaseTransform.html
        """

        success, (width, height) = get_buffer_size(self.srcpad.get_current_caps())
        if not success:
            # https://lazka.github.io/pgi-docs/Gst-1.0/enums.html#Gst.FlowReturn
            return Gst.FlowReturn.ERROR
       
        with map_gst_buffer(inbuffer, Gst.MapFlags.READ) as mapped:
            frame = np.ndarray((height, width, self.CHANNELS), buffer=mapped, dtype=np.uint8)

        overlay = self.overlay()
        x = width - overlay.shape[1] 
        y = height - overlay.shape[0] 
        draw_image(frame, overlay, x, y)

        return Gst.FlowReturn.OK


def register(plugin):
    # https://lazka.github.io/pgi-docs/#GObject-2.0/functions.html#GObject.type_register
    type_to_register = GObject.type_register(GstOverlayOpenCv)

    # https://lazka.github.io/pgi-docs/#Gst-1.0/classes/Element.html#Gst.Element.register
    return Gst.Element.register(plugin, GST_OVERLAY_OPENCV, 0, type_to_register)       


def register_by_name(plugin_name):
    
    # Parameters explanation
    # https://lazka.github.io/pgi-docs/Gst-1.0/classes/Plugin.html#Gst.Plugin.register_static
    name = plugin_name
    description = "gst.Element draws on image buffer"
    version = '1.12.4'
    gst_license = 'LGPL'
    source_module = 'gstreamer'
    package = 'gstoverlay'
    origin = 'lifestyletransfer.com'
    if not Gst.Plugin.register_static(Gst.VERSION_MAJOR, Gst.VERSION_MINOR,
                                      name, description,
                                      register, version, gst_license,
                                      source_module, package, origin):
        raise ImportError("Plugin {} not registered".format(plugin_name)) 
    return True

register_by_name(GST_OVERLAY_OPENCV)
```

Code for GStreamer pipeline:

```python no-exec id=0b8ebe8c-1315-401c-9367-12469e27a4c3
import logging

import gi
from gi.repository import Gst
gi.require_version('Gst', '1.0')


class GstPipeline(object):
    """
        Base class to initialize any Gstreamer Pipeline from string
    """
    
    def __init__(self, command):
        """
            :param command: gstreamer plugins string
            :type command: str
            :param fps_interval: raise FPS event every N seconds
            :type fps_interval: float (in seconds)
        """

        if not isinstance(command, str):
            raise ValueError("Invalid type. {} != {}".format(type(command), 
                                                             "str"))
        
        super(GstPipeline, self).__init__()

        self._pipeline = None
        self._active = False

        logging.info('%s %s', 'gst-launch-1.0', command)

        """
            Gsteamer Pipeline
            https://gstreamer.freedesktop.org/documentation/application-development/introduction/basics.html
        """
        self._pipeline = Gst.parse_launch(command)

        if not isinstance(self._pipeline, Gst.Pipeline):
            raise ValueError("Invalid type. {} != {}".format(type(self._pipeline), 
                                                             "Gst.Pipeline"))

        """
            Gsteamer Message Bus
            https://gstreamer.freedesktop.org/documentation/application-development/basics/bus.html
        """
        self._bus = self._pipeline.get_bus()  
        self._bus.add_signal_watch()
        self._bus.connect("message", self._bus_call, None)
    
    @staticmethod
    def create_element(self, name):
        """
            Creates Gstreamer element
            :param name: https://gstreamer.freedesktop.org/documentation/plugins.html
            :type name: str
            :rtype: Gst.Element
        """  
        return Gst.ElementFactory.make(name)

    def get_element(self, name):
        """
            Get Gst.Element from pipeline by name
            :param name:
            :type name: str
            :rtype: Gst.Element
        """  
        element = self._pipeline.get_by_name(name)
        return element is not None, element 

    def start(self):        
        # https://lazka.github.io/pgi-docs/Gst-1.0/enums.html#Gst.StateChangeReturn
        self._pipeline.set_state(Gst.State.PLAYING)

    def stop(self):
        # https://lazka.github.io/pgi-docs/Gst-1.0/enums.html#Gst.StateChangeReturn
        self._pipeline.set_state(Gst.State.NULL)

    def bus(self):
        return self._bus    
    
    def pipeline(self):
        return self._pipeline
    
    def _bus_call(self, bus, message, loop):
        mtype = message.type

        """
            Gstreamer Message Types and how to parse
            https://lazka.github.io/pgi-docs/Gst-1.0/flags.html#Gst.MessageType
        """
        if mtype == Gst.MessageType.EOS:
            self.stop()
            
        elif mtype == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logging.error("{0}: {1}".format(err, debug))      
            self.stop()                  

        elif mtype == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            logging.warning("{0}: {1}".format(err, debug))             
            
        return True 
```

Some additional functions:

```python id=95a49749-d963-4f40-8cde-f85b5454d129
import cairo
import cv2

import os

import datetime
import glob


def list_files(folder, file_format='.jpg'):   
    """
        List files in folder with specific pattern
        :param folder:
        :type folder: str
        :param file_format: .{your format}
        :type file_format: str
        :rtype: list[str] (Absolute paths to files)
    """ 
    pattern = '{0}/*{1}'.format(folder, file_format)
    filenames = glob.glob(pattern)    
    return filenames


def draw_image(source, image, x, y):
    """
        Places "image" with alpha channel on "source" at x, y
        :param source:  
        :type source: numpy.ndarray
        :param image:  
        :type image: numpy.ndarray (with alpha channel)
        :param x: 
        :type x: int
        :param y:  
        :type y: int
        :rtype: numpy.ndarray
    """
    h, w = image.shape[:2]    
    
    max_x, max_y = x + w, y + h      
    alpha = image[:, :, 3] / 255.0
    for c in range(0, 3):
        color = image[:, :, c] * (alpha)
        beta = source[y:max_y, x:max_x, c] * (1.0 - alpha)
        source[y:max_y, x:max_x, c] = color + beta
    return source


def load_image_cv(filename):
    """
        :param filename:  
        :type filename: str
        :rtype: numpy.ndarray
    """
    assert os.path.isfile(filename), "Invalid filename: {}".format(filename)
    return cv2.imread(filename, -1)


def load_image_cairo(filename):
    """
        :param filename:  
        :type filename: str
        :rtype: cairo.Surface
    """
    assert os.path.isfile(filename), "Invalid filename: {}".format(filename)
    return cairo.ImageSurface.create_from_png(filename)
```

```python id=c2e7dfb7-3c62-48de-bb0e-28a2776ce24b
exec(open("gstpipeline.py").read())
import os
from .utils import list_files, load_image_cairo, load_image_cv

class Animation(object):

    def __init__(self, images, keyframe=5):
        """
            :param images: contains images
            :type images: list
            :param images: makes animation slower[0]/faster[maxint]
            :type images: keyframe [0, maxint]
        """
        assert len(images), "Invalid data. Empty Images: {}".format(len(images))
        self._images = images
        self._image_id, self._frame_id = 0, 0

        self._keyframe = keyframe
    
    def __call__(self):
        """
            Returns next image when call:
                a = Animation(items)
                image = a()
            :rtype: image_type (numpy, cairo.surface, etc.)
        """

        # Keyframe makes animation faster/slower
        self._frame_id += 1
        if self._frame_id >= self._keyframe:
            self._frame_id = 0
            self._image_id += 1

        if self._image_id >= len(self._images):
            self._image_id = 0

        return self._images[self._image_id]


def create_animation_from_folder_cairo(folder, file_format=".png"):
    assert os.path.isdir(folder), "Invalid folder: {}".format(folder)
    return Animation([load_image_cairo(fl) for fl in list_files(folder, file_format=file_format)])


def create_animation_from_folder_cv(folder, file_format=".png"):
    assert os.path.isdir(folder), "Invalid folder: {}".format(folder)
    return Animation([load_image_cv(fl) for fl in list_files(folder, file_format=file_format)])
```

# Results

The idea is be able to support gi repository at this notebook and be able to see the Cairo working about 33% better than OpenCV:

![results.png][nextjournal#file#2a6501bd-68da-4cbc-8823-a4042230a8e0]


[nextjournal#file#b5634dbd-c12b-41e9-88aa-8f5cfc31e976]:
<https://nextjournal.com/data/QmVtxRM9gcYFPKDugCkoP1NdX95ZWeeernankw1PjCfuha?filename=car.mpg&content-type=video/mpeg>

[nextjournal#file#283daaa9-5b53-4f7b-85ad-01de79f033fe]:
<https://nextjournal.com/data/QmSMGuTG18rkL84tY9hpQyr2kBfBXoLDfbt6gLQCrQ6sDw?filename=cat-20200526T031116Z-001.zip&content-type=application/zip>

[nextjournal#file#2a6501bd-68da-4cbc-8823-a4042230a8e0]:
<https://nextjournal.com/data/Qmf9ESayJfoD7khnb6Xb4xznrxQAxfqiP8DYh9EcosMi2S?filename=results.png&content-type=image/png> (Cairo vs OpenCV Overlay Resolution x FPS)

<details id="com.nextjournal.article">
<summary>This notebook was exported from <a href="https://nextjournal.com/a/MaVhFnWhWpvH4aYVoiGS9?change-id=Chyf3dSydyMSD7CsAMPGZX">https://nextjournal.com/a/MaVhFnWhWpvH4aYVoiGS9?change-id=Chyf3dSydyMSD7CsAMPGZX</a></summary>

```edn nextjournal-metadata
{:article
 {:settings
  {:use-gpu? true, :image "nextjournal/ubuntu:17.04-658650854"},
  :nodes
  {"0b8ebe8c-1315-401c-9367-12469e27a4c3"
   {:id "0b8ebe8c-1315-401c-9367-12469e27a4c3",
    :kind "code-listing",
    :name "gstpipeline.py"},
   "102cd9d4-777d-4330-b6fb-f7a803e62253"
   {:id "102cd9d4-777d-4330-b6fb-f7a803e62253",
    :kind "code-listing",
    :name "gst_overlay_cairo.py"},
   "283daaa9-5b53-4f7b-85ad-01de79f033fe"
   {:id "283daaa9-5b53-4f7b-85ad-01de79f033fe", :kind "file"},
   "2a6501bd-68da-4cbc-8823-a4042230a8e0"
   {:id "2a6501bd-68da-4cbc-8823-a4042230a8e0", :kind "file"},
   "5af2cdfa-640a-4367-b08f-d13baf65eae5"
   {:environment
    [:environment
     {:article/nextjournal.id
      #uuid "5b45e08b-5b96-413e-84ed-f03b5b65bd66",
      :change/nextjournal.id
      #uuid "5df5e18c-0be4-4d8d-b099-6ce55ca12cf4",
      :node/id "0149f12a-08de-4f3d-9fd3-4b7a665e8624"}],
    :id "5af2cdfa-640a-4367-b08f-d13baf65eae5",
    :kind "runtime",
    :language "python",
    :type :nextjournal},
   "78b110c5-3c74-4fd9-8add-e5d8eb9c3da2"
   {:id "78b110c5-3c74-4fd9-8add-e5d8eb9c3da2",
    :kind "code-listing",
    :name "gst_overlay_opencv.py"},
   "95a49749-d963-4f40-8cde-f85b5454d129"
   {:id "95a49749-d963-4f40-8cde-f85b5454d129",
    :kind "code",
    :name "utils.py",
    :runtime [:runtime "5af2cdfa-640a-4367-b08f-d13baf65eae5"]},
   "b5634dbd-c12b-41e9-88aa-8f5cfc31e976"
   {:id "b5634dbd-c12b-41e9-88aa-8f5cfc31e976", :kind "file"},
   "c2e7dfb7-3c62-48de-bb0e-28a2776ce24b"
   {:compute-ref #uuid "376d7edc-9f75-4031-840e-ad598a934d7f",
    :exec-duration 171,
    :id "c2e7dfb7-3c62-48de-bb0e-28a2776ce24b",
    :kind "code",
    :name "animation.py",
    :output-log-lines {},
    :runtime [:runtime "5af2cdfa-640a-4367-b08f-d13baf65eae5"]}},
  :nextjournal/id #uuid "02df846b-0020-4e78-9a74-ac5b5d1accd6",
  :article/change
  {:nextjournal/id #uuid "5ecd3108-3cc1-497d-a038-9e26ceb9a1ca"}}}

```
</details>
