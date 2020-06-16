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

import numpy as np
from contextlib import contextmanager
from datetime import timedelta
from time import monotonic
from tflite_runtime.interpreter import Interpreter

class InferenceTimer:
    def __init__(self):
        self.time = 0
        self.int_time = 0

    @contextmanager
    def timeit(self, message: str = None):
        begin = monotonic()
        try:
            yield
        finally:
            end = monotonic()
            self.convert(end-begin)
            print("{0}: {1}".format(message, self.time))

    def convert(self, elapsed):
        self.int_time = elapsed
        self.time = str(timedelta(seconds=elapsed))

class TFLiteInterpreter:
    def __init__(self, model=None):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.inference_time = None
        self.time_to_print = []

        if model is not None:
            self.interpreter = Interpreter(model)
            self.interpreter.allocate_tensors()

            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

    def dtype(self):
        return self.input_details[0]['dtype']

    def height(self):
        return self.input_details[0]['shape'][1]

    def width(self):
        return self.input_details[0]['shape'][2]

    def get_tensor(self, index, squeeze=False):
        if squeeze:
            return np.squeeze(self.interpreter.get_tensor(
                                   self.output_details[index]['index']))

        return self.interpreter.get_tensor(
                    self.output_details[index]['index'])

    def set_tensor(self, image):
        self.interpreter.set_tensor(self.input_details[0]['index'], image)

    def get_time_average(self):
        return self.time_to_print

    def run_inference(self):
        timer = InferenceTimer()
        with timer.timeit("Inference time"):
            self.interpreter.invoke()
        self.inference_time = timer.time
        self.time_to_print.append(timer.int_time)
