import os
import sys
import numpy as np

from collections import deque
from itertools import cycle

import logging as log

from openvino.runtime import AsyncInferQueue

path = os.path.join(os.getcwd(), 'common\\python\\openvino\\model_zoo')
print(path)
sys.path.append(path)

from model_api.models.utils import resize_image


def crop(frame, roi):
    p1 = roi.position.astype(int)
    p1 = np.clip(p1, [0, 0], [frame.shape[1], frame.shape[0]])
    p2 = (roi.position + roi.size).astype(int)
    p2 = np.clip(p2, [0, 0], [frame.shape[1], frame.shape[0]])
    return frame[p1[1]:p2[1], p1[0]:p2[0]]

def cut_rois(frame, rois):
    return [crop(frame, roi) for roi in rois]

def resize_input(image, target_shape, nchw_layout):
    if nchw_layout:
        _, _, h, w = target_shape
    else:
        _, h, w, _ = target_shape
    resized_image = resize_image(image, (w, h))
    if nchw_layout:
        # HWC->CHW
        resized_image = resized_image.transpose((2, 0, 1)) 
    resized_image = resized_image.reshape(target_shape)
    return resized_image

def preprocess_frame(frame, size, chw_layout=True):
    frame = resize_input(frame, size, chw_layout)
    return frame

class AsyncWrapper:
    def __init__(self, ie_model, num_requests):
        self.model = ie_model
        self.num_requests = num_requests
        self._result_ready = False
        self._req_ids = cycle(range(num_requests))
        self._result_ids = cycle(range(num_requests))
        self._frames = deque(maxlen=num_requests)

    def infer(self, model_input, frame=None):
        next_req_id = next(self._req_ids)
        self.model.async_infer(model_input, next_req_id)

        last_frame = self._frames[0] if self._frames else frame

        self._frames.append(frame)
        if next_req_id == self.num_requests - 1:
            self._result_ready = True

        if self._result_ready:
            result_req_id = next(self._result_ids)
            result = self.model.wait_request(result_req_id)
            return result, last_frame
        else:
            return None, None

class SISO_IEModel:
    '''
    Single input/single output model.
    '''
    def __init__(self, model_path, core, target_device, num_requests, model_type):
        log.info('Reading {} model {}'.format(model_type, model_path))
        self.model = core.read_model(model_path)
        if len(self.model.inputs) != 1:
            log.error("SISO supports only models with 1 input")
            sys.exit(1)

        if len(self.model.outputs) != 1:
            log.error("SISO supports only models with 1 output")
            sys.exit(1)

        self.outputs = {}
        compiled_model = core.compile_model(self.model, target_device)
        self.output_tensor = compiled_model.outputs[0]
        self.input_name = self.model.inputs[0].get_any_name()
        self.input_shape = self.model.inputs[0].shape

        self.num_requests = num_requests
        self.infer_queue = AsyncInferQueue(compiled_model, num_requests)
        self.infer_queue.set_callback(self.completion_callback)
        log.info('The {} model {} is loaded to {}'.format(model_type, model_path, target_device))

    def completion_callback(self, infer_request, id):
        self.outputs[id] = infer_request.results[self.output_tensor]

    def async_infer(self, frame, req_id):
        input_data = {self.input_name: frame}
        self.infer_queue.start_async(input_data, req_id)

    def wait_request(self, req_id):
        self.infer_queue[req_id].wait()
        return self.outputs.pop(req_id, None)

    def cancel(self):
        for ireq in self.infer_queue:
            ireq.cancel()

class SIMO_IEModel:
    '''
    Single input/multiple output model.
    '''
    def __init__(self, model_path, core, target_device, num_requests, model_type):
        log.info('Reading {} model {}'.format(model_type, model_path))
        self.model = core.read_model(model_path)
        if len(self.model.inputs) != 1:
            log.error("SIMO supports only models with 1 input")
            sys.exit(1)

        self.outputs = {}
        compiled_model = core.compile_model(self.model, target_device)
        self.output_count = len(compiled_model.outputs)
        self.output_tensor = compiled_model.outputs[0]
        self.input_name = self.model.inputs[0].get_any_name()
        self.input_shape = self.model.inputs[0].shape

        self.num_requests = num_requests
        self.infer_queue = AsyncInferQueue(compiled_model, num_requests)
        self.infer_queue.set_callback(self.completion_callback)
        log.info('The {} model {} is loaded to {}'.format(model_type, model_path, target_device))

    def completion_callback(self, infer_request, id):
        results = infer_request.results
        aggregated_results = []
        for i in range(self.output_count):
            aggregated_results.append(results[self.output_tensor[i]])
        self.outputs[id] = aggregated_results

    def async_infer(self, frame, req_id):
        input_data = {self.input_name: frame}
        self.infer_queue.start_async(input_data, req_id)

    def wait_request(self, req_id):
        self.infer_queue[req_id].wait()
        return self.outputs.pop(req_id, None)

    def cancel(self):
        for ireq in self.infer_queue:
            ireq.cancel()

class MISO_IEModel:
    '''
    Multiple input/single output model.
    '''
    def __init__(self, model_path, core, target_device, num_requests, model_type):
        log.info('Reading {} model {}'.format(model_type, model_path))
        self.model = core.read_model(model_path)

        if len(self.model.outputs) != 1:
            log.error("MISO supports only models with 1 output")
            sys.exit(1)

        self.outputs = {}
        compiled_model = core.compile_model(self.model, target_device)
        self.output_tensor = compiled_model.outputs[0]
        self.input_count = len(self.model.inputs)
        self.input_names = [self.model.inputs[i].get_any_name() for i in range(self.input_count)]
        self.input_shapes = [self.model.inputs[i].shape for i in range(self.input_count)]

        self.num_requests = num_requests
        self.infer_queue = AsyncInferQueue(compiled_model, num_requests)
        self.infer_queue.set_callback(self.completion_callback)
        log.info('The {} model {} is loaded to {}'.format(model_type, model_path, target_device))

    def completion_callback(self, infer_request, id):
        self.outputs[id] = infer_request.results[self.output_tensor]

    def async_infer(self, inputs, req_id):
        assert self.input_count == len(inputs), 'Provided inputs do not match expected inputs'
        input_data = {}
        for i in range(self.input_count):
            input_data[self.input_names[i]] = inputs[i]
        self.infer_queue.start_async(input_data, req_id)

    def wait_request(self, req_id):
        self.infer_queue[req_id].wait()
        return self.outputs.pop(req_id, None)

    def cancel(self):
        for ireq in self.infer_queue:
            ireq.cancel()