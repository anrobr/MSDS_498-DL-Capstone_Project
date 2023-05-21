import os
import sys
import numpy as np

from math import sin, cos, pi
from .models import resize_input, cut_rois
from numpy_ringbuffer import RingBuffer

common_path = os.path.join(os.getcwd(), 'common/python')
sys.path.append(common_path)

from model_api.models import OutputTransform

class ROI:
    def __init__(self, position : np.array, size : np.array):
        self.position = position
        self.size = size

class Point2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class DTO:
    def __init__(self):
        self.face_result = None
        self.facial_landmarks_result = None
        self.closed_eyes_result = None
        self.head_pose_result = None
        self.gaze_result = None
        self.tracking_result = None

class FaceDetectorResult:
        def __init__(self, output):
            self.image_id = output[0]
            self.label = int(output[1])
            self.confidence = output[2]
            self.roi = ROI(
                # position: (x, y)
                np.array((output[3], output[4])), 
                 # size: (w, h)
                np.array((output[5], output[6])))

        def rescale_roi(self, roi_scale_factor):
            self.roi.position -= self.roi.size * 0.5 * (roi_scale_factor - 1.0)
            self.roi.size *= roi_scale_factor

        def resize_roi(self, frame_width, frame_height):
            self.roi.position[0] *= frame_width
            self.roi.position[1] *= frame_height
            self.roi.size[0] = self.roi.size[0] * frame_width - self.roi.position[0]
            self.roi.size[1] = self.roi.size[1] * frame_height - self.roi.position[1]

        def clip(self, width, height):
            min = [0, 0]
            max = [width, height]
            self.roi.position[:] = np.clip(self.roi.position, min, max)
            self.roi.size[:] = np.clip(self.roi.size, min, max)

        def set_bounding_box(self, frame_shape):
            p = self.roi.position.astype('int')
            s = self.roi.size.astype('int')
            # clipping: bounding box face to min and max frame dimensions
            self.bounding_box = (max(p[0], 0),                          # x_min
                                 max(p[1], 0),                          # y_min
                                 min(p[0] + s[0], frame_shape[1]),      # x_max
                                 min(p[1] + s[1], frame_shape[0]))      # y_max

class FaceDetector:
    def __init__(self, confidence_threshold, roi_scale_factor):
         self.confidence_threshold = confidence_threshold
         self.roi_scale_factor = roi_scale_factor

    def preprocess(self, frame, target_shape, nchw_layout):
        return resize_input(frame, target_shape, nchw_layout)

    def process(self, outputs, frame_shape):
        for output in outputs[0][0]:
            result = FaceDetectorResult(output)
            if result.confidence < self.confidence_threshold:
                # results are sorted by confidence decrease
                break 

            result.resize_roi(frame_shape[1], frame_shape[0])
            result.rescale_roi(self.roi_scale_factor)
            result.clip(frame_shape[1], frame_shape[0])
            result.set_bounding_box(frame_shape)
            # return best match (highest confidence -> driver)
            return result
        # no suitable detection
        return None
     
class FacialLandmarksDetectorResult:
    def __init__(self, 
                 frame_shape, 
                 face_result, 
                 outputs, 
                 eye_bb_shape):
        # for rendering the landmarks
        self.coordinates = [out.reshape((-1, 2)).astype(np.float64) for out in outputs]
        # for rendering eye bounding boxes and for extracting two eye patches for follow-up steps
        output_transform = OutputTransform(frame_shape[:2], None)
        bb_w_h = eye_bb_shape[0] / 2
        bb_h_h = eye_bb_shape[1] / 2

        # face_result.bounding box contains the absolute coordinates of the face bb wrt. the full frame
        left_eye_coordinate = self.coordinates[0][0]
        x = face_result.bounding_box[0] + output_transform.scale(face_result.roi.size[0] * left_eye_coordinate[0])
        y = face_result.bounding_box[1] + output_transform.scale(face_result.roi.size[1] * left_eye_coordinate[1])
        # x_min, y_min, x_max, y_max
        self.left_eye_bb = (int(x - bb_w_h), int(y - bb_h_h), int(x + bb_w_h), int(y + bb_h_h))
        self.left_eye_roi = ROI(np.array([int(x - bb_w_h), int(y - bb_h_h)]), 
                                np.array([int(x + bb_w_h) - int(x - bb_w_h), int(y + bb_h_h) - int(y - bb_h_h)]))

        right_eye_coordinate = self.coordinates[0][1]
        x = face_result.bounding_box[0] + output_transform.scale(face_result.roi.size[0] * right_eye_coordinate[0])
        y = face_result.bounding_box[1] + output_transform.scale(face_result.roi.size[1] * right_eye_coordinate[1])
        # x_min, y_min, x_max, y_max
        self.right_eye_bb = (int(x - bb_w_h), int(y - bb_h_h), int(x + bb_w_h), int(y + bb_h_h))
        self.right_eye_roi = ROI(np.array([int(x - bb_w_h), int(y - bb_h_h)]), 
                                 np.array([int(x + bb_w_h) - int(x - bb_w_h), int(y + bb_h_h) - int(y - bb_h_h)]))

class FacialLandmarksDetector:
    def __init__(self, eye_bb_shape):
        self.eye_bb_shape = eye_bb_shape

    def preprocess(self, frame, rois, target_shape, nchw_layout):
        # rois: coordinates of the face's bounding box
        inputs = cut_rois(frame, rois)
        inputs = [resize_input(input, target_shape, nchw_layout) for input in inputs]
        return inputs
    
    def process(self, frame_shape, face_result, outputs):
        # convert to 5 (x,y)-coordinates
        return FacialLandmarksDetectorResult(frame_shape, face_result, outputs, self.eye_bb_shape)
    
class ClosedEyesDetectorResult:
    def __init__(self, eyes_detectable, left_eye_closed, right_eye_closed):
        # if no face or eye regions could be detected closed eyes cannot be determined
        self.eyes_detectable = eyes_detectable
        self.left_eye_closed = left_eye_closed
        self.right_eye_closed = right_eye_closed

class ClosedEyesDetector:
    def __init__(self):
        pass

    def set_eyes_detectable(self, dto):
        if dto.face_result == None or \
            dto.facial_landmarks_result == None or \
            dto.facial_landmarks_result.left_eye_bb == None or dto.facial_landmarks_result.right_eye_bb == None or \
            len(dto.facial_landmarks_result.left_eye_bb) != 4 or len(dto.facial_landmarks_result.right_eye_bb) != 4:
            dto.closed_eyes_result = ClosedEyesDetectorResult(False, None, None)
        else:
            dto.closed_eyes_result = ClosedEyesDetectorResult(True, None, None)

    def preprocess(self, 
                   frame, 
                   facial_landmarks_result, 
                   target_shape, 
                   nchw_layout):
        
        rois = cut_rois(frame, [facial_landmarks_result.left_eye_roi])
        l_eye = [resize_input(l_eye, target_shape, nchw_layout) for l_eye in rois]
        
        rois = cut_rois(frame, [facial_landmarks_result.right_eye_roi])
        r_eye = [resize_input(r_eye, target_shape, nchw_layout) for r_eye in rois]

        return (l_eye[0], r_eye[0])

    def process(self, eyes_detectable, left_eye_outputs, right_eye_outputs):
        return ClosedEyesDetectorResult(True, False, False)
        #
        # NOTE:
        # Disabled since gaze already sufficiently covers closed eyes scenario
        # return ClosedEyesDetectorResult(eyes_detectable, np.argmax(left_eye_outputs) == 0, np.argmax(right_eye_outputs) == 0)

class HeadPoseEstimatorResult:
    def __init__(self, 
                 frame_shape,
                 face_result, 
                 nose_landmark, 
                 angles_degrees, 
                 vector_length):
        
        self.angles = [
            # yaw
            angles_degrees[0][0][0], 
            # pitch
            angles_degrees[1][0][0], 
            # roll
            angles_degrees[2][0][0]]     
         
        output_transform = OutputTransform(frame_shape[:2], None)
        x = int(face_result.bounding_box[0] + output_transform.scale(face_result.roi.size[0] * nose_landmark[0]))
        y = int(face_result.bounding_box[1] + output_transform.scale(face_result.roi.size[1] * nose_landmark[1]))
        # conversion of angles (degrees) to xyz coordinates for annotation in frame
        self.cos_y = cos(self.angles[0] * pi / 180)
        self.sin_y = sin(self.angles[0] * pi / 180)
        self.cos_p = cos(self.angles[1] * pi / 180)
        self.sin_p = sin(self.angles[1] * pi / 180)
        self.cos_r = cos(self.angles[2] * pi / 180)
        self.sin_r = sin(self.angles[2] * pi / 180)
        # vector center -> right
        self.v_cr = Point2D(
            (x, y), 
            (x + int(vector_length * (self.cos_r * self.cos_y + self.sin_y * self.sin_p * self.sin_r)), y + int(vector_length * self.cos_p * self.sin_r))
        )
        # vector center -> top
        self.v_ct = Point2D(
            (x, y), 
            (x + int(vector_length * (self.cos_r * self.sin_y * self.sin_p + self.cos_y * self.sin_r)), y - int(vector_length * self.cos_p * self.cos_r))
        )
        # vector center -> front
        self.v_cf = Point2D(
            (x, y), 
            (x + int(vector_length * self.sin_y * self.cos_p), y + int(vector_length * self.sin_p))
        )

class HeadPoseEstimator:
    def __init__(self):
        pass

    def preprocess(self, frame, rois, target_shape, nchw_layout):
        # rois: coordinates of the face
        inputs = cut_rois(frame, rois)
        inputs = [resize_input(input, target_shape, nchw_layout) for input in inputs]
        return inputs

    def process(self, frame_shape, face_result, nose_landmark, outputs, vector_length):
        return HeadPoseEstimatorResult(frame_shape, face_result, nose_landmark, outputs, vector_length)
    
class GazeEstimatorResult:
    def __init__(self, frame_shape, face_result, landmarks_result, outputs, vector_length):
        self.yaw = outputs[0][0]
        self.pitch = outputs[0][1]
        self.roll = outputs[0][2]
        self.coordinates = []
        output_transform = OutputTransform(frame_shape[:2], None)
        for i, c in enumerate(landmarks_result.coordinates[0]):
            if i > 1:
                break
            x_start = int(face_result.bounding_box[0] + output_transform.scale(face_result.roi.size[0] * c[0]))
            y_start = int(face_result.bounding_box[1] + output_transform.scale(face_result.roi.size[1] * c[1]))
            # x_start, y_start, x_end, y_end (for each eye)
            self.coordinates.append((x_start, 
                                     y_start,
                                     int(x_start + self.yaw * vector_length), 
                                     int(y_start - self.pitch * vector_length)))

class GazeEstimator:
    def __init__(self, vector_length):
        self.vector_length = vector_length

    def preprocess(self, 
                   frame, 
                   facial_landmarks_result, 
                   target_shape, 
                   nchw_layout):
        
        rois = cut_rois(frame, [facial_landmarks_result.left_eye_roi])
        l_eye = [resize_input(l_eye, target_shape, nchw_layout) for l_eye in rois]

        rois = cut_rois(frame, [facial_landmarks_result.right_eye_roi])
        r_eye = [resize_input(r_eye, target_shape, nchw_layout) for r_eye in rois]

        return (l_eye[0], r_eye[0])

    def process(self, frame_shape, face_result, landmarks_result, outputs):
        return GazeEstimatorResult(frame_shape, face_result, landmarks_result, outputs, self.vector_length)
    
class TrackingResult:
    def __init__(self, raise_alarm):
        self.raise_alarm = raise_alarm

class Tracker:
    def __init__(self, angles_window_size, min_distraction_time_in_frames, threshold):
        # averaging angles to increase robustness
        self.angles_window_size = angles_window_size
        self.angles_window = RingBuffer(self.angles_window_size, dtype=np.float)
        # init with "neutral" values
        for i in range(self.angles_window_size):
            self.angles_window.append(0.0)
        # the min distraction time in frames until an alarm is raised (e.g. 2 seconds -> 60 fp, assuming 30 fps)
        self.min_distraction_time_in_frames = min_distraction_time_in_frames
        self.distraction_window = RingBuffer(self.min_distraction_time_in_frames, dtype=bool)
        # init with "neutral" values
        for i in range(self.min_distraction_time_in_frames):
            self.distraction_window.append(False)
        # min gaze angle to qualify for a distraction
        self.threshold = threshold

    def preprocess(self, pitch):
        self.angles_window.append(pitch)

    def postprocess(self, aggregateResult):
        # if median < threshold add
        if np.mean(np.array(self.angles_window)) < self.threshold:
            self.distraction_window.append(True)
        else:
            self.distraction_window.append(False)

        aggregateResult.tracking_result = TrackingResult(np.all(np.array(self.distraction_window)))
        return aggregateResult