import time
import cv2

from .meters import MovingAverageMeter
from .models import AsyncWrapper
from .pipeline import AsyncPipeline 
from .pipeline import PipelineStep
from .queue import Signal
from .components import DTO

from pipeline.components import FaceDetector
from pipeline.components import FacialLandmarksDetector
from pipeline.components import ClosedEyesDetector
from pipeline.components import HeadPoseEstimator
from pipeline.components import GazeEstimator
from pipeline.components import Tracker

def run_pipeline(capture, 
                 models, 
                 render_fn, 
                 video_start_frame, 
                 video_stop_frame, 
                 fps=30):
    
    pipeline = AsyncPipeline()

    # one pipeline "step" encompasses at most
    # - pre-processing
    # - inference
    # - post-processing
    
    pipeline.add_step("DataAquisition",
                      DataStep(capture, video_start_frame, video_stop_frame),  
                      parallel=False)

    pipeline.add_step("FaceDetection",
                      # confidence threshold, slack
                      FaceDetectionStep(models[0], 0.7, 1.15),
                      parallel=False)
    
    pipeline.add_step("FacialLandmarkDetection",
                      # eye bounding box width, height
                      FacialLandmarksStep(models[1], (60, 60)),
                      parallel=False)
    
    pipeline.add_step("ClosedEyesDetection",
                      ClosedEyesStep(models[2]),
                      parallel=False)

    pipeline.add_step("HeadPoseEstimation",
                      # vector length
                      HeadPoseStep(models[3], 100),
                      parallel=False)
    
                      # vector length
    pipeline.add_step("GazeEstimation",
                      GazeStep(models[4], 750), 
                      parallel=False)
    
    pipeline.add_step("Tracking",
                      TrackingStep(5, 60, -0.1),
                      parallel=False)

    pipeline.add_step("OSD", 
                      RenderStep(render_fn, fps=fps), 
                      parallel=True)

    pipeline.run()
    pipeline.close()
    pipeline.print_statistics()


class DataStep(PipelineStep):
    def __init__(self, capture, first_frame, last_frame):
        super().__init__()
        self.cap = capture
        self.first_frame = first_frame
        for _ in range(first_frame):
            _ = self.cap.read()
        self.current_frame = self.first_frame
        self.last_frame = last_frame

    def setup(self):
        pass

    def process(self, _):
        self.current_frame += 1
        frame = self.cap.read()
        if frame is None or self.current_frame > self.last_frame:
            return Signal.STOP
        return frame

    def end(self):
        pass


class FaceDetectionStep(PipelineStep):
    def __init__(self, inferer, confidence_threshold, roi_scale_factor, num_requests = 1):
        super().__init__()
        self.faceDetector = FaceDetector(confidence_threshold, roi_scale_factor)
        self.inferer = inferer
        self.input_shape = self.inferer.input_shape
        assert len(self.input_shape) == 4, 'Expecting an input layer dimension of four.'
        assert self.input_shape[2] == self.input_shape[3], 'Expecting the width and height of the input layer to be identical.'
        self.async_model = AsyncWrapper(self.inferer, num_requests)

    def __del__(self):
        pass

    def process(self, frame):        
        # pre-process
        preprocessed = self.faceDetector.preprocess(frame, self.input_shape, True)
        # infer
        outputs, frame = self.async_model.infer(preprocessed, frame)
        if outputs is None:
            return None
        # post-process
        dto = DTO()
        dto.face_result = self.faceDetector.process(outputs, frame.shape)
        return frame, dto, {'face_detector': self.own_time.last}


class FacialLandmarksStep(PipelineStep):
    def __init__(self, inferer, eye_bb_shape, num_requests = 1):
        super().__init__()
        self.facialLandmarksDetector = FacialLandmarksDetector(eye_bb_shape)
        self.inferer = inferer
        self.input_shape = self.inferer.input_shape
        assert len(self.input_shape) == 4, 'Expecting an input layer dimension of four.'
        assert self.input_shape[2] == self.input_shape[3], 'Expecting the width and height of the input layer to be identical.'
        self.async_model = AsyncWrapper(self.inferer, num_requests)

    def __del__(self):
        pass

    def process(self, aggregateResult):
        frame = aggregateResult[0]
        dto = aggregateResult[1]

        if dto.face_result == None:
            # skip landmark detection (no face detected in frame)
            return frame, dto, {'facial_landmarks_detector': self.own_time.last}
        
        # pre-process
        preprocessed = self.facialLandmarksDetector.preprocess(frame, dto.face_result.roi, self.input_shape, True)
        assert len(preprocessed) == 1, 'Remove if moving to batched processing'
        # inference
        outputs, frame = self.async_model.infer(preprocessed[0], frame)
        if outputs is None:
            return None
        # post-process  
        dto.facial_landmarks_result = self.facialLandmarksDetector.process(frame.shape, dto.face_result, outputs)
        return frame, dto, {'facial_landmarks_detector': self.own_time.last}


class ClosedEyesStep(PipelineStep):
    def __init__(self, inferer,  num_requests = 1):
        super().__init__()
        self.close_eyes_detector = ClosedEyesDetector()
        self.inferer = inferer
        self.input_shape = self.inferer.input_shape
        assert len(self.input_shape) == 4, 'Expecting an input layer dimension of four.'
        assert self.input_shape[1] == self.input_shape[2], 'Expecting the width and height of the input layer to be identical.'
        self.async_model = AsyncWrapper(self.inferer, num_requests)

    def __del__(Self):
        pass

    def process(self, aggregateResult):
        frame = aggregateResult[0]
        dto = aggregateResult[1]

        self.close_eyes_detector.set_eyes_detectable(dto)
        if not dto.closed_eyes_result.eyes_detectable:
            return frame, dto, {'closed_eyes_detector': self.own_time.last}
        
        facial_landmarks_result = dto.facial_landmarks_result

        # pre-process
        left_eye, right_eye = self.close_eyes_detector.preprocess(frame, facial_landmarks_result, self.input_shape, False)
        # inference
        left_eye_outputs, frame = None, frame # self.async_model.infer(left_eye, frame) 
        right_eye_outputs, frame = None, frame # self.async_model.infer(right_eye, frame)
        # post-process
        dto.closed_eyes_result = self.close_eyes_detector.process(dto, left_eye_outputs, right_eye_outputs)

        return frame, dto, {'closed_eyes_detector': self.own_time}


class HeadPoseStep(PipelineStep):
    def __init__(self, inferer, vector_length, num_requests = 1):
        super().__init__()
        self.headPoseEstimator = HeadPoseEstimator()
        self.inferer = inferer
        self.input_shape = self.inferer.input_shape
        self.vector_length = vector_length
        self.async_model = AsyncWrapper(self.inferer, num_requests)

    def __del__(self):
        pass

    def process(self, aggregateResult):
        frame = aggregateResult[0]
        dto = aggregateResult[1]

        if not dto.closed_eyes_result.eyes_detectable:
            # skip head pose estimation (essential information are missing)
            return frame, dto, {'head_pose_estimator': self.own_time.last}
       
        face_result = dto.face_result
        landmarks_result = dto.facial_landmarks_result

        # pre-process
        preprocessed = self.headPoseEstimator.preprocess(frame, face_result.roi, self.input_shape, True)
        # inference
        outputs, frame = self.async_model.infer(preprocessed[0], frame)
        if outputs is None:
            return None
        # post-process
        dto.head_pose_result = self.headPoseEstimator.process(
            frame.shape,
            face_result, 
            landmarks_result.coordinates[0][2], 
            outputs,
            self.vector_length)
        return frame, dto, {'head_pose_estimator': self.own_time.last}


class GazeStep(PipelineStep):
    def __init__(self, inferer, vector_length, num_requests = 1):
        super().__init__()
        self.gazeEstimator = GazeEstimator(vector_length)
        self.inferer = inferer
        assert self.inferer.input_shapes[0] == self.inferer.input_shapes[1], 'Expected input shapes for left and right eye detector to be equal'
        self.input_shape = self.inferer.input_shapes[0]
        self.async_model = AsyncWrapper(self.inferer, num_requests)

    def __del__(self):
        pass

    def process(self, aggregateResult):
        frame = aggregateResult[0]
        dto = aggregateResult[1]

        if not dto.closed_eyes_result.eyes_detectable:
            # skip gaze estimation (essential inputs missing)
            return frame, dto, {'gaze_estimator': self.own_time.last}

        facial_landmarks_result = dto.facial_landmarks_result

        head_pose_result = dto.head_pose_result
        if head_pose_result == None:
            # skip gaze estimation (essential inputs missing)
            return frame, dto, {'gaze_estimator': self.own_time.last}

        # pre-process
        left_eye, right_eye = self.gazeEstimator.preprocess(frame, facial_landmarks_result, self.input_shape, True)
        # inference
        outputs, frame = self.async_model.infer([left_eye, right_eye, [head_pose_result.angles]], frame)
        # post-process
        dto.gaze_result = self.gazeEstimator.process(frame.shape, dto.face_result, facial_landmarks_result, outputs)

        return frame, dto, {'gaze_estimator': self.own_time.last}


class RenderStep(PipelineStep):
    def __init__(self, render_fn, fps):
        super().__init__()
        self.render = render_fn
        self.fps = fps
        self._frames_processed = 0
        self._t0 = None
        self._render_time = MovingAverageMeter(0.9)

    def process(self, aggregateResult):
        if aggregateResult is None:
            return
        self._sync_time()
        render_start = time.time()
        status = self.render(*aggregateResult, self._frames_processed, self.fps)
        self._render_time.update(time.time() - render_start)

        self._frames_processed += 1
        if status is not None and status < 0:
            return Signal.STOP_IMMEDIATELY
        return status

    def end(self):
        cv2.destroyAllWindows()

    def _sync_time(self):
        now = time.time()
        if self._t0 is None:
            self._t0 = now
        expected_time = self._t0 + (self._frames_processed + 1) / self.fps
        if self._render_time.avg:
            expected_time -= self._render_time.avg
        if expected_time > now:
            time.sleep(expected_time - now)


class TrackingStep(PipelineStep):
    def __init__(self, sliding_window_size, waiting_time_in_frames, threshold):
        super().__init__()
        self.tracker = Tracker(sliding_window_size, waiting_time_in_frames, threshold)
        
    def process(self, aggregateResult):
        frame = aggregateResult[0]
        dto = aggregateResult[1]

        if not dto.closed_eyes_result.eyes_detectable:
            return frame, dto, {'gaze_estimator': self.own_time.last}

        if dto.gaze_result == None:
            return frame, dto, {'tracker': self.own_time.last}

        self.tracker.preprocess(dto.gaze_result.pitch)
        self.tracker.postprocess(dto)
        
        return frame, dto, {'tracker': self.own_time.last}