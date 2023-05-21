import os
import sys
import logging
import cv2
import uuid
import numpy as np

from collections import defaultdict
from functools import partial

from .meters import WindowAverageMeter

common_path = os.path.join(os.getcwd(), 'common/python')
sys.path.append(common_path)

from model_api.models import OutputTransform

FONT_STYLE = cv2.FONT_HERSHEY_DUPLEX
FONT_SIZE = 1
TEXT_VERTICAL_INTERVAL = 45
TEXT_LEFT_MARGIN = 15

class ResultRenderer:
    def __init__(self, 
                 no_show, 
                 presenter, 
                 output, 
                 limit, 
                 output_height):
        self.no_show = no_show
        self.presenter = presenter
        self.output = output
        self.limit = limit
        self.video_writer = cv2.VideoWriter()
        self.output_height = output_height
        self.meters = defaultdict(partial(WindowAverageMeter, 16))
        self.frame_nr = 0

    def update_timers(self, timers):
        inference_time = 0.0
        for key, val in timers.items():
            self.meters[key].update(val)
            inference_time += self.meters[key].avg
        return inference_time

    def render_face_bounding_box(self, frame, face_result):
        fbb = face_result.bounding_box
        cv2.rectangle(frame, (fbb[0], fbb[1]), (fbb[2], fbb[3]), (0, 255, 0), thickness = 1)
        
    def render_facial_landmarks(self, frame, face_result, landmarks_result, save_samples = False):
        l_eye_bb = landmarks_result.left_eye_bb
        r_eye_bb = landmarks_result.right_eye_bb

        if save_samples:
            suffix = f'{str(uuid.uuid4())}'
            l_im = Image.fromarray(frame[l_eye_bb[1]:l_eye_bb[3], l_eye_bb[0]:l_eye_bb[2]])
            l_im.save(f'samples\\images\\eyes\\l_eye_{suffix}.jpeg')
            r_im = Image.fromarray(frame[r_eye_bb[1]:r_eye_bb[3], r_eye_bb[0]:r_eye_bb[2]])
            r_im.save(f'samples\\images\\eyes\\r_eye_{suffix}.jpeg')
    
        cv2.rectangle(frame, (l_eye_bb[0], l_eye_bb[1]), (l_eye_bb[2], l_eye_bb[3]), (255, 0, 0), 1)
        cv2.rectangle(frame, (r_eye_bb[0], r_eye_bb[1]), (r_eye_bb[2], r_eye_bb[3]), (255, 0, 0), 1)

        output_transform = OutputTransform(frame.shape[:2], None)
        for _, c in enumerate(landmarks_result.coordinates[0]):
            x = face_result.bounding_box[0] + output_transform.scale(face_result.roi.size[0] * c[0])
            y = face_result.bounding_box[1] + output_transform.scale(face_result.roi.size[1] * c[1])
            cv2.circle(frame, (int(x), int(y)), 1, (255, 255, 255), 2)

    def render_head_pose_angles(self, frame, head_pose_result):
        hPR = head_pose_result
        cv2.line(frame, hPR.v_cr.x, hPR.v_cr.y, (0, 0, 255), thickness=1)
        cv2.line(frame, hPR.v_ct.x, hPR.v_ct.y, (0, 255, 0), thickness=1)
        cv2.line(frame, hPR.v_cf.x, hPR.v_cf.y, (255, 0, 0), thickness=1)

    def render_gaze_angles(self, frame, gaze_result):
        for c in gaze_result.coordinates:
            cv2.line(frame, (c[0], c[1]), (c[2], c[3]), (0, 0, 255), thickness = 2)

    def render_warning(self, frame, tracking_result):
        fill_area(frame, (0, 70), (500, 0), alpha=0.6, color=(0, 0, 0))
        if tracking_result.raise_alarm:
            text_loc = (TEXT_LEFT_MARGIN, TEXT_VERTICAL_INTERVAL)
            cv2.putText(frame, "WARNING - Driver Distracted", text_loc, FONT_STYLE, FONT_SIZE, (0, 0, 255))

    def render_frame(self, frame, result, timers, frame_ind, fps):
        inference_time = self.update_timers(timers)
        self.frame_nr += 1

        w, h, _ = frame.shape
        new_h = self.output_height
        new_w = int(h * (new_h / w))
        frame = cv2.resize(frame, (new_w, new_h))

        self.presenter.drawGraphs(frame)

        face_result = result.face_result
        if face_result != None and \
           face_result.bounding_box != None and len(face_result.bounding_box) == 4:
            self.render_face_bounding_box(frame, face_result)

            landmarks_result = result.facial_landmarks_result
            if landmarks_result != None and \
                landmarks_result.coordinates != None and len(landmarks_result.coordinates) > 0:
                self.render_facial_landmarks(frame, face_result, landmarks_result, False)

                if result.head_pose_result != None:
                    self.render_head_pose_angles(frame, result.head_pose_result)

                    if result.gaze_result != None:
                        self.render_gaze_angles(frame, result.gaze_result)

                        if result.tracking_result != None:
                            self.render_warning(frame, result.tracking_result)

                # NOTE: closed eye detection not incorporated (e.g. via tracker) due to insufficient accuracy and
                #       due to the gaze estimation already covering the desired use cases
                #
                # if result.closed_eyes_result != None:
                    # print(f'Eyes closed: {result.closed_eyes_result.left_eye_closed and result.closed_eyes_result.right_eye_closed}')

        if frame_ind == 0  and \
            self.output and \
            not self.video_writer.open(self.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0])):
            logging.error("Can't open video writer")
            return -1

        if self.video_writer.isOpened() and (self.limit <= 0 or frame_ind <= self.limit-1):
            self.video_writer.write(frame)

        if not self.no_show:
            cv2.imshow("Distracted Driving Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in {ord('q'), ord('Q'), 27}:
                return -1
            self.presenter.handleKey(key)        

def fill_area(image, bottom_left, top_right, color=(0, 0, 0), alpha=1.):
    """Fills area with the specified color"""
    xmin, ymax = bottom_left
    xmax, ymin = top_right

    image[ymin:ymax, xmin:xmax, :] = image[ymin:ymax, xmin:xmax, :] * (1 - alpha) + np.asarray(color) * alpha
    return image
