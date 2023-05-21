#!/usr/bin/env python3

import os
import sys
import logging as log

from argparse import ArgumentParser, SUPPRESS
from openvino.runtime import Core, get_version
from pipeline.models import SISO_IEModel, SIMO_IEModel, MISO_IEModel
from pipeline.result_renderer import ResultRenderer
from pipeline.steps import run_pipeline

path = os.path.join(os.getcwd(), 'common/python')
sys.path.append(path)

from images_capture import open_images_capture
import monitors

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
   
    args.add_argument('-i', '--input', required=True, help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    
    args.add_argument('--loop', default=False, action='store_true',
                      help='Optional. Enable reading the input in a loop.')
    
    args.add_argument('-o', '--output', required=False,
                      help='Optional. Name of the output file(s) to save.')
    
    args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                      help='Optional. Number of frames to store in output. '
                           'If 0 is set, all frames are stored.')
    
    args.add_argument('-m_fd', '--m_facedetector', help='Required. Path to face detection model.', required=True, type=str)
    args.add_argument('-m_flmd', '--m_faciallandmarksdetector', help='Required. Path to facial landmark detection model.', required=True, type=str)
    args.add_argument('-m_ce', '--m_closedeyesdetector', help='Required. Path to closed eyes detection model.', required=True, type=str)
    args.add_argument('-m_hpe', '--m_headposeestimator', help='Required. Path to head pose estimator model.', required=True, type=str)
    args.add_argument('-m_ge', '--m_gazeestimator', help='Required. Path to gaze estimator model.', required=True, type=str)

    args.add_argument('-d', '--device',
                      help="Optional. Specify a device to infer on (the list of available devices is shown below). Use "
                           "'-d MYRIAD to specify NCS2 plugin. Use "
                           "'-d CPU to specify CPU plugin. Use "
                           "'-d GPU to specify GPU plugin. Default is MYRIAD",
                      default='MYRIAD', type=str)
    
    args.add_argument('--no_show', action='store_true', help="Optional. Don't show output.")

    args.add_argument('-u', '--utilization-monitors', default='', type=str, help='Optional. List of monitors to show initially.')
    return parser


def main():
    args = build_argparser().parse_args()

    log.info('OpenVINO Runtime')
    log.info('\tbuild: {}'.format(get_version()))
    core = Core()

    decoder_target_device = 'CPU'
    if args.device != 'CPU':
        target_device = args.device
    else:
        target_device = decoder_target_device

    models = []
    models.append(SISO_IEModel(args.m_facedetector, core, target_device, model_type='Face Detector', num_requests=1))
    models.append(SISO_IEModel(args.m_faciallandmarksdetector, core, target_device, model_type='Facial Landmarks Detector', num_requests=1))
    models.append(SISO_IEModel(args.m_closedeyesdetector, core, target_device, model_type='Closed Eyes Detector', num_requests=1))
    models.append(SIMO_IEModel(args.m_headposeestimator, core, target_device, model_type='Head Pose Estimator', num_requests=1))    
    models.append(MISO_IEModel(args.m_gazeestimator, core, target_device, model_type='Head Pose Estimator', num_requests=1))    

    presenter = monitors.Presenter(args.utilization_monitors, 70)

    renderer = ResultRenderer(
        no_show=args.no_show, 
        presenter=presenter, 
        output=args.output, 
        limit=args.output_limit,
        output_height=720)
    
    cap = open_images_capture(args.input, args.loop)

    # 3450, 3750      [not d. - quick glances down]
    # 8300, 8700      [not d. - left turn - 90 degrees]
    # 20800, 21500    [not d. - right turn - 90 degrees]
    # 17400, 17800    [not d. - illumination change]
    # 18360, 18900    [not d. - several gaze deviations with small distraction times]

    # 8700, 9200      [d. - glance down right]  
    # 23920, 24200    [d. - glance down right]
    # 19450, 19800    [d. - glance down left]
    # 9000, 9340      [d. - short glance slightly above acceptable time]
    # 10200, 10600    [d. - prolonged distraction and quick warning reset when facing forward again]

    # 12360, 12900    [d. - closed eyes, long, short, below min time]
    # 15300, 15600    [d. - closed eyes]
    # 14800, 15650    [d. - closed eyes, but while driving curve]
    # 15600, 16000    [d. - closed eyes and distracted]

    start_end_frames = (12360, 12900) 

    run_pipeline(
        cap, 
        models, 
        renderer.render_frame, 
        start_end_frames[0],
        start_end_frames[1],
        fps=cap.fps())

    for rep in presenter.reportMeans():
        log.info(rep)

if __name__ == '__main__':
    sys.exit(main() or 0)