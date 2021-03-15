import cv2 as cv
import numpy as np
import math
from timeit import default_timer as timer
from train_model import create_model, WEIGHTS_PATH
from keras.applications.mobilenet_v2 import preprocess_input
from flask import Response
from flask import Flask
from flask import render_template
import datetime
import argparse
import timeit
import threading

font = cv.FONT_HERSHEY_SIMPLEX
green = (0, 255, 0)
red = (0, 0, 255)
orage = (0, 0, 200)
line_type = cv.LINE_AA
IMAGE_SIZE = 224
MHI_DURATION = 1500  # milliseconds
THRESHOLD = 32
GAUSSIAN_KERNEL = (3, 3)
move_count = 0
fallen_count = 0

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


def start_fall_detector_realtime(input_path=1):
    ''' Capture RGB and MHI in real time and feed into model '''
    global outputFrame, lock

    model = create_model(WEIGHTS_PATH)

    # 캠 비디오 열기
    # input_path에 카메라 장치 번호 입력
    cap = cv.VideoCapture(input_path)
    print(input_path)
    if not cap.isOpened():
        print("Cannot open video/webcam {}".format(input_path))
        return

    # fps 및 프레임 너비와 높이 설정
    fps = int(cap.get(cv.CAP_PROP_FPS))
    print(fps)
    cap_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    cap_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    interval = int(max(1, math.ceil(fps / 10) if (fps / 10 - math.floor(fps / 10)) >= 0.5 else math.floor(fps / 10)))
    ms_per_frame = 1000 / fps  # milliseconds
    count = interval

    # 영상 저장 설정
    size = (int(cap_width), int(cap_height))
    label_file_path = './demo1.avi'
    crop_file_path = './crop.avi'
    mhi_file_path = './mhi.avi'
    fourcc = cv.VideoWriter_fourcc(*'DIVX')  # 인코딩 포맷 문자
    label_out = cv.VideoWriter(label_file_path, fourcc, 30, size)  # VideoWriter 객체 생성
    crop_out = cv.VideoWriter(crop_file_path, fourcc, 30, (IMAGE_SIZE, IMAGE_SIZE))  # VideoWriter 객체 생성
    mhi_out = cv.VideoWriter(mhi_file_path, fourcc, 30, (IMAGE_SIZE, IMAGE_SIZE))  # VideoWriter 객체 생성

    # mhi 설정
    prev_mhi = [np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.float32) for i in range(interval)]
    prev_mhi_short = [np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.float32)] * interval
    prev_timestamp = [i * ms_per_frame for i in range(interval)]
    prev_frames = [None] * interval
    for i in range(interval):
        ret, frame = cap.read()
        frame = cv.resize(frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv.INTER_AREA)
        frame = cv.GaussianBlur(frame, GAUSSIAN_KERNEL, 0)
        prev_frames[i] = frame.copy()
        MHI_DURATION_SHORT = 300  # Uses for putting bounding box on recent motion

        # --------------------- optical flow --------------------------
        # # Converts frame to grayscale because we
        # # only need the luminance channel for
        # # detecting edges - less computationally
        # # expensive
        # prev_gray = cv.cvtColor(prev_frames[i], cv.COLOR_BGR2GRAY)
        #
        # # Creates an image filled with zero
        # # intensities with the same dimensions
        # # as the frame
        # mask = np.zeros_like(prev_frames[i])
        #
        # # Sets image saturation to maximum
        # mask[..., 1] = 255
        # --------------------- optical flow --------------------------

    fall_frames_seen = 0  # Number of consecutive fall frames seen so far
    fall_detected = False
    fallen_detected = False
    fallen = False
    MIN_NUM_FALL_FRAME = int(13)  # Need at least some number of frames to avoid flickery classifications

    # cv.namedWindow("Capture")
    # cv.namedWindow("Cropped")
    # cv.namedWindow("MHI")
    # cv.moveWindow("Capture", 100, 100)
    # cv.moveWindow("Cropped", 500, 100)
    # cv.moveWindow("MHI", 800, 100)

    while True:

        start_time = timer()
        ret, orig_frame = cap.read()
        if not ret:
            break

        # 알고리즘 시작 시점
        start_t = timeit.default_timer()

        # Create MHI
        prev_ind = count % interval
        prev_timestamp[prev_ind] += interval * ms_per_frame
        count += 1

        frame = cv.resize(orig_frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv.INTER_AREA)
        frame = cv.GaussianBlur(frame, GAUSSIAN_KERNEL, 0)
        frame_diff = cv.absdiff(frame, prev_frames[prev_ind])
        gray_diff = cv.cvtColor(frame_diff, cv.COLOR_BGR2GRAY)
        _, motion_mask = cv.threshold(gray_diff, THRESHOLD, 1, cv.THRESH_BINARY)
        prev_frames[prev_ind] = frame.copy()

        cv.motempl.updateMotionHistory(motion_mask, prev_mhi[prev_ind], prev_timestamp[prev_ind], MHI_DURATION)
        cv.motempl.updateMotionHistory(motion_mask, prev_mhi_short[prev_ind], prev_timestamp[prev_ind],
                                       MHI_DURATION_SHORT)
        mhi = np.uint8(
            np.clip((prev_mhi[prev_ind] - (prev_timestamp[prev_ind] - MHI_DURATION)) / MHI_DURATION, 0, 1) * 255)
        mhi_short = np.uint8(
            np.clip((prev_mhi_short[prev_ind] - (prev_timestamp[prev_ind] - MHI_DURATION_SHORT)) / MHI_DURATION_SHORT,
                    0, 1) * 255)

        # --------------------- optical flow --------------------------
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # flow = cv.calcOpticalFlowFarneback(prev_gray, gray,
        #                                    None,
        #                                    0.5, 3, 15, 3, 5, 1.2, 0)
        # # Computes the magnitude and angle of the 2D vectors
        # magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        #
        # # Sets image hue according to the optical flow
        # # direction
        # mask[..., 0] = angle * 180 / np.pi / 2
        #
        # # Sets image value according to the optical flow
        # # magnitude (normalized)
        # mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        #
        # # Converts HSV to RGB (BGR) color representations
        # rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
        # --------------------- optical flow --------------------------

        # Crop out motion

        # Crop image
        x_start = y_start = IMAGE_SIZE
        x_end = y_end = 0
        contours, _ = cv.findContours(mhi_short, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if (len(contours) > 0):
            for c in contours:
                contour = cv.approxPolyDP(c, 3, True)
                x, y, w, h = cv.boundingRect(contour)
                if x < x_start:
                    x_start = x
                if y < y_start:
                    y_start = y
                if x + w > x_end:
                    x_end = x + w
                if y + h > y_end:
                    y_end = y + h
        else:
            x_start = y_start = 0
            x_end = y_end = IMAGE_SIZE

        x_start = int(np.round(x_start / IMAGE_SIZE * cap_width))
        y_start = int(np.round(y_start / IMAGE_SIZE * cap_height))
        x_end = int(np.round(x_end / IMAGE_SIZE * cap_width))
        y_end = int(np.round(y_end / IMAGE_SIZE * cap_height))
        labelled_frame = orig_frame.copy()
        cv.rectangle(
            labelled_frame, (x_start, y_start), (x_end, y_end),
            color=green, lineType=line_type
        )
        cropped = orig_frame[y_start:y_end, x_start:x_end].copy()
        try:
            cropped = cv.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv.INTER_LINEAR)
        except:
            cropped = cv.resize(orig_frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv.INTER_LINEAR)

        # Prepare input
        spatial_input = cropped.copy().astype(np.float32)
        spatial_input = cv.cvtColor(spatial_input, cv.COLOR_BGR2RGB)
        spatial_input = np.array([spatial_input])
        temporal_input = mhi.copy().astype(np.float32)
        temporal_input = cv.cvtColor(temporal_input, cv.COLOR_GRAY2RGB)
        temporal_input = np.array([temporal_input])
        preprocess_input(spatial_input)
        preprocess_input(temporal_input)
        # print(np.mean(temporal_input))

        # Make prediction
        # -------------------------------- Add Rule Based ----------------------------------
        prob = model.predict([spatial_input, temporal_input])
        if prob > 0.8:
            prediction = 1
            # print("%0.2f" % prob)
        else:
            prediction = 0

        # fall detection predict frame 수 설정
        is_fall = prediction == 1
        if is_fall:
            fall_frames_seen = min(fall_frames_seen + 1, MIN_NUM_FALL_FRAME)
            print(fall_frames_seen)
        else:
            fall_frames_seen = max(fall_frames_seen - 1, 0)

        # fall detect -0.8보다 작은 frame만 카운트
        if np.mean(temporal_input) < -0.8:
            # print(move_count, 'move_count')
            move_count += 1
        else:
            move_count = 0

        # -1에 가까울수록 움직임이 없음
        if np.mean(temporal_input) < -0.9:
            # print(np.mean(temporal_input))
            fallen_count += 1
        else:
            fallen_count = 0
            fallen_detected = False
            fallen = False

        if fall_frames_seen == MIN_NUM_FALL_FRAME and move_count >= MIN_NUM_FALL_FRAME:
            fall_detected = True
            fallen_detected = True
            print('falldown')
        elif fall_frames_seen == 0:
            fall_detected = False
            move_count = 0

        if fallen_detected and fallen_count >= fps:
            print(fallen_count, 'Fallen_count')
            fallen = True
        # -------------------------------------------------------------------------------

        # ########### 추가 fps계산 ##################
        # 알고리즘 종료 시점
        terminate_t = timeit.default_timer()
        FPS = int(1. / (terminate_t - start_t))

        # 프레임 수를 문자열에 저장
        str_fps = "FPS : %0.1f" % FPS

        cv.putText(
            labelled_frame, str_fps, (labelled_frame.shape[1] - 100, labelled_frame.shape[0] - 10),
            fontFace=font, fontScale=0.5, color=green, lineType=line_type
        )

        cv.putText(
            labelled_frame, "Status: {}".format("Fall detected!!" if fall_detected else "Fallen" if fallen else "Not Fall"), (10, 50),
            fontFace=font, fontScale=1.5, color=red if fall_detected else orage if fallen else green, lineType=line_type
        )

        #시간 표시
        # grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()
        cv.putText(labelled_frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, labelled_frame.shape[0] - 10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)


        # Show images
        # cv.imshow("Capture", labelled_frame)
        # cv.imshow("Cropped", cropped)
        # cv.imshow("MHI", mhi)

        # 웹에 뿌려줄 frame으로 복사
        with lock:
            outputFrame = labelled_frame.copy()

        # 영상 저장
        label_out.write(labelled_frame)
        crop_out.write(cropped)
        mhi_out.write(mhi)

        # Opens a new window and displays the output frame
        # cv.imshow("dense optical flow", rgb)

        # Compensate for elapsed time used to process frame
        wait_time = int(max(1, ms_per_frame - (timer() - start_time) * 1000))
        if cv.waitKey(wait_time) == 27:
            break

    label_out.release()
    crop_out.release()
    mhi_out.release()
    cap.release()
    cv.destroyAllWindows()


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock, test

    # test = 2

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                # print('outputFrame is None!!')
                continue

            # encode the frame in JPEG format
            # cv2.imshow(outputFrame)
            (flag, encodedImage) = cv.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--video_number", type=int, default=1,
                    help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    # start a thread that will perform motion detection
    t = threading.Thread(target=start_fall_detector_realtime, args=(
        args["video_number"],))
    t.daemon = True
    t.start()

    print(args['ip'])
    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)


#
# if __name__ == "__main__":
#     # start_fall_detector_FDD(
#     #     "datasets/FDD/Coffee_room_01/Videos/video (5).avi",
#     #     "datasets/FDD/Coffee_room_01/Annotation_files/video (5).txt"
#     # )
#     #start_fall_detector_realtime("datasets/FDD/Home_02/Videos/video (40).avi")
#     # start_fall_detector_realtime("datasets/URFD/Fall_videos/fall-15-cam0-rgb.avi")
#     #start_fall_detector_realtime("./S024C001P057R001A071_rgb.avi")
#
#     #start_fall_detector_realtime("./datasets/FDD/Home_01/Videos/video (5).avi")
#     start_fall_detector_realtime()