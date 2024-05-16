import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

realsense_ctx = rs.context()
connected_devices = []
for for_a in range(len(realsense_ctx.devices)):
    detected_camera = realsense_ctx.devices[for_a].get_info(rs.camera_info.serial_number)
    connected_devices += [detected_camera]

device = connected_devices[0]

pipeline = rs.pipeline()
config = rs.config()

config.enable_device(device)

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

while 1:
    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()
    if not depth:
        break

    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    width = color_image.shape[1]
    height = color_image.shape[0]

    depth_image = cv2.convertScaleAbs(depth_image, alpha = 0.03)
    depth_color = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

    frame_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    img = color_image

    result_process = {}
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for ids, landmrk in enumerate(hand_landmarks.landmark):
                cx = int(landmrk.x * width)
                cy = int(landmrk.y * height)

                if cx >= width:
                    cx = 0

                if cy >= height:
                    cy = 0

                result_process[ids] = [cx, cy]

                cv2.circle(img, (cx, cy), 1, (255, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(depth_color, (cx, cy), 1, (255, 255, 0), -1, cv2.LINE_AA)

    if 8 in result_process:
        # print(result_process[8][1], result_process[8][0])
        print(depth_image[result_process[8][1]][result_process[8][0]])

    # images = cv2.addWeighted(img, 0.5, depth_color, (1 - 0.5), 0) 
    images = depth_color

    cv2.imshow('Test', images)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()