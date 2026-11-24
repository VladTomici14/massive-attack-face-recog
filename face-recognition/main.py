import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import argparse


def generateRandomString():
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result_str = ''.join(np.random.choice(list(letters), size=5))
    return result_str

def drawTitleRectangle(image, start_point, end_point, text):
    cv2.rectangle(image, start_point, end_point, (0, 0, 255), 2)
    cv2.rectangle(image, (start_point[0], start_point[1] - 30),
                  (start_point[0] + (int)((end_point[0] - start_point[0]) * 3 / 4), start_point[1]), (0, 0, 255), -1)

    cv2.putText(image, text, (start_point[0] + 5, start_point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

def procesFrame(image, detector):
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)

    blank_image = np.zeros_like(image)

    if detection_result.detections:
        for detection in detection_result.detections:
            # ----- extracting the bounding rectangles -----
            bbox = detection.bounding_box

            # Convert normalized/relative coordinates to pixel coordinates
            start_point = (int(bbox.origin_x), int(bbox.origin_y))
            end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))

            # ----- extracting the face ROI -----
            extracted_face = image[start_point[1]:end_point[1], start_point[0]:end_point[0]]

            # ----- applying a grayscale filter -----
            gray_face = cv2.cvtColor(extracted_face, cv2.COLOR_BGR2GRAY)
            gray_face_bgr = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2BGR)
            # (random_x, random_y) = (np.random.randint(0, blank_image.shape[1] - extracted_face.shape[1]),
            #                         np.random.randint(0, blank_image.shape[0] - extracted_face.shape[0]))

            # blank_image[random_y:random_y + extracted_face.shape[0], random_x:random_x + extracted_face.shape[
            #     1]] = gray_face_bgr

            blank_image[start_point[1]:end_point[1], start_point[0]:end_point[0]] = gray_face_bgr

            # ----- drawing rectangle and title -----
            drawTitleRectangle(blank_image, start_point, end_point, generateRandomString())

    return blank_image

if __name__ == "__main__":
    # ----- setup MediaPipe Face Detection -----
    base_options = python.BaseOptions(model_asset_path='detector.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    # ----- configuring the parameters -----
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input", type=str, default="0", help="Path to input video file. Leave blank to use webcam")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to output video file")
    args = parser.parse_args()

    # ----- setting up the video capture source -----
    if args.input == "0":
        cap = cv2.VideoCapture(1)
    else:
        cap = cv2.VideoCapture(args.input)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30

    # ----- setting up the video writer -----
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # ----- extracting faces from the video capture source -----
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # ----- processing the frame -----
        processedFrame = procesFrame(image, detector)

        # ----- writing the processed frame to output video -----
        out.write(processedFrame)

        # ----- displaying the output video -----
        cv2.imshow("massive attack live show", processedFrame)
        cv2.waitKey(50)

        # ----- breaking the loop if 'esc' is pressed -----
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

