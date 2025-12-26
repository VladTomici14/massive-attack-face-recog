import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import argparse

parser = argparse.ArgumentParser(description='Face Detection with MediaPipe')
parser.add_argument('--input', type=str, default="0", help='Path to input video file. Leave blank to use webcam')
parser.add_argument('--output', type=str, default="output.mp4", help='Path to output video file')
args = parser.parse_args()

def generateRandomString():
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    result_str = ''.join(np.random.choice(list(letters), size=5))
    return result_str


def drawTitleRectangle(image, start_point, end_point, text):
    cv2.rectangle(image, (start_point[0], start_point[1] - 30),
                  (start_point[0] + (int)((end_point[0] - start_point[0]) * 3 / 4), start_point[1]), (0, 0, 255), -1)

    cv2.putText(image, text, (start_point[0] + 5, start_point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)


def extractFaces():
    # ----- setup MediaPipe Face Detection -----
    base_options = python.BaseOptions(model_asset_path='detector.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    cap = cv2.VideoCapture("input2.mp4")
    # cap = cv2.VideoCapture(1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30

    # --- initialising the writer ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('massive-attack-output.mp4', fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # 2. Convert OpenCV BGR to MediaPipe Image object
        # Note: MediaPipe Tasks handle the RGB conversion internally if you use .create_from_ndarray
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 3. Detect faces
        detection_result = detector.detect(mp_image)

        blank_image = np.zeros_like(image)

        # 4. Process results
        if detection_result.detections:
            for detection in detection_result.detections:
                bbox = detection.bounding_box

                # Convert normalized/relative coordinates to pixel coordinates
                start_point = (int(bbox.origin_x), int(bbox.origin_y))
                end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))

                extracted_face = image[start_point[1]:end_point[1], start_point[0]:end_point[0]]

                gray_face = cv2.cvtColor(extracted_face, cv2.COLOR_BGR2GRAY)

                gray_face_bgr = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2BGR)

                # (random_x, random_y) = (np.random.randint(0, blank_image.shape[1] - extracted_face.shape[1]),
                #                         np.random.randint(0, blank_image.shape[0] - extracted_face.shape[0]))

                # blank_image[random_y:random_y + extracted_face.shape[0], random_x:random_x + extracted_face.shape[
                #     1]] = gray_face_bgr

                blank_image[start_point[1]:end_point[1], start_point[0]:end_point[0]] = gray_face_bgr

                # Draw the box
                # cv2.rectangle(blank_image, (random_x, random_y), (random_x + extracted_face.shape[1], random_y + extracted_face.shape[0]), (0, 0, 255), 2)

                drawTitleRectangle(blank_image, start_point, end_point, generateRandomString())

                cv2.rectangle(blank_image, start_point, end_point, (0, 0, 255), 2)

            out.write(blank_image)

            # 5. Display
            cv2.imshow('MediaPipe New Tasks API', blank_image)
            cv2.waitKey(50)
            if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    extractFaces()
