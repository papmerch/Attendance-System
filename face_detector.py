"""
Face Detection using OpenCV DNN (Deep Neural Network) Module

This program captures real-time video from the webcam and detects human faces
using a pre-trained Caffe model (Single Shot Detector - SSD) loaded via
OpenCV's DNN module. Bounding boxes and confidence scores are drawn on each
detected face.

Requirements:
    - opencv-python
    - deploy.prototxt (model architecture)
    - res10_300x300_ssd_iter_140000.caffemodel (pre-trained weights)
"""

import cv2
import sys
import os


def load_face_detection_model(prototxt_path, model_path):
    """
    Load the pre-trained face detection model using OpenCV's DNN module.

    Args:
        prototxt_path (str): Path to the deploy.prototxt file.
        model_path (str): Path to the .caffemodel weights file.

    Returns:
        cv2.dnn_Net: Loaded DNN model.

    Raises:
        FileNotFoundError: If either model file does not exist.
    """
    # Check if model files exist before loading
    if not os.path.exists(prototxt_path):
        raise FileNotFoundError(
            f"Prototxt file not found: {prototxt_path}\n"
            "Please download it from:\n"
            "  https://raw.githubusercontent.com/opencv/opencv/master/"
            "samples/dnn/face_detector/deploy.prototxt"
        )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Caffemodel file not found: {model_path}\n"
            "Please download it from:\n"
            "  https://github.com/opencv/opencv_3rdparty/raw/"
            "dnn_samples_face_detector_20170830/"
            "res10_300x300_ssd_iter_140000.caffemodel"
        )

    print("[INFO] Loading face detection model...")
    # Load the Caffe model using OpenCV's DNN module
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    print("[INFO] Model loaded successfully.")
    return net


def preprocess_frame(frame, target_size=(300, 300), scale_factor=1.0):
    """
    Preprocess a frame into a blob suitable for DNN inference.

    Steps:
        1. Resize the frame to the target size expected by the model.
        2. Perform mean subtraction (the model was trained with mean values
           (104.0, 177.0, 123.0) for BGR channels).
        3. Optionally scale pixel values.

    Args:
        frame (numpy.ndarray): Input frame from webcam.
        target_size (tuple): (width, height) for resizing.
        scale_factor (float): Multiplier for pixel values.

    Returns:
        cv2.dnn.blob: Preprocessed blob ready for model input.
    """
    # blobFromImage performs: resize, mean subtraction, scaling, and
    # channel reordering (HWC -> CHW) in one call
    blob = cv2.dnn.blobFromImage(
        frame,
        scale_factor,
        target_size,
        (104.0, 177.0, 123.0)  # Mean subtraction values
    )
    return blob


def draw_detections(frame, detections, confidence_threshold=0.5):
    """
    Draw bounding boxes and confidence scores on detected faces.

    The model outputs a 4D tensor of shape:
      (1, 1, num_detections, 7)
    where each detection is: [batch_id, class_id, confidence, x1, y1, x2, y2]
    with coordinates normalized to [0, 1].

    Args:
        frame (numpy.ndarray): The original video frame (used for dimensions).
        detections (numpy.ndarray): Raw output from the DNN forward pass.
        confidence_threshold (float): Minimum confidence to consider a detection.

    Returns:
        int: Number of faces detected (above the threshold).
    """
    height, width = frame.shape[:2]
    face_count = 0

    # Loop over all detections returned by the model
    for i in range(detections.shape[2]):
        # Extract confidence (probability) for this detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections below the threshold
        if confidence < confidence_threshold:
            continue

        face_count += 1

        # Compute bounding box coordinates in original frame dimensions
        # The model outputs normalized coordinates in [0, 1]
        box = detections[0, 0, i, 3:7] * [width, height, width, height]
        x1, y1, x2, y2 = box.astype("int")

        # Ensure coordinates stay within frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)

        # Draw a green rectangle around the detected face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Prepare the confidence text label
        label = f"Confidence: {confidence * 100:.2f}%"

        # Place the label above the bounding box
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
        cv2.putText(
            frame, label, (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    return face_count


def main():
    """
    Main entry point for the face detection program.

    Workflow:
        1. Load the pre-trained Caffe model.
        2. Access the webcam.
        3. Continuously capture frames.
        4. For each frame: preprocess, run inference, draw detections.
        5. Show the output in a window.
        6. Exit when the user presses 'q'.
    """

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    # Paths to the model files (adjust if your directory structure differs)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROTOTXT_PATH = os.path.join(BASE_DIR, "models", "deploy.prototxt")
    MODEL_PATH = os.path.join(
        BASE_DIR, "models",
        "res10_300x300_ssd_iter_140000.caffemodel"
    )
    CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to display a detection

    # ------------------------------------------------------------------
    # Step 1: Load the face detection model
    # ------------------------------------------------------------------
    try:
        net = load_face_detection_model(PROTOTXT_PATH, MODEL_PATH)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Access the webcam
    # ------------------------------------------------------------------
    print("[INFO] Starting webcam...")
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        print("  - Check if the camera is connected.")
        print("  - Make sure no other application is using it.")
        sys.exit(1)

    print("[INFO] Webcam opened successfully.")
    print("[INFO] Press 'q' in the video window to quit.")

    # ------------------------------------------------------------------
    # Step 3: Main loop — capture, detect, display
    # ------------------------------------------------------------------
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # If the frame could not be read, skip this iteration
        if not ret:
            print("[WARNING] Failed to grab frame. Skipping...")
            continue

        # Flip the frame horizontally for a more natural mirror view
        frame = cv2.flip(frame, 1)

        # --------------------------------------------------------------
        # Step 3a: Preprocess the frame into a blob
        # --------------------------------------------------------------
        blob = preprocess_frame(frame)

        # --------------------------------------------------------------
        # Step 3b: Run the DNN forward pass (face detection)
        # --------------------------------------------------------------
        net.setInput(blob)
        detections = net.forward()

        # --------------------------------------------------------------
        # Step 3c: Draw bounding boxes and confidence scores
        # --------------------------------------------------------------
        face_count = draw_detections(frame, detections, CONFIDENCE_THRESHOLD)

        # --------------------------------------------------------------
        # Step 3d: Show the frame with detections
        # --------------------------------------------------------------
        # Display the face count on the frame
        info_text = f"Faces detected: {face_count}"
        cv2.putText(
            frame, info_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

        # Show the output in a resizable window
        cv2.imshow("Face Detection - OpenCV DNN", frame)

        # --------------------------------------------------------------
        # Step 3e: Exit condition — press 'q' to quit
        # --------------------------------------------------------------
        # cv2.waitKey(1) returns the ASCII code of the pressed key
        # 0xFF ensures we handle only the lower 8 bits (cross-platform)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] 'q' pressed. Exiting...")
            break

    # ------------------------------------------------------------------
    # Step 4: Cleanup — release resources
    # ------------------------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Resources released. Goodbye!")


if __name__ == "__main__":
    print("=" * 60)
    print("  NOTE: You're running the DETECTION-ONLY script!")
    print("  This only shows 'Confidence' — not names.")
    print()
    print("  To see NAMES + ATTENDANCE tracking, run:")
    print("    python face_recognizer.py")
    print("=" * 60)
    print()
    main()
