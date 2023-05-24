import tensorflow as tf
import numpy as np
import cv2

def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_frame(frame):
    input_size = (640, 640)
    frame = cv2.resize(frame, input_size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32)
    frame = (frame / 127.5) - 1
    frame = np.expand_dims(frame, axis=0)
    return frame

def postprocess_predictions(raw_predictions, score_threshold=0.35, iou_threshold=0.45, max_output_size=300):
    # Filter predictions based on score threshold
    scores = raw_predictions[0, :, 4]
    indices = scores >= score_threshold
    filtered_predictions = raw_predictions[0, indices, :]

    # Apply non-max suppression on filtered predictions
    boxes, scores, classes = non_max_suppression(
        filtered_predictions, iou_threshold, max_output_size)
    return boxes, scores, classes

def non_max_suppression(
        prediction,
        iou_thres=0.80,
        max_output_size=300,
):
    # Assume bounding box format in the prediction is [x, y, width, height]
    x = prediction[:, 0]
    y = prediction[:, 1]
    width = prediction[:, 2]
    height = prediction[:, 3]

    # Convert to the format [ymin, xmin, ymax, xmax]
    boxes = np.stack([
        y - height / 2,  # ymin
        x - width / 2,  # xmin
        y + height / 2,  # ymax
        x + width / 2  # xmax
    ], axis=-1)

    scores = prediction[:, 4]
    classes = tf.argmax(prediction[:, 5:], axis=-1)

    indices = tf.image.non_max_suppression(boxes, scores, max_output_size, iou_threshold=iou_thres)

    selected_boxes = tf.gather(boxes, indices)
    selected_scores = tf.gather(scores, indices)
    selected_classes = tf.gather(classes, indices)

    return selected_boxes.numpy(), selected_scores.numpy(), selected_classes.numpy()


def visualize_predictions(frame, predictions):
    class_names = ['front', 'front_hand', 'side', 'side_hand']
    boxes, scores, classes = predictions
    height, width, _ = frame.shape
    print('len :', len(boxes))
    for box, score, cls in zip(boxes, scores, classes):
        ymin, xmin, ymax, xmax = box
        ymin = int(ymin * height)
        xmin = int(xmin * width)
        ymax = int(ymax * height)
        xmax = int(xmax * width)
        label = '{} {:.2f}'.format(class_names[cls], score)
        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        frame = cv2.putText(frame, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

def run_inference(interpreter, cap):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_data = preprocess_frame(frame)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        raw_predictions = interpreter.get_tensor(output_details[0]['index'])
        predictions = postprocess_predictions(raw_predictions)

        frame = visualize_predictions(frame, predictions)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error opening video stream or file")

    interpreter = load_model('racket_best-fp16.tflite')

    run_inference(interpreter, cap)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
