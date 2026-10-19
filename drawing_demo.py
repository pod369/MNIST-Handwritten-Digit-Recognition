"""Draw up to five digits and recognize them with a saved MNIST model."""
from __future__ import annotations

import argparse
from pathlib import Path


def preprocess_digit(image, cv, np):
    """Crop ink, fit inside 20x20, then center its mass in a 28x28 MNIST image."""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image
    ink = 255 - gray
    ink[ink < 20] = 0
    points = cv.findNonZero(ink)
    if points is None or len(points) < 5:
        return None
    x, y, width, height = cv.boundingRect(points)
    crop = ink[y:y + height, x:x + width]
    scale = 20 / max(width, height)
    resized = cv.resize(crop, (max(1, round(width * scale)), max(1, round(height * scale))),
                        interpolation=cv.INTER_AREA if scale < 1 else cv.INTER_LINEAR)
    output = np.zeros((28, 28), dtype=np.uint8)
    rh, rw = resized.shape
    top, left = (28 - rh) // 2, (28 - rw) // 2
    output[top:top + rh, left:left + rw] = resized
    moments = cv.moments(output)
    if moments["m00"]:
        shift = np.float32([[1, 0, 13.5 - moments["m10"] / moments["m00"]],
                            [0, 1, 13.5 - moments["m01"] / moments["m00"]]])
        output = cv.warpAffine(output, shift, (28, 28))
    return output.astype(np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=Path(__file__).resolve().parent / "outputs/deep_mlp.keras")
    parser.add_argument("--boxes", type=int, choices=range(1, 6), default=5)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Model not found: {args.model}. Run train.py first.")
    import cv2 as cv
    import numpy as np
    import tensorflow as tf

    model = tf.keras.models.load_model(args.model, compile=False)
    if tuple(model.input_shape[1:]) != (784,) or model.output_shape[-1] != 10:
        parser.error("Expected a 784-input, 10-class MNIST MLP model")
    size, gap, top = 180, 16, 70
    width = max(620, gap + (size + gap) * args.boxes)
    rectangles = [(gap + i * (size + gap), top, size, size) for i in range(args.boxes)]
    canvas = np.full((top + size + 65, width, 3), 255, dtype=np.uint8)
    previous = [None]

    def reset():
        canvas[:] = 255
        cv.putText(canvas, "r: recognize | e: erase | s: preview | q: quit", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.62, (40, 40, 40), 1)
        for x, y, w, h in rectangles:
            cv.rectangle(canvas, (x - 1, y - 1), (x + w, y + h), (0, 0, 200), 1)
        previous[0] = None

    def inside(x, y):
        for index, (rx, ry, w, h) in enumerate(rectangles):
            if rx + 4 <= x < rx + w - 4 and ry + 4 <= y < ry + h - 4:
                return index
        return None

    def draw(event, x, y, flags, _):
        box = inside(x, y)
        if event == cv.EVENT_LBUTTONDOWN or (event == cv.EVENT_MOUSEMOVE and flags & cv.EVENT_FLAG_LBUTTON):
            if box is None:
                previous[0] = None
                return
            last = previous[0]
            if last is not None and last[2] == box:
                cv.line(canvas, (last[0], last[1]), (x, y), (0, 0, 0), 10, cv.LINE_AA)
            cv.circle(canvas, (x, y), 5, (0, 0, 0), -1, cv.LINE_AA)
            previous[0] = (x, y, box)
        elif event == cv.EVENT_LBUTTONUP:
            previous[0] = None

    def digits():
        return [preprocess_digit(canvas[y:y + h, x:x + w], cv, np)
                for x, y, w, h in rectangles]

    reset()
    cv.namedWindow("MNIST Drawing")
    cv.setMouseCallback("MNIST Drawing", draw)
    try:
        while True:
            cv.imshow("MNIST Drawing", canvas)
            key = cv.waitKey(20) & 0xFF
            if key in (ord("q"), 27) or cv.getWindowProperty("MNIST Drawing", cv.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("e"):
                reset()
            elif key == ord("s"):
                images = [np.zeros((28, 28), dtype=np.float32) if d is None else d for d in digits()]
                preview = np.hstack(images)
                cv.imshow("Preprocessed digits", cv.resize(preview, None, fx=5, fy=5, interpolation=cv.INTER_NEAREST))
            elif key == ord("r"):
                images = digits()
                indices = [index for index, digit in enumerate(images) if digit is not None]
                labels = ["blank"] * args.boxes
                if indices:
                    scores = model.predict(np.stack([images[i].reshape(784) for i in indices]), verbose=0)
                    for index, probability in zip(indices, scores):
                        digit = int(np.argmax(probability))
                        labels[index] = f"{digit} ({float(probability[digit]):.0%})"
                canvas[top + size + 10:, :] = 255
                for (x, _, _, _), label in zip(rectangles, labels):
                    cv.putText(canvas, label, (x, top + size + 40), cv.FONT_HERSHEY_SIMPLEX,
                               0.65, (180, 70, 0), 2)
                print(" | ".join(labels))
    finally:
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
