"""
Displays text stretched to exactly fill the whole screen (width AND height,
edge to edge), white on black, with adjustable boldness.

Requirements:
    pip install opencv-python numpy

Run:
    python fullscreen_text.py

Press any key (with the window focused) to close it.
"""

import cv2
import numpy as np
import tkinter as tk

# ---- CHANGE THESE ----
TEXT = "HELLO"
BOLDNESS = 15        # 0 = normal weight, higher = bolder (try 5-30)
PADDING = 0.05        # fraction of screen size to leave as black border (0 = edge-to-edge, 0.05 = 5% margin)
# -----------------------


def get_screen_resolution():
    root = tk.Tk()
    root.withdraw()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return width, height


def render_text_mask(text, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=10, thickness=20):
    """Render text onto a generously-padded canvas (padding gets cropped away later,
    so it's just there to avoid clipping strokes/dilation)."""
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = thickness * 3
    canvas_h = text_h + baseline + pad * 2
    canvas_w = text_w + pad * 2

    img = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    x = pad
    y = text_h + pad
    cv2.putText(img, text, (x, y), font, font_scale, 255, thickness, cv2.LINE_AA)
    return img


def crop_to_content(mask):
    """Crop the image tightly to the bounding box of the non-black pixels,
    so there is zero empty margin on any side (this is what makes the final
    result truly touch every edge of the screen after resizing)."""
    ys, xs = np.where(mask > 0)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return mask[y0:y1, x0:x1]


def main():
    width, height = get_screen_resolution()

    mask = render_text_mask(TEXT, font_scale=10, thickness=20)

    # Bolder = dilate the white strokes BEFORE cropping
    if BOLDNESS > 0:
        kernel = np.ones((BOLDNESS, BOLDNESS), np.uint8)
        mask = cv2.dilate(mask, kernel)

    # Crop tightly to the actual glyph pixels (removes baseline gap / padding
    # that was leaving black margins top & bottom before)
    tight = crop_to_content(mask)

    # Leave a black margin on all sides, sized as a fraction of the screen
    pad_x = int(width * PADDING)
    pad_y = int(height * PADDING)
    target_w = width - pad_x * 2
    target_h = height - pad_y * 2

    # Stretch independently in x and y to fill that inner (padded) area
    stretched = cv2.resize(tight, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Paste it onto a full black screen-sized canvas, centered
    full = np.zeros((height, width), dtype=np.uint8)
    full[pad_y:pad_y + target_h, pad_x:pad_x + target_w] = stretched

    img = cv2.cvtColor(full, cv2.COLOR_GRAY2BGR)  # white text, black bg

    window_name = "Fullscreen Text"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
