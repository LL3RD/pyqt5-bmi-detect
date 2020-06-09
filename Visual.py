import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.figure as mplfigure
from matplotlib.backends.backend_agg import FigureCanvasAgg

_KEYPOINT_THRESHOLD = 0.0005
_BLACK = (0, 0, 0)
_RED = (1.0, 0, 0)
_class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush']

_keypoint_names_coco = (
'nose', 'right_eye', 'left_eye', 'right_ear', 'left_ear', 'right_shoulder', 'left_shoulder', 'right_elbow',
'left_elbow', 'right_wrist', 'left_wrist', 'right_hip', 'left_hip', 'right_knee', 'left_knee', 'right_ankle',
'left_ankle')

_keypoint_names_openpose = ("Nose", "Neck", "RShoulder", "RElbow", "RWrist",
                            "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
                            "RAnkle", "LHip", "LKnee", "LAnkle", "REye",
                            "LEye", "REar", "LEar", "Background")


class VisImage(object):
    def __init__(self, img, scale=1.0):
        # img RGB H*W*3
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        ax.set_xlim(0.0, self.width)
        ax.set_ylim(self.height)

        self.fig = fig
        self.ax = ax
        self.canvas = FigureCanvasAgg(fig)

    def get_image(self):
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        if (self.width, self.height) != (width, height):
            img = np.array(Image.fromarray(self.img, 'RGB').resize((width, height), Image.ANTIALIAS))
        else:
            img = self.img

        buffer = np.frombuffer(s, dtype="uint8")
        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)

        alpha = alpha.astype("float32") / 255.0
        visualized_image = img * (1 - alpha) + rgb * alpha

        visualized_image = visualized_image.astype("uint8")
        return visualized_image


class Visual(object):
    def __init__(self, img_rgb, scale=1.0):
        self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        self.output = VisImage(self.img, scale=scale)

    #         self.cpu_device = torch.device("cpu")

    def draw_masks_predictions(self, masks_predictions):
        if not isinstance(masks_predictions, np.ndarray):
            masks_predictions = np.asarray(masks_predictions.to('cpu'))
        color = [0, 0, 1]  # random_color(rgb=True,maximum=1)
        masks_predictions = masks_predictions  # .astype("uint8")
        alpha = 0.5
        shape2d = (masks_predictions.shape[0], masks_predictions.shape[1])

        rgba = np.zeros(shape2d + (4,), dtype="float32")
        rgba[:, :, :3] = color
        rgba[:, :, 3] = (masks_predictions > 0.1).astype("float32") * alpha

        self.output.ax.imshow(rgba)

        return self.output

    def draw_keypoints_predictions(self, keypoints_predictions, _keypoint_names='_keypoint_names_coco'):
        visiable = {}
        keypoint_names = _keypoint_names
        for idx, keypoint in enumerate(keypoints_predictions):
            # if not (idx == 0 or idx == 4 or idx == 6 or idx == 12 or idx == 14):
            #     continue
            x, y, prob = keypoint
            if prob == 1:
                self.draw_circle((x, y), color=_RED)
                if keypoint_names:
                    keypoint_name = keypoint_names[idx]
                    visiable[keypoint_name] = (x, y)
                    # print(keypoint_name)
                    # print(x,y)

        return self.output

    def draw_circle(self, circle_coord, color, radius=5):
        # x,y = circle_coord
        self.output.ax.add_patch(
            mpl.patches.Circle(circle_coord, radius=2, fill=True, color=[0, 0, 1])
        )
        self.output.ax.add_patch(
            mpl.patches.Circle(circle_coord, radius=1, fill=True, color=[0, 1, 0])
        )
        return self.output

    @property
    def image(self):
        return self.output.get_image()
