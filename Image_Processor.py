import numpy as np
# import torchvision
import torch
import transforms
import detection
import torch.nn as nn
import joblib
import functional
import resnet

def _get_image_size(img):
    if functional._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


class Resize(transforms.Resize):

    def __call__(self, img):
        h, w = _get_image_size(img)
        scale = max(w, h) / float(self.size)
        new_w, new_h = int(w / scale), int(h / scale)
        return functional.resize(img, (new_w, new_h), self.interpolation)

class Body_Figure(object):
    def __init__(self, waist_width, thigh_width, hip_width, head_width, Area, height, shoulder_width):
        self._waist_width = waist_width
        self._thigh_width = thigh_width
        self._hip_width = hip_width
        self._head_width = head_width
        self._Area = Area
        self._height = height
        self._shoulder_width = shoulder_width

    @property
    def WSR(self):
        return self._waist_width / self._shoulder_width

    @property
    def WTR(self):
        return (self._waist_width / self._thigh_width)  # **2

    @property
    def WHpR(self):
        return (self._waist_width / self._hip_width)  # **2

    @property
    def WHdR(self):
        return (self._waist_width / self._head_width)  # **2

    @property
    def HpHdR(self):
        return (self._hip_width / self._head_width)  # **2

    @property
    def Area(self):
        return self._Area

    @property
    def H2W(self):
        return self._height / self._waist_width


IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

class Image_Processor(object):

    def __init__(self):
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__init_key()
        self.__init_mask()
        self.__init_extract()
        # self._HumanParser = HumanParser()
        self.__init_trans()
        self.regR = joblib.load('Model/krr_regression_model.pkl')

    def __init_extract(self):
        self.exT = resnet.resnet101(pretrained=False)
        self.exT.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 20),

        )
        model_dict = self.exT.state_dict()
        Pred_dict = torch.load('Model/Ours.pkl', map_location=self.DEVICE)
        Pred_dict = {k: v for k, v in Pred_dict.items() if k in model_dict and (k != 'fc.8.bias' and k != 'fc.8.weight')}
        model_dict.update(Pred_dict)
        self.exT.load_state_dict(model_dict)
        # self.exT.load_state_dict()
        self.exT = self.exT.to(self.DEVICE)

    def __init_trans(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(IMG_MEAN, IMG_STD)
        ])

    def __init_mask(self):
        self._ContourPredictor = detection.maskrcnn_resnet50_fpn(pretrained=True)
        self._ContourPredictor = self._ContourPredictor.to(self.DEVICE)
        self._ContourPredictor.eval()

    def __init_key(self):
        self._KeypointsPredictor = detection.keypointrcnn_resnet50_fpn(pretrained=True)
        self._KeypointsPredictor = self._KeypointsPredictor.to(self.DEVICE)
        self._KeypointsPredictor.eval()

    def Img_Trans(self, img):
        img = transforms.Compose([
            transforms.ToPILImage(),
            Resize(224),
            transforms.Pad(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD)
        ])(img)
        img = img.to(self.DEVICE)
        img = torch.unsqueeze(img, dim=0)
        return img

    def Predict(self):
        pass

    def _detected(self, img):
        img = self.transform(img).to(self.DEVICE)
        KeypointsOutput = self._KeypointsPredictor([img])
        key = KeypointsOutput[0]['keypoints'].detach().cpu().numpy()[0]

        ContourOutput = self._ContourPredictor([img])[0]
        mask_labels = ContourOutput['labels'].detach().cpu().numpy()
        peo_idx = np.where(mask_labels == 1)[0]
        max_scores = 0
        for i in peo_idx:
            scores = ContourOutput['scores'][i]
            if scores > max_scores:
                mask_people = ContourOutput['masks'][i]
                max_scores = scores

        mask_people = np.squeeze(mask_people.detach().cpu().numpy())
        # mask_people = np.asarray(list(map(lambda x:list(map(lambda y:y > 0.1, x)),mask_people)))
        return key, mask_people

    def Process(self, img_RGB):
        # try:
        if(1):
            img_keypoints, img_mask = self._detected(img_RGB)
            nose, left_ear, right_ear, left_shoulder, right_shoulder = img_keypoints[0], img_keypoints[4], img_keypoints[3], \
                                                                       img_keypoints[6], img_keypoints[5]
            left_hip, right_hip, left_knee, right_knee = img_keypoints[12], img_keypoints[11], img_keypoints[14], \
                                                         img_keypoints[13]
            # return img_keypoints
            y_hip = (left_hip[1] + right_hip[1]) / 2
            y_knee = (left_knee[1] + right_knee[1]) / 2

            center_shoulder = (left_shoulder + right_shoulder) / 2
            y_waist = y_hip * 2 / 3 + (nose[1] + center_shoulder[1]) / 6
            left_thigh = (left_knee + left_hip) / 2
            right_thigh = (right_knee + right_hip) / 2

            # estimate the waist width
            waist_width = self.waist_width_estimate(center_shoulder, y_waist, img_mask)
            # estimate the thigh width
            thigh_width = self.thigh_width_estimate(left_thigh, right_thigh, img_mask)
            # estimate the hip width
            hip_width = self.hip_width_estimate(center_shoulder, y_hip, img_mask)
            # estimate the head_width
            head_width = self.head_width_estimate(left_ear, right_ear)
            # estimate the Area
            Area = self.Area_estimate(y_waist, y_hip, waist_width, hip_width, img_mask)
            # estimate the height2waist
            height = self.Height_estimate(y_knee, nose[1])
            # estimate tht shoulder_width
            shoulder_width = self.shoulder_width_estimate(left_shoulder, right_shoulder)

            F = Body_Figure(waist_width, thigh_width, hip_width, head_width, Area, height, shoulder_width)

            img = self.Img_Trans(img_RGB)
            xs = torch.squeeze(self.exT(img).detach()).cpu().numpy()
            #
            values = []
            values.append(F.WSR)
            values.append(F.WTR)
            values.append(F.WHpR)
            values.append(F.WHdR)
            values.append(F.HpHdR)
            values.append(F.Area)
            values.append(F.H2W)
            for x in xs:
                values.append(x)


            values = np.expand_dims(np.asarray(values), axis=0)
            values[np.isnan(values)]=0
            values[np.isinf(values)]=0
            BMI = self.regR.predict(values)[0]

            return img_keypoints, img_mask,BMI

        # except:
        #     return None,None,23.33

    def Height_estimate(self, y_k, y_n):
        Height = np.abs(y_n - y_k)
        return Height

    def Area_estimate(self, y_w, y_h, W_w, H_w, mask):
        '''
            Area is expressed as thenumber of
            pixels per unit area between waist and hip
        '''
        pixels = np.sum(mask[int(y_w):int(y_h)][:])
        area = (y_h - y_w) * 0.5 * (W_w + H_w)
        Area = pixels / area
        return Area

    def shoulder_width_estimate(self, left_shoulder, right_shoulder):
        shoulder_width = np.sqrt(
            (right_shoulder[0] - left_shoulder[0]) ** 2 + (right_shoulder[1] - left_shoulder[1]) ** 2)
        return shoulder_width

    def head_width_estimate(self, left_ear, right_eat):
        head_width = np.sqrt((right_eat[0] - left_ear[0]) ** 2 + (right_eat[1] - left_ear[1]) ** 2)
        return head_width

    def hip_width_estimate(self, center_shoulder, y_hip, img_mask):
        x_hip_center = int(center_shoulder[0])
        x_lhb = np.where(img_mask[int(y_hip)][:x_hip_center] < 0.1)[0]
        x_lhb = x_lhb[-1] if len(x_lhb) else 0
        x_rhb = np.where(img_mask[int(y_hip)][x_hip_center:] < 0.1)[0]
        x_rhb = x_rhb[0] + x_hip_center if len(x_rhb) else len(img_mask[0])
        hip_width = x_rhb - x_lhb
        return hip_width

    def thigh_width_estimate(self, left_thigh, right_thigh, mask):
        lx, ly = int(left_thigh[0]), int(left_thigh[1])
        rx, ry = int(right_thigh[0]), int(right_thigh[1])

        x_ltb = np.where(mask[ly][:lx] < 0.1)[0]
        x_ltb = x_ltb[-1] if len(x_ltb) else 0
        x_rtb = np.where(mask[ry][rx:] < 0.1)[0]
        x_rtb = x_rtb[0] + rx if len(x_rtb) else len(mask[0])

        l_width = (lx - x_ltb) * 2
        r_width = (x_rtb - rx) * 2

        thigh_width = (l_width + r_width) / 2
        return thigh_width

    def waist_width_estimate(self, center_shoulder, y_waist, img_mask):
        x_waist_center = int(center_shoulder[0])
        # plt.imshow(img_mask)
        # plt.show()
        x_lwb = np.where(img_mask[int(y_waist)][:x_waist_center] < 0.1)[0]
        x_lwb = x_lwb[-1] if len(x_lwb) else 0
        x_rwb = np.where(img_mask[int(y_waist)][x_waist_center:] < 0.1)[0]
        x_rwb = x_rwb[0] + x_waist_center if len(x_rwb) else len(img_mask[0])
        # print(x_rwb)
        waist_width = x_rwb - x_lwb
        return waist_width

    def Vis(self, img):
        pass
