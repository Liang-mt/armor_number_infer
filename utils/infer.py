import cv2
import torch
import onnxruntime
import numpy as np
from utils import poly_postprocess, min_rect, ValTransform, demo_postprocess_armor, demo_postprocess_buff

class infer(object):
    def __init__(
        self,
        onnx_model_path,
        num_apex,
        num_class,
        num_color,
        device="cpu",
        legacy=False,
    ):
        self.session = onnxruntime.InferenceSession(onnx_model_path)
        self.num_apexes = num_apex
        self.num_classes = num_class
        self.num_colors = num_color
        self.confthre = 0.25  # conf
        self.nmsthre = 0.3    # nms
        self.test_size = (416, 416)
        self.device = device
        self.preproc = ValTransform(legacy=legacy)
        self.tracker = None  # 缩放跟踪器
        self.color_id = ["B","R","N","P"]
        self.cls_id = ["B","1","2","3","4","5","G","O","base"]
    def inference(self, img):
        img_info = {}

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float().numpy()

        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        feed_dict = {input_name: img}

        outputs = self.session.run([output_name], input_feed=feed_dict)

        if self.num_apexes == 4:
            outputs = demo_postprocess_armor(outputs[0], self.test_size, p6=False)[0]
        elif self.num_apexes == 5:
            outputs = demo_postprocess_buff(outputs[0], self.test_size, p6=False)[0]

        bbox_preds = []
        for i in range(outputs.shape[0]):
            bbox = min_rect(outputs[i, :, :self.num_apexes * 2])
            bbox_preds.append(bbox)

        bbox_preds = torch.stack(bbox_preds)

        conf_preds = outputs[:, :, self.num_apexes * 2].unsqueeze(-1)

        cls_preds = outputs[:, :, self.num_apexes * 2 + 1 + self.num_colors:].repeat(1, 1, self.num_colors)
        colors_preds = torch.clone(cls_preds)

        for i in range(self.num_colors):
            colors_preds[:, :, i * self.num_classes:(i + 1) * self.num_classes] = outputs[:, :,
                                                                                  self.num_apexes * 2 + 1 + i:self.num_apexes * 2 + 1 + i + 1].repeat(
                1, 1, self.num_classes)

        cls_preds_converted = (colors_preds + cls_preds) / 2.0

        outputs_rect = torch.cat((bbox_preds, conf_preds, cls_preds_converted), dim=2)
        outputs_poly = torch.cat((outputs[:, :, :self.num_apexes * 2], conf_preds, cls_preds_converted), dim=2)
        outputs = poly_postprocess(
            outputs_rect,
            outputs_poly,
            self.num_apexes,
            self.num_classes * self.num_colors,
            self.confthre,
            self.nmsthre
        )
        return outputs, img_info

    def visual(self, output, img_info, conf=0.35):
        detections = []
        ratio = img_info["ratio"]
        img = img_info["raw_img"]

        if output is None:
            return detections

        output = output.cpu()

        boxes = output[:, 0:self.num_apexes * 2]
        boxes /= ratio

        cls_ids = output[:, self.num_apexes * 2 + 2]
        scores = output[:, self.num_apexes * 2] * output[:, self.num_apexes * 2 + 1]

        for i in range(len(boxes)):
            d = boxes[i]
            if self.num_apexes == 4:
                pt0 = (int(d[0]), int(d[1]))
                pt1 = (int(d[2]), int(d[3]))
                pt2 = (int(d[4]), int(d[5]))
                pt3 = (int(d[6]), int(d[7]))
                position = np.array([pt0, pt1, pt2, pt3], dtype=np.float32).reshape(4, 2)  # 关键修改
            elif self.num_apexes == 5:
                pt0 = (int(d[0]), int(d[1]))
                pt1 = (int(d[2]), int(d[3]))
                pt2 = (int(d[4]), int(d[5]))
                pt3 = (int(d[6]), int(d[7]))
                pt4 = (int(d[8]), int(d[9]))
                position = np.array([pt0, pt1, pt2, pt3, pt4], dtype=np.float32).reshape(5, 2)

            cls_id = int(cls_ids[i])
            color,cls = self.get_color_and_tag(cls_id)
            color = self.color_id[color]
            cls = self.cls_id[cls]
            #cls = class_names[cls_id]
            score = scores[i]
            score = round(score.item(), 2)
            if score < conf:
                continue
            detections.append({'color': color, 'cls': cls, 'conf': score, 'position': position})

        return detections

    def detect(self, frame):
        outputs, img_info = self.inference(frame)
        detections = self.visual(outputs[0], img_info, self.confthre)

        original_frame = self.draw_detections(frame.copy(), detections, (0, 255, 0))

        return original_frame, detections

    def draw_detections(self, frame, detections, color=(0, 255, 0)):
        for d in detections:
            pts = np.array(d['position'], dtype=np.int32)
            cv2.polylines(frame, [pts], True, color, 2)
            cv2.putText(frame, f"{d['color']}{d['cls']} {d['conf']}",
                        (pts[0][0],pts[0][1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def get_color_and_tag(self,label):
        # 根据规则计算 color 和 tag
        color = label // 9
        tag = label % 9
        return color, tag