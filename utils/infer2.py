import math

import cv2
import torch
import onnxruntime
import numpy as np
from utils import poly_postprocess, min_rect, ValTransform, demo_postprocess_armor, demo_postprocess_buff
from utils.mlp_predict import number_cls

class infer2(object):
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
        self.cls_name = ['B','R','O']
        self.number_id = ["1", "2", "3", "4", "5", "outpost", "guard", "base", "negative"]
        self.number = number_cls("./model/mlp.onnx")
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
            # color = self.color_id[color]
            # cls = self.cls_id[cls]s
            cls = self.cls_name[cls_id]
            score = scores[i]
            score = round(score.item(), 2)
            if score < conf:
                continue
            detections.append({'cls': cls, 'conf': score, 'position': position})

        return detections

    def detect(self, frame):
        outputs, img_info = self.inference(frame)
        detections = self.visual(outputs[0], img_info, self.confthre)

        original_frame = self.draw_detections(frame.copy(), detections, (0, 255, 0))

        return original_frame, detections

    def calculate_pdistance(self,pt1, pt2):
        """
        计算两点之间的距离和中心点
        :param pt1: 第一个点，格式为 (x1, y1)
        :param pt2: 第二个点，格式为 (x2, y2)
        :return: 包含距离和中心点的元组 (length, center)
        """
        length = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
        center = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)
        return length, center

    def is_big_armor(self,pts):
        """
        判断是否为大装甲板
        :param pts: 装甲板的四个点，格式为 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        :return: True 表示是大装甲板，False 表示不是
        """
        # 假设 pts 顺序为：左灯条底部、左灯条顶部、右灯条顶部、右灯条底部
        length1, center1 = self.calculate_pdistance(pts[0], pts[1])
        length2, center2 = self.calculate_pdistance(pts[3], pts[2])

        # 计算平均灯条长度
        avg_light_length = (length1 + length2) / 2
        # 计算灯条中心之间的距离并除以平均灯条长度
        center_distance = math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) / avg_light_length

        # 阈值 3.2 可根据实际情况调整
        return center_distance > 3.2
    def process_roi(self, raw_img, pts, num_apexes):
        # 定义目标顶点
        light_length = 12
        warp_height = 28
        small_armor_width = 32
        large_armor_width = 54
        roi_size = (20, 28)

        top_light_y = (warp_height - light_length) // 2 - 1
        bottom_light_y = top_light_y + light_length
        warp_width = large_armor_width if self.is_big_armor(pts) else small_armor_width
        target_vertices = np.array([
            [0, bottom_light_y],
            [0, top_light_y],
            [warp_width - 1, top_light_y],
            [warp_width - 1, bottom_light_y]
        ], dtype=np.float32)

        # 确保传入的源点是 4 个
        src_pts = pts[:4] if len(pts) >= 4 else pts
        if len(src_pts) != 4:
            print(f"Error: src_pts length is {len(src_pts)}, expected 4.")
            return None

        # 确保源点是 float32 类型且维度正确
        src_pts = np.array(src_pts, dtype=np.float32).reshape(-1, 2)

        # 透视变换
        try:
            rotation_matrix = cv2.getPerspectiveTransform(src_pts, target_vertices)
            number_image = cv2.warpPerspective(raw_img, rotation_matrix, (warp_width, warp_height))

            # 获取 ROI
            x = (warp_width - roi_size[0]) // 2
            y = 0
            number_image = number_image[y:y + roi_size[1], x:x + roi_size[0]]

            # 二值化
            number_image = cv2.cvtColor(number_image, cv2.COLOR_RGB2GRAY)
            _, number_image = cv2.threshold(number_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # 将ROI图像顺时针旋转90度
            # number_image = cv2.rotate(number_image, cv2.ROTATE_180)
            # number_image = cv2.flip(number_image, 1)


            return number_image
        except cv2.error as e:
            print(f"Error in getPerspectiveTransform: {e}")
            return None

    def draw_detections(self, frame, detections, color=(0, 255, 0)):
        # 进行ROI处理并显示
        rois = []
        for d in detections:
            pts = np.array(d['position'], dtype=np.int32)
            #roi 点的顺序是  左下 左上，右上，右下
            pts2 = [pts[1],pts[0],pts[3],pts[2]]
            roi = self.process_roi(frame, pts2, self.num_apexes)
            test = self.number.predict(roi)
            num_id = self.number_id[test['class_index']]
            if roi is not None:
                rois.append(roi)
                #cv2.imshow(f"ROI_{len(rois)}", roi)
            cv2.imshow(f"ROI", roi)
            cv2.putText(frame, f"Detections: {len(detections)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.polylines(frame, [pts], True, color, 2)
            cv2.putText(frame, f"{d['cls']}{num_id} {d['conf']}",(pts[0][0],pts[0][1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def get_color_and_tag(self,label):
        # 根据规则计算 color 和 tag
        color = label // 9
        tag = label % 9
        return color, tag