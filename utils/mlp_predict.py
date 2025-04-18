import onnxruntime as ort
import numpy as np
import cv2


class number_cls(object):
    def __init__(self, model_path):
        # 初始化ONNX Runtime会话
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name



    def preprocess(self, number_image):
        """
        输入要求：来自process_roi输出的二值化图像（20x28）
        输出格式：符合模型输入的张量（1x20x28x1）
        """
        # 类型转换与归一化
        if number_image.dtype != np.float32:
            number_image = number_image.astype(np.float32) / 255.0

        # 添加通道和批次维度
        return number_image.reshape(1, 20, 28, 1)

    def predict(self, number_image):
        # 数据预处理
        if number_image is None or number_image.size == 0:
            raise ValueError("输入图像为空")

        input_data = self.preprocess(number_image)

        # 执行推理
        outputs = self.session.run([self.output_name],
                                   {self.input_name: input_data})

        # 解析结果
        pred_probs = outputs[0][0]  # 获取第一个batch结果
        predicted_class = np.argmax(pred_probs)

        return {
            "class_index": int(predicted_class),
            "confidence": float(pred_probs[predicted_class])
        }


# 使用示例
if __name__ == "__main__":
    # 初始化识别器
    recognizer = n("mlp_2.onnx")

    # 模拟process_roi输出（20x28二值化图像）
    dummy_image = np.random.randint(0, 255, (20, 28), dtype=np.uint8)
    _, test_image = cv2.threshold(dummy_image, 127, 255, cv2.THRESH_BINARY)

    # 执行推理
    result = recognizer.predict(test_image)
    print(f"预测结果: 类别{result['class_index']} 置信度{result['confidence']:.2f}")