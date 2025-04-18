import cv2

from utils import poly_postprocess, vis, min_rect, ValTransform, demo_postprocess_armor, demo_postprocess_buff
from utils import infer, infer2

# if __name__ == "__main__":
#
#     video_path = "./video/15.mp4"
#     #video_path = "./前哨站/蓝方前哨站狙击点视角全速.mp4"
#
#     #onnx_model_path = "./model/500.onnx"
#     onnx_model_path = "./model/TUP/best_06_02.onnx"
#     #onnx_model_path = "./model/armor1000.onnx"
#     #onnx_model_path = "./model/train_1000.onnx"
#     #根据自己模型的不同可对关键点数量，颜色数量，类别数量进行相对应的修改
#     predictor = Predictor(onnx_model_path, num_apex = 4, num_class = 8,num_color = 8)
#
#     cap = cv2.VideoCapture(video_path)
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         outputs, img_info = predictor.inference(frame)
#         result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
#
#         cv2.imshow("Video", result_image)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下 'q' 键退出循环
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "./video/英雄_2.mp4"
    onnx_model_path = "./model/yolox.onnx"
    #onnx_model_path = "./model/og1000.onnx"
    #predictor = infer(onnx_model_path, num_apex=4, num_class=9, num_color=4)
    #predictor = infer(onnx_model_path, num_apex=5, num_class=2, num_color=2)
    predictor = infer2(onnx_model_path, num_apex=4, num_class=1, num_color=3)

    cap = cv2.VideoCapture(video_path)

    # Define the codec and create a VideoWriter object
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output_video.avi', fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result_image, detections = predictor.detect(frame)
        # Save the modified frame
        out.write(result_image)

        cv2.imshow("Video", result_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()



