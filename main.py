# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import cv2
import numpy as np
import onnxruntime as ort


def read_img(image_path):
    load_size = [256, 256]
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, load_size).astype(np.float32)
    img /= 255.0
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    img = img[np.newaxis, ...]
    return img.astype(np.float32)


def init_model(model_path):
    sess_opt = ort.SessionOptions()
    sess_opt.log_severity_level = 4
    sess_opt.enable_cpu_mem_arena = False

    ort_session = ort.InferenceSession(model_path,
                                       sess_options=sess_opt,
                                       providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name
    return ort_session, input_name


def infer(ort_session, input_name):
    ort_inputs = {input_name: img.astype(np.float32)}
    result_img = ort_session.run(None, ort_inputs)[0]
    result_img = np.transpose(result_img.squeeze(), (1, 2, 0))
    result_img = (result_img + 1) / 2.0 * 255.0
    return result_img[..., ::-1]


if __name__ == '__main__':
    onnx_path = 'models/facades_label2photo_pretrained.onnx'
    ort_session, input_name = init_model(onnx_path)

    image_path = 'test_images/facades.jpg'
    img = read_img(image_path)

    img = infer(ort_session, input_name)
    cv2.imwrite('result.png', img)
    print('The inference is successful.')
