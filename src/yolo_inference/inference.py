import cv2
import numpy as np
import paddle.inference as inference

class Inference(object):
    def __init__(self, model_path="./inference_model/model.pdmodel", param_path="./inference_model/model.pdiparams", crop_size=[512, 512], k_top=2):
        self.config = inference.Config(model_path, param_path)
        self.predictor = inference.create_predictor(self.config)
        self.crop_size = crop_size
        self.k_top = k_top

    def predict(self, src_img):
        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])
        input_handle.copy_from_cpu(np.array([self.crop_size, ]))
        input_handle = self.predictor.get_input_handle(input_names[1])
        input_handle.copy_from_cpu(handle(src_img, self.crop_size))
        input_handle = self.predictor.get_input_handle(input_names[2])
        input_handle.copy_from_cpu(np.array([[1, 1], ]))
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])

        self.predictor.run()
        output_data = output_handle.copy_to_cpu()

        return output_data[:self.k_top]


def normalize(src_img, mean, std):
    src_img = src_img.astype(np.float32, copy=False)
    mean = np.array(mean)[np.newaxis, np.newaxis, :]
    std = np.array(std)[np.newaxis, np.newaxis, :]
    src_img = src_img / 255.0
    src_img -= mean
    src_img /= std

    return src_img

def handle(src_img, crop_size):
    src_img = cv2.resize(src_img, (crop_size[0], crop_size[1]))
    src_img = normalize(src_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    src_img = src_img.transpose([2, 0, 1])
    tensor_img = src_img[None, :].astype("float32")

    return tensor_img