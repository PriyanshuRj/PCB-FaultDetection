import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import tensorflow as tf
Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
def attempt_download(file):
    print('attempt herrrre')
    # Attempt file download if does not exist
    file = Path(str(file).strip().replace("'", ''))
    return str(file)

class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights=''):
        super().__init__()
        w = weights
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        w = attempt_download(w)  # download if not local
        
        print(f'Loading {w} for TensorFlow Lite inference...')
        interpreter = Interpreter(model_path=w)  # load TFLite model
        interpreter.allocate_tensors()  # allocate
        input_details = interpreter.get_input_details()  # inputs
        output_details = interpreter.get_output_details()  # outputs
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width  
        im = im.permute(0, 2, 3, 1).cpu().numpy()  
        input, output = self.input_details[0], self.output_details[0]
        int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model

        if int8:
            print('qwiheucvyigiusegucsvnesuovos\n\n\n')
            scale, zero_point = input['quantization']
            im = (im / scale + zero_point).astype(np.uint8)  # de-scale
        self.interpreter.set_tensor(input['index'], im)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(output['index'])

        if int8:
            scale, zero_point = output['quantization']
            y = (y.astype(np.float32) - zero_point) * scale  # re-scale
        y[..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=self.device)
        return (y, []) if val else y

class Detect():

    def __init__(self, weights='', device=torch.device('cpu')):
        super().__init__()
        w = weights
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        w = attempt_download(w)  # download if not local

        
        print(f'Loading {w} for TensorFlow Lite inference...')
        interpreter = Interpreter(model_path=w)  # load TFLite model
        interpreter.allocate_tensors()  # allocate
        input_details = interpreter.get_input_details()  # inputs
        output_details = interpreter.get_output_details()  # outputs
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):

        b, ch, h, w = im.shape  # batch, channel, height, width
        print(im)

        # im = torch.tensor(im)
        # print(im)
        im = np.transpose(im,(0, 2, 3, 1))
        # im = im.permute(0, 2, 3, 1).cpu().numpy()
        # print(im)
        input, output = self.input_details[0], self.output_details[0]
        int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
        self.interpreter.set_tensor(input['index'], im)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(output['index'])
        y[..., :4] *= [w, h, w, h]  # xywh normalized to pixels
    
        # if isinstance(y, np.ndarray):
        #     y = torch.tensor(y,device='cpu')
    
        return (y, []) if val else y
