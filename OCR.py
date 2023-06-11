import os
import subprocess
from itertools import groupby
from pathlib import Path
import cv2
import numpy as np
from openvino.runtime import Core
import sys


from easygoogletranslate import EasyGoogleTranslate

def print_extracted_text(text):
    encoded_text = text.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
    print(encoded_text)

class Translator():

    def __init__(self):
        print("made a Translator")
        print("CWD", os.getcwd())
        #check if the models exsist
        self.model_folder="./model"
        self.data_folder="./data"
        self.charlist_folder = f"{self.data_folder}/text/"
        self.translator = EasyGoogleTranslate()
        self.precision = "FP16"
        #note this only checks the parent folder not the files themselves. 
        path1exsists=os.path.isdir("./model/intel/handwritten-japanese-recognition-0001")
        path2exsists=os.path.isdir("./model/intel/handwritten-simplified-chinese-recognition-0001")

        print(os.path.isdir(path1exsists))
        #note this requires openvino-dev to be installed
        if not path1exsists :
            command=f'omz_downloader --name handwritten-japanese-recognition-0001 --output_dir {self.model_folder} --precision {self.precision}'
            print(command)
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            print(result)

        if not path2exsists:
            print(command)
            command=f'omz_downloader --name handwritten-simplified-chinese-recognition-0001 --output_dir {self.model_folder} --precision {self.precision}'
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            print(result)

        #load and compile both models
        path_to_japanease_model_weights = Path(f'{self.model_folder}/intel/handwritten-japanese-recognition-0001/{self.precision}/handwritten-japanese-recognition-0001.bin')
        path_to_chinease_model_weights = Path(f'{self.model_folder}/intel/handwritten-simplified-chinese-recognition-0001/{self.precision}/handwritten-simplified-chinese-recognition-0001.bin')

        ie = Core()

        path_to_japanease_model = path_to_japanease_model_weights.with_suffix(".xml")
        path_to_chinease_model = path_to_chinease_model_weights.with_suffix(".xml")

        japanease_model = ie.read_model(model=path_to_japanease_model)
        chinease_model = ie.read_model(model=path_to_chinease_model)

        self.jp_ocr = ie.compile_model(model=japanease_model, device_name="CPU")
        self.cn_ocr = ie.compile_model(model=chinease_model, device_name="CPU")

        self.jp_recogn_out_layer = self.jp_ocr.output(0)
        self.jp_recog_in_layer = self.jp_ocr.input(0)

        self.cn_recogn_out_layer = self.cn_ocr.output(0)
        self.cn_recog_in_layer = self.cn_ocr.input(0)

        # set up the char lists 
        blank_char = "~"
        with open(f"{self.charlist_folder}/japanese_charlist.txt", "r", encoding="utf-8") as charlist:
            self.jp_letters = blank_char + "".join(line.strip() for line in charlist)
        with open(f"{self.charlist_folder}/chinese_charlist.txt", "r", encoding="utf-8") as charlist:
            self.cn_letters = blank_char + "".join(line.strip() for line in charlist)
    
    def extract_text_jp(self,img_data):
        image_height, _ = img_data.shape
        # B,C,H,W = batch size, number of channels, height, width.
        _, _, H, W = self.jp_recog_in_layer.shape

        # Calculate scale ratio between the input shape height and image height to resize the image.
        scale_ratio = H / image_height

        # Resize the image to expected input sizes.
        resized_image = cv2.resize(img_data, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_AREA)

        # Pad the image to match input size, without changing aspect ratio.
        resized_image = np.pad(resized_image, ((0, 0), (0, W - resized_image.shape[1])), mode="edge")

        # Reshape to network input shape.
        input_image = resized_image[None, None, :, :]
        predictions = self.jp_ocr([input_image])[self.jp_recogn_out_layer]
        predictions = np.squeeze(predictions)

        # Run the `argmax` function to pick the symbols with the highest probability.
        predictions_indexes = np.argmax(predictions, axis=1)
        # Use the `groupby` function to remove concurrent letters, as required by CTC greedy decoding.
        output_text_indexes = list(groupby(predictions_indexes))

        # Remove grouper objects.
        output_text_indexes, _ = np.transpose(output_text_indexes, (1, 0))

        # Remove blank symbols.
        output_text_indexes = output_text_indexes[output_text_indexes != 0]

        # Assign letters to indexes from the output array.
        output_text = [self.jp_letters[letter_index] for letter_index in output_text_indexes]

        
        self.translator = EasyGoogleTranslate(
            source_language='jp',
            target_language='en',
            timeout=30
        )
        result = self.translator.translate(''.join(output_text))
        print_extracted_text(''.join(output_text))
        return result

    def extract_text_cn(self,img_data):
        image_height, _ = img_data.shape
        # B,C,H,W = batch size, number of channels, height, width.
        _, _, H, W = self.cn_recog_in_layer.shape

        # Calculate scale ratio between the input shape height and image height to resize the image.
        scale_ratio = H / image_height

        # Resize the image to expected input sizes.
        resized_image = cv2.resize(img_data, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_AREA)

        # Pad the image to match input size, without changing aspect ratio.
        resized_image = np.pad(resized_image, ((0, 0), (0, W - resized_image.shape[1])), mode="edge")

        # Reshape to network input shape.
        input_image = resized_image[None, None, :, :]
        predictions = self.cn_ocr([input_image])[self.cn_recogn_out_layer]
        predictions = np.squeeze(predictions)

        # Run the `argmax` function to pick the symbols with the highest probability.
        predictions_indexes = np.argmax(predictions, axis=1)
        # Use the `groupby` function to remove concurrent letters, as required by CTC greedy decoding.
        output_text_indexes = list(groupby(predictions_indexes))

        # Remove grouper objects.
        output_text_indexes, _ = np.transpose(output_text_indexes, (1, 0))

        # Remove blank symbols.
        output_text_indexes = output_text_indexes[output_text_indexes != 0]

        # Assign letters to indexes from the output array.
        output_text = [self.cn_letters[letter_index] for letter_index in output_text_indexes]

        
        self.translator = EasyGoogleTranslate(
            source_language='cn',
            target_language='en',
            timeout=30
        )
        result = self.translator.translate(''.join(output_text))
        print_extracted_text(''.join(output_text))
        return result