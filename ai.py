"""
Place to contain the ai computer vision
"""
import cv2
import torch
import timm

import numpy as np
import re
import requests
from logger import setup_logger

logger = setup_logger('SmartGV AI')


# TODO: model to recognize different sorts
# TODO: model to revognize stages of growth
# TODO: extract any other data somehow from the image

class Classifier:
    def __init__(self, model: str, classes: str):
        """
               Initialize the Classifier.

               Args:
                   model (str): Path to the saved PyTorch model (.pth file).
                   classes (str): Path to the text file containing class names, one per line.

               Sets up the device, loads class names and model.
        """

        self.model_path = model
        self.classes_names_path = classes
        self.image_size = 224
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.classes = None
        self.model = None
        self.prediction = None

        self._load_classes()
        self._load_model()

        logger.debug(f'Initialized classifier with {self.device}')

    def _load_classes(self):
        with open(self.classes_names_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        logger.debug(f'Loaded classes for {self.model_path}')

    def _load_model(self):
        self.model = timm.create_model("mobilenetv3_small_050", pretrained=False, num_classes=len(self.classes))
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        logger.debug('Loaded model')

    def get_prediction(self):
        """
               Return the predicted class name of the last processed image.

               Raises:
                   Exception: If no image has been processed yet.

               Returns:
                   str: Predicted class name.
        """
        if self.prediction:
            logger.debug(f'Got predictions {self.prediction}')
            return self.prediction
        else:
            logger.error('process_image() must be called first!')
            raise Exception('process_image() must be called first!')

    def process_image(self, img: cv2.imread):
        """
               Preprocess an input image and run the model to get a prediction.

               Args:
                   img (numpy.ndarray): Image loaded using OpenCV (BGR format).

               Steps:
                   - Converts BGR to RGB.
                   - Resizes to model input size.
                   - Normalizes pixel values to [0,1].
                   - Converts to PyTorch tensor with batch dimension.
                   - Runs inference with the model.
                   - Stores the predicted class name in self.prediction.

               Prints the predicted class.
        """

        logger.debug('Processing the image...')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        img_tensor = torch.tensor(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)
            pred_index = output.argmax(1).item()
            pred_class = self.classes[pred_index]

        logger.debug(f"Predicted class: {pred_class}")
        self.prediction = pred_class


class ModelHandler:
    STATUS_PLANT_TYPE = 0
    TEST = 1

    # TODO: add more such predefined models like that in future in case they need handling

    def __init__(self):

        self.purpose = None
        self.handlers = {
            self.STATUS_PLANT_TYPE: self._handle_spt,
            self.TEST: self._test,
        }

    def set_purpose(self, purpose: int):
        """
                Set the current purpose for the handler.

                Args:
                    purpose (int): An integer representing the purpose. Must correspond
                                   to one of the predefined status constants of this class.

                This value controls which handler method is called in the result() method.
        """
        self.purpose = purpose
        logger.debug(f'Purpose is set to {self.purpose}')

    def result(self, arg=None):
        """
                Execute the handler method corresponding to the current purpose.

                Args:
                    arg (any, optional): Optional argument to pass to the handler.

                Returns:
                    Depends on the handler method called.

                Raises:
                    ValueError: If purpose is not set or unknown.

                Calls the handler associated with self.purpose, passing arg if provided.
        """
        if self.purpose is None:
            logger.error('Purpose is not set! Please use .set_purpose() before using .result().')
            raise ValueError('Purpose is not set! Please use .set_purpose() before using .result().')

        try:
            handler = self.handlers[self.purpose]
        except KeyError:
            logger.error(f"Unknown purpose: {self.purpose}")
            raise ValueError(f"Unknown purpose: {self.purpose}")

        # Call with arg if it’s not None, otherwise without args
        logger.debug('Calling handler function...')
        if arg is not None:
            return handler(arg)
        return handler()

    @staticmethod
    def _handle_spt(string: str):
        """
                Parse a CamelCase string into three components: status, plant, and type.

                Args:
                    string (str): A CamelCase string expected to have exactly three parts.

                Returns:
                    tuple: A tuple of three strings (status, plant, type_).

                Raises:
                    ValueError: If the input string does not have exactly three CamelCase components.
        """
        parts = re.findall(r'[A-Z][a-z]*', string)
        if len(parts) != 3:
            logger.debug(f'Expected 3 CamelCase components, got {len(parts)}: {parts}')
            raise ValueError(f'Expected 3 CamelCase components, got {len(parts)}: {parts}')
        status, plant, type_ = parts
        logger.debug('Finished handeling SPT')
        return status, plant, type_

    @staticmethod
    def _test():
        logger.debug('Calling a test method')
        return 'test method'


class Processor:
    models = {
        ModelHandler.STATUS_PLANT_TYPE: {
            'model_path': 'models/status_plant_type.pth',
            'classes_path': 'datasets/fruits_and_vegies/class_names.txt'
        }
    }

    # TODO: add other model data here

    def __init__(self):
        self.input = None  # image to work with
        self.data = {
            'Status': '',
            'Plant': '',
            'Type': '',
            'Subtype': '',  # requires to be dedicated model for that
            'Eatable': '',
            'Time till fully grown': '',  # might need a growth determining model
            'Time past eatable': '',
            'Recommendations': '',  # maybe try to use again wikipedia to get some
            'Summary': ''
            # TODO: add more in future
        }

        self.model_handler = ModelHandler()
        pass

    def read_img(self, img_path: str):
        self.input = cv2.imread(img_path)

    def set_img(self, image: cv2.imread):
        self.input = image

    def run(self):
        """
            Executes the full processing pipeline on the loaded image.

            This includes running all relevant models, extracting predictions,
            using the model handler to interpret results, and populating the
            `self.data` dictionary with structured outputs.

            Currently supports STATUS_PLANT_TYPE prediction.
        """
        # here run all the models that are going to be ran on the image and apply all logic

        spt_classifier = self._create_classifier_and_set_handler(ModelHandler.STATUS_PLANT_TYPE)
        spt_classifier.process_image(self.input)
        spt = spt_classifier.get_prediction()

        #TODO: check how certain is the model and if it's below 60% announce it with a warning

        status, plant, type_ = self.model_handler.result(spt)
        self.data['Status'] = status
        self.data['Plant'] = plant
        self.data['Type'] = type_
        self.data['Eatable'] = 'Yes' if status == 'Fresh' else 'No'
        self.data['Summary'] = self.get_wikipedia_summary_for_plant()

        logger.debug('Collected all data so far')

        #TODO: here call the models in the same manner in order to fill the rest of the data.
        # Make sure to keep the dataset structure the same for different models

        # anotother_classifier = self._create_classifier_and_set_handler(ModelHandler.ANOTHER)
        # anotother_classifier.process_image(self.input)
        # another = anotother_classifier.get_prediction()

    def get_data(self):
        """
            Returns a dictionary of processed data fields with non-empty values only.

            This acts as a cleaner output of the processing pipeline,
            filtering out any unfilled/default values from `self.data`.

            Returns:
                dict: Dictionary containing only filled (non-empty) prediction data.
        """
        # handler for unfilled data
        filled_data = {k: v for k, v in self.data.items() if v != ''}
        logger.debug('Returning filled data')
        return filled_data

    def _create_classifier_and_set_handler(self, model: int):
        """
          Sets the appropriate handler in the model handler and creates a classifier instance.

          Args:
              model (int): The identifier for the type of model to use (must match keys in `Processor.models`).

          Returns:
              Classifier: An initialized classifier with the model and class names loaded.
        """
        logger.debug('Creating the classifier and setting the model handler purpose...')
        self.model_handler.set_purpose(model)
        model_path = Processor.models[model]['model_path']
        classes_path = Processor.models[model]['classes_path']
        return Classifier(model=model_path, classes=classes_path)

    def get_wikipedia_summary_for_plant(self):
        """
        Gets a summary from wikipedia about the given plant.
        NOTE: self.data['Plant'] has to be set first!
        :return:
        """
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{self.data['Plant'].replace(' ', '_')}"
        # print(f"Fetching: {url}")  # Debug URL
        response = requests.get(url)
        # print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            # print(data)  # Debug full response
            return data.get("extract")
        return None


if __name__ == '__main__':

    processor = Processor()
    processor.read_img('datasets/fruits_and_vegies/RottenTomatoVegetable/rottenTomato (1).jpg')
    processor.run()
    data = processor.get_data()
    print(data)

    # model_path = 'models/status_plant_type.pth'
    # class_names_path = 'datasets/fruits_and_vegies/class_names.txt'  # TODO to be moved into a variable that is reachable from everywhere
    # test_image = 'datasets/fruits_and_vegies/RottenTomatoVegetable/rottenTomato (1).jpg'
    # img = cv2.imread(test_image)
    #
    # classifier = Classifier(model=model_path, classes=class_names_path)
    #
    # classifier.process_image(img)
    # pred = classifier.get_prediction()
    # print(pred)
    #
    # mh = ModelHandler()
    # mh.set_purpose(ModelHandler.STATUS_PLANT_TYPE)
    # s, p, t = mh.result(pred)
    # print(f'Status: {s}\nPlant: {p}\nType: {t}')
    #
    # mh.set_purpose(ModelHandler.TEST)
    # print(mh.result())
