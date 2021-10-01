import numpy as np
from django.conf import settings
from django.core.files.storage import default_storage
from django.shortcuts import render
from keras.applications import inception_v3
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.python.keras.backend import set_session
import json


with open('label.json', 'r') as f:
    labelInfo=f.read()
labelInfo = json.loads(labelInfo)

def index(request):
    if request.method == "POST":
        #
        # Django image API
        #
        file = request.FILES["imageFile"]
        file_name = default_storage.save(file.name, file)
        file_url = default_storage.path(file_name)
        
        #
        # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/load_img
        #
        image = load_img(file_url, target_size=(224, 224))
        numpy_array = img_to_array(image)
        image_batch = np.expand_dims(numpy_array, axis=0)
        processed_image = inception_v3.preprocess_input(image_batch.copy())

        #
        # get the predicted probabilities
        #
        with settings.GRAPH1.as_default():
            set_session(settings.SESS)
            predictions = settings.IMAGE_MODEL.predict(processed_image)
            print('**************')
            print(predictions)
            print('**************')
            print(np.argmax(predictions[0]))

        #
        # Output/Return data
        #
        # label = decode_predictions(predictions, top=10)
        label = labelInfo[str(np.argmax(predictions[0]))]
        return render(request, "index.html", {"predictions": label})

    else:
        return render(request, "index.html")
    
