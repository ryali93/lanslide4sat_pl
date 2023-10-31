import json
from pprint import pprint
from google.cloud import storage
import torch

from ee_utils import *

def get_model(model_path):
    # model_path = "model_resnet34_6b_ns"
    with open(f"models/{model_path}.json", "r") as json_file:
        json_config = json_file.read()
    model = tf.keras.models.model_from_json(json_config)
    model.load_weights(f"models/{model_path}.h5")
    
    return model

def main():
    # 1. Read model
    model = get_model("model_vanilla_6b_l4s")

    bucket_name = 'rgee_dev'
    folder_name = 'tesis'
    imageFilePrefix = f'{folder_name}/ld'
    filesList = list_blobs(bucket_name, folder_name)

    exportFilesList = [f"gs://{bucket_name}/{s}" for s in filesList if imageFilePrefix in s]
    imageFilesList = []
    jsonFile = None
    for f in exportFilesList:
      if f.endswith('.tfrecord.gz'):
        imageFilesList.append(f)
      elif f.endswith('.json'):
        jsonFile = f

    json_text = None
    with tf.io.gfile.GFile(jsonFile, 'r') as f:
      json_text = f.read()
      mixer = json.loads(json_text)

    # Parsing the mixer file
    fileNames = imageFilesList
    side = 128
    bands = ['B2', 'B3', 'B4', 'B13', 'B14', 'B15']
    predict_db = predict_input_fn(fileNames=fileNames, side=side, bands=bands)

    # Predicting the data
    predictions = model.predict(predict_db)
    print(predictions.shape)

    # Upload to GCS
    USER_NAME = 'ryali93'
    NAME_OUT = 'ld_out_predictions'
    outputAssetID = f'users/{USER_NAME}/{NAME_OUT}'
    print('Writing to ' + outputAssetID)

if __name__ == "__main__":
    main()
