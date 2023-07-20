import json
from pprint import pprint
from google.cloud import storage

import tensorflow as tf

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
    model = get_model("model_resnet34_6b_ns")

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
    # Instantiate the writer.
    # PATCH_WIDTH , PATCH_HEIGHT = [128,128]
    # outputImageFile = f'gs://{bucket_name}/{folder_name}/{NAME_OUT}.TFRecord'
    # writer = tf.io.TFRecordWriter(outputImageFile)
    # curPatch = 1
    # for  prediction in predictions:
    #   patch = prediction.squeeze().T.flatten().tolist()

    #   if (len(patch) == PATCH_WIDTH * PATCH_HEIGHT):
    #     print('Done with patch ' + str(curPatch) + '...')
    #     # Create an example
    #     example = tf.train.Example(
    #        features=tf.train.Features(
    #         feature={
    #           'ld_prob': tf.train.Feature(
    #               float_list=tf.train.FloatList(
    #                   value=patch))
    #         }
    #       )
    #     )

    #     writer.write(example.SerializeToString())
    #     curPatch += 1
    # writer.close()


    # # REPLACE WITH YOUR USERNAME:
    # USER_NAME = 'ryali93'
    # outputAssetID = 'users/' + USER_NAME + '/ld_out'
    # print('Writing to ' + outputAssetID)

    # !earthengine upload image --asset_id={outputAssetID} {outputImageFile} {jsonFile}
    # Translate this line to python


    

    # Upload to GCS
    # client = storage.Client()
    # bucket = client.get_bucket(bucket_name)
    # blob = bucket.blob(f"{folder_name}/predictions.tfrecord")
    # blob.upload_from_filename("predictions.tfrecord")

    # Upload to GEE
    # Upload the predictions to Earth Engine.
    # image = ee.Image('users/ryali93/ld_out')
    # predictions = ee.Image('users/ryali93/ld_out_predictions')
    # predictions = predictions.addBands(image.select('B15'))
    # predictions = predictions.rename(['B2', 'B3', 'B4', 'B13', 'B14', 'B15'])
    # predictions = predictions.multiply(10000).int16()
    # task = ee.batch.Export.image.toAsset(
    #     image=predictions,
    #     description='Image Export',
    #     assetId='users/ryali93/ld_out_predictions',
    #     region=mixer['patchDimensions'],
    #     scale=10,
    #     maxPixels=1e13
    # )
    # task.start()

if __name__ == "__main__":
    main()

#     # Instantiate the writer.
#     PATCH_WIDTH , PATCH_HEIGHT = [128,128]
#     outputImageFile = 'gs://' + outputBucket + '/tesis/ld_out.TFRecord'
#     writer = tf.io.TFRecordWriter(outputImageFile)
#     curPatch = 1
#     for  prediction in predictions:
#       patch = prediction.squeeze().T.flatten().tolist()

#       if (len(patch) == PATCH_WIDTH * PATCH_HEIGHT):
#         print('Done with patch ' + str(curPatch) + '...')
#         # Create an example
#         example = tf.train.Example(
#           features=tf.train.Features(
#             feature={
#               'crop_prob': tf.train.Feature(
#                   float_list=tf.train.FloatList(
#                       value=patch))
#             }
#           )
#         )

#         writer.write(example.SerializeToString())
#         curPatch += 1
#     writer.close()

# !gsutil ls -l {outputImageFile}

# # REPLACE WITH YOUR USERNAME:
# USER_NAME = 'ryali93'
# outputAssetID = 'users/' + USER_NAME + '/ld_out'
# print('Writing to ' + outputAssetID)

# # Start the upload. It step might take a while.
# !earthengine upload image --asset_id={outputAssetID} {outputImageFile} {jsonFile}

