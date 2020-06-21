import sys

from google.cloud import automl_v1beta1
from google.cloud.automl_v1beta1.proto import service_pb2


def get_prediction(content, project_id, model_id):
    prediction_client = automl_v1beta1.PredictionServiceClient()
    name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
    payload = {'text_snippet': {'content': content}}
    params = {}
    request = prediction_client.predict(name, payload, params)
    translated_content = request.payload[0].translation.translated_content
    result=u"\"Translated_content\": \"{0}\"".format(translated_content.content)
    return result  # waits till request is returned

def setup_prediction(file_path, project_id, model_id):
    with open(file_path, 'rb') as ff:
        content = ff.read()
    return get_prediction(content, project_id,  model_id)

if __name__ == '__main__':
  file_path = sys.argv[1]
  project_id = sys.argv[2]
  model_id = sys.argv[3]

  with open(file_path, 'rb') as ff:
    content = ff.read()

  print get_prediction(content, project_id,  model_id)

