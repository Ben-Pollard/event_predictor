from flask import Flask, jsonify, request
from Exceptions import UnprocessableEntity
from sklearn.externals import joblib
import pandas as pd
from Model import model

app = Flask(__name__)

@app.errorhandler(UnprocessableEntity)
def handle_unprocessable_entity(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route('/', methods=['POST'])
@app.route('/index', methods=['POST'])
@app.route('/test', methods=['POST'])
def test():
    return 'Hello world!'

@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.json

    #Check for presence of required fields
    reqDiffAct = set(required_fields) - set(json_.keys())
    if len(reqDiffAct) > 0:
        raise UnprocessableEntity('Missing fields: %s' % reqDiffAct, status_code=422)

    #Check types
    #Infill dummies

    df = pd.DataFrame({k:[v] for k,v in json_.items()})[required_fields]
    prediction = event_predictor.score(json_)

    return jsonify({'prediction':prediction})


if __name__ == '__main__':
    scorObj = joblib.load('Data/scoring_objects.pkl')
    required_fields = scorObj['required_fields']
    event_predictor = model(scorObj['rfr_search'],
                            scorObj['memberships_per_group'],
                            scorObj['maxRsvpLimit'],
                            scorObj['w2vModel'],
                            scorObj['vecSize'])

    app.run(host='0.0.0.0', port=80)