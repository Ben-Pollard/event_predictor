from flask import Flask

app = Flask(__name__)

@app.route('/', methods=['POST'])
@app.route('/index', methods=['POST'])
@app.route('/test', methods=['POST'])
def test():
    return 'Hello world!'

@app.route('/predict', methods=['POST'])
def predict():
    return 'predict end-point'

if __name__ == '__main__':
     app.run(port=8080)