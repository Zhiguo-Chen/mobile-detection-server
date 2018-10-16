import tensorflow as tf
from flask import Flask, request, jsonify
from threading import Thread
from six.moves import _thread
from src.utils.predicting import PredictorNetwork
from src.tools.checkpoints import get_chekpoint_config


HOST = '127.0.0.1'
PORT = 5200


app = Flask(__name__)
print('app')


@app.route('/api/fasterrcnn/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        print('request coming GET')
        return jsonify({'key': 'value'})

    if request.method == 'POST':
        print('request coming POST')
        return jsonify({'_key': '_value'})


def start_network(config=None):
    global PREDICT_NETWORK
    try:
        PREDICT_NETWORK = PredictorNetwork(config)
    except ImportError as error:
        print(error.path)
    except Exception as e:
        tf.logging.error(e)
        _thread.interrupt_main()


def start(argv=None):

    print('Hello World')
    config = get_chekpoint_config('accurate')
    global NETWORK_START_THREAD
    NETWORK_START_THREAD = Thread(target=start_network, args=(config,))
    NETWORK_START_THREAD.start()
    app.run(host=HOST, port=PORT, debug=True)
