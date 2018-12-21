import tensorflow as tf
from flask import Flask, request, jsonify, current_app, make_response
from threading import Thread
from six.moves import _thread
from src.utils.predicting import PredictorNetwork
from src.tools.checkpoints import get_chekpoint_config
from PIL import Image
from datetime import timedelta
from functools import update_wrapper
from flask_cors import CORS

HOST = '127.0.0.1'
PORT = 5200


app = Flask(__name__)
# CORS(app)


def crossdomain(origin=None, methods=None, headers=None,
                max_age=21600, attach_to_all=True,
                automatic_options=True):
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, str):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, str):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers

            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator


@app.route('/api/<model_name>/predict/', methods=['GET', 'POST'])
@crossdomain(origin='*')
def predict(model_name):
    if request.method == 'GET':
        print('request coming GET')
        return jsonify({'key': 'value'})

    if request.method == 'POST':
        print('request coming POST')
        try:
            image_array = get_image()
        except ValueError:
            return jsonify(error='Missing image'), 400
        except OSError:
            return jsonify(error='Incompatitle file type'), 400
        total_predictions = request.args.get('total')
        if total_predictions is not None:
            try:
                total_predictions = int(total_predictions)
            except ValueError:
                total_predictions = None
        NETWORK_START_THREAD.join()
        objects = PREDICT_NETWORK.predict_image(image_array)
        objects = objects[:total_predictions]
        return jsonify({'objects': objects})


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


def get_image():
    image = request.files.get('image')
    if not image:
        raise ValueError
    image = Image.open(image.stream).convert('RGB')
    return image
