# coding=utf-8

from flask import Flask
from flask import request

import json
import pickle
import numpy as np
import cv2
import base64

import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax

haar_cascade = cv2.CascadeClassifier("cascade.xml")
IMAGE_MEAN = pickle.load(open('mean.pkl'))
params = pickle.load(open('coins_model.pickle'))


def build_model():
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc7'] = DenseLayer(net['fc6'], num_units=4096)
    net['fc8'] = DenseLayer(net['fc7'], num_units=1000, nonlinearity=None)
    net['prob'] = DenseLayer(net['fc7'], num_units=8, nonlinearity=softmax)

    return net


net = build_model()
lasagne.layers.set_all_param_values(net['prob'], params)
X_sym = T.tensor4()
prediction = lasagne.layers.get_output(net['prob'], X_sym)
pred_fn = theano.function([X_sym], prediction, allow_input_downcast=True)


def localizeCoins(img):
    currentHeight, currentWidth = img.shape[:2]
    aspectRatio = currentWidth / float(currentHeight)
    sizeDim = 1200
    if currentWidth > currentHeight:
        img = cv2.resize(img, (sizeDim, int(1 / aspectRatio * sizeDim)), interpolation=cv2.INTER_CUBIC)
    else:
        img = cv2.resize(img, (int(aspectRatio * sizeDim), sizeDim), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detects = haar_cascade.detectMultiScale(gray, 1.1, 1)

    coins = []
    for (x, y, w, h) in detects:
        coins.append(img[y:y + h, x:x + w])

    return coins


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape)[:2] / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[:2], flags=cv2.INTER_CUBIC)
    return result


def rotateAndBatch(imgsrc):
    images = []

    currentHeight, currentWidth = imgsrc.shape[:2]
    x = int(0.15 * currentWidth)
    y = int(0.15 * currentHeight)
    w = int(0.80 * currentWidth)
    h = int(0.80 * currentHeight)

    for angle in xrange(0, 360, 60):
        img = rotate_image(imgsrc, angle)[y:y + h, x:x + w]
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        img = np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)
        img = img[:3]
        # Преобразуем в BGR
        img = img[::-1, :, :]
        img = img - IMAGE_MEAN

        images.append(img)

    return images


def classifyBatch(batch):
    y_pred = pred_fn(batch).argmax(-1)
    stats = {}
    for i in y_pred:
        if i != 0:
            stats[i] += 1
    return max(stats, key=stats.get)


def indexToCoin(idx):
    idxToCoin = {
        1: 'oneR',
        2: 'twoR',
        3: 'fiveR',
        4: 'tenR',
        5: 'fiveK',
        6: 'tenK',
        7: 'fiftyK',
    }

    return idxToCoin[idx]


app = Flask(__name__)


@app.route('/sendphoto', methods=['POST'])
def photo():
    try:
        data = json.loads(request.data)
        photobytes = base64.b64decode(data['photo'])
        img = cv2.imdecode(np.frombuffer(photobytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        print 'Localizing coins...'
        coins = localizeCoins(img)
        result = {}
        for c in coins:
            print 'Rotating coins...'
            batch = rotateAndBatch(c)
            print 'Classifying coins...'
            index = classifyBatch(batch)
            result[indexToCoin(index)] += 1

        return json.dumps(result)
    except BaseException as e:
        print 'Error: ', e
        return json.dumps({'error': e.message})


app.run(host='0.0.0.0', port=5000)
