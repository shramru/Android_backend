# coding=utf-8

from flask import Flask
from flask import request

import time
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
bin_params = pickle.load(open('bin_coins_model.pickle'))
coin_params = pickle.load(open('seven_coins_model.pickle'))


def build_model_bin():
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
    net['prob'] = DenseLayer(net['fc7'], num_units=2, nonlinearity=softmax)

    return net


def build_model_coins():
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
    net['prob'] = DenseLayer(net['fc7'], num_units=7, nonlinearity=softmax)

    return net


net_bin = build_model_bin()
lasagne.layers.set_all_param_values(net_bin['prob'], bin_params)
net_coins = build_model_coins()
lasagne.layers.set_all_param_values(net_coins['prob'], coin_params)

X_sym = T.tensor4()
prediction_bin = lasagne.layers.get_output(net_bin['prob'], X_sym)
pred_fn_bin = theano.function([X_sym], prediction_bin, allow_input_downcast=True)
prediction_coins = lasagne.layers.get_output(net_coins['prob'], X_sym)
pred_fn_coins = theano.function([X_sym], prediction_coins, allow_input_downcast=True)


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

    def determineRect(x1, y1, w1, h1):
        for (x2, y2, w2, h2) in detects:
            if x1 > x2 and y1 > y2 and x1 + w1 < x2 + w2 and y1 + h1 < y2 + h2:
                return True
        return False

    detects = [(x, y, w, h) for (x, y, w, h) in detects if not determineRect(x, y, w, h)]

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

    for angle in xrange(0, 360, 72):
        img = rotate_image(imgsrc, angle)[y:y + h, x:x + w]
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        img = np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)
        img = img[:3]
        # Преобразуем в BGR
        img = img[::-1, :, :]
        img = img - IMAGE_MEAN

        images.append(img)

    return np.asarray(images) / np.float32(256)


def classifyCoinBatch(batch):
    y_pred = pred_fn_coins(batch).argmax(-1)
    print y_pred
    stats = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0
    }
    for i in y_pred:
        stats[i] += 1
    return max(stats, key=stats.get)


def classifyBinBatch(batch):
    y_pred = pred_fn_bin(batch).argmax(-1)
    print y_pred
    for i in y_pred:
        if i == 1:
            return False
    return True


def indexToCoin(idx):
    idxToCoin = {
        0: 'oneR',
        1: 'twoR',
        2: 'fiveR',
        3: 'tenR',
        4: 'fiveK',
        5: 'tenK',
        6: 'fiftyK',
    }

    return idxToCoin[idx]


app = Flask(__name__)


@app.route('/sendphoto', methods=['POST'])
def photo():
    try:
        data = json.loads(request.data)
        photobytes = base64.b64decode(data['photo'])
        img = cv2.imdecode(np.frombuffer(photobytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        start_time = time.time()
        print 'Localizing coins...'
        coins = localizeCoins(img)
        print 'Found %s coins' % len(coins)
        result = {
            'oneR': 0,
            'twoR': 0,
            'fiveR': 0,
            'tenR': 0,
            'fiveK': 0,
            'tenK': 0,
            'fiftyK': 0
        }

        for c in coins:
            print 'Rotating coins...'
            batch = rotateAndBatch(c)
            print 'Classifying is it coin...'
            if classifyBinBatch(batch):
                print "It's coin!"
                print 'Classifying coin...'
                index = classifyCoinBatch(batch)
                result[indexToCoin(index)] += 1

        print("--- %s seconds ---" % (time.time() - start_time))
        return json.dumps(result)
    except BaseException as e:
        print 'Error: ', e
        return json.dumps({'error': e.message})


app.run(host='0.0.0.0', port=5000)
