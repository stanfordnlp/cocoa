import pytest
import numpy as np
import tensorflow as tf
from numpy.testing import assert_array_equal
from cocoa.model.negotiation.price_predictor import PricePredictor

@pytest.fixture(scope='module')
def price_predictor():
    return PricePredictor(5, 3, 0)

@pytest.fixture(scope='module')
def init_price():
    return np.array([[[9,10,11], [-9,-10,-11]],
                     [[5,6,7], [-5,-6,-7]]
                    ])

@pytest.fixture(scope='module')
def input_price():
    return np.array([[1,2,3,4], [4,3,2,1]])

class TestPricePredictor(object):
    def test_update_price(self, price_predictor, init_price, input_price, capsys):
        price_predictor._build_inputs()
        price_hists = price_predictor.update_price()
        partner = np.array([True, True, True, False])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {
                    price_predictor.inputs: input_price,
                    price_predictor.init_price: init_price,
                    price_predictor.partner: partner,
                    }
            result = sess.run(price_hists, feed_dict=feed_dict)
        with capsys.disabled():
            # Batch 1, step 1
            ans_b1t1 = np.array([[9,10,11], [-10,-11,1]])
            assert_array_equal(ans_b1t1, result[0][0])
            # Batch 2, step -1
            ans_b2tN = np.array([[3,2,1], [-5,-6,-7]])
