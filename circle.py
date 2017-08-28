'''
Module generating points in circle for training data.
'''

#  import numpy as np
from pprint import pprint as pp
import numpy as np
import random

def make_regionA(r, w, n):
    '''
    Args:
        r : radius
        w : width of circle
        n : the number of data
    Returns:
        List[(x : float, y : float)]
    '''
    n_tmp = 0
    xs = []
    ys = []
    max_r = r + w/2.
    min_r = r - w/2.
    max_r_square = max_r**2
    min_r_square = min_r**2
    while(n_tmp < n):
        x = random.uniform(-max_r, max_r)
        y = random.uniform(-max_r, max_r)
        diag_square = x**2 + y**2
        if (y >= 0 and
                diag_square >= min_r_square and
                diag_square <= max_r_square):
            xs.append(x)
            ys.append(y)
            n_tmp += 1
    return (xs, ys)

def translateA2B(regionA, r, d):
    xs_A, ys_A = regionA
    xs_B = [x + r for x in xs_A]
    ys_B = [-y - d for y in ys_A]
    return xs_B, ys_B

def make_regionB(r, w, d, n):
    regionA = make_regionA(r, w, n)
    regionB = translateA2B(regionA, r, d)
    return regionB

def gen_circles(r, w, d, n):
    regionA = make_regionA(r, w, n)
    regionB = make_regionB(r, w, d, n)
    return regionA, regionB

def make_a_epoch(regionA, regionB):
    '''
    shuffle data and return one epoch
    '''
    data_A = zip(*regionA)
    data_B = zip(*regionB)
    data_label_pairs_A = [(datum, -1) for datum in data_A]
    data_label_pairs_B = [(datum, 1) for datum in data_B]
    data_label_pairs_all = data_label_pairs_A + data_label_pairs_B
    random.shuffle(data_label_pairs_all)
    return data_label_pairs_all

def test_batch(regionA, regionB):
    data_A = zip(*regionA)
    data_B = zip(*regionB)
    data = data_A + data_B
    labels = [[-1]]*len(data_A) + [[1]]*len(data_B)
    return data, labels

def make_decision_line(sess, output_layer):
    xs = np.linspace(-15, 25, 1)
    ys = np.linspace(-10, 15, 0.1)
    ys_res = []
    for x in xs:
        output_values = []
        for y in ys:
            output_value = sess.run([output_layer], feed_dict={
                    input_layer : [(x,y)],
                })
            output_values.append(output_value)
            return output_values
        #  print(output_value)
        #  index = np.argmin(output_values)
        #  ys_res.append(ys[index])
    #  return xs, ys_res

if __name__ == '__main__':
    pp(regionA(3, 1, 1000))
