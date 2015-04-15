# coding: utf-8
from numpy import *
import cart


def readdata():
    rdata = [l.split() for l in open('lenses.txt').readlines()]
    for l in rdata:
        if l[-2] == 'no':
            l[-2] += ' ' + l[-1]
            l.pop(-1)
    data = array(rdata)
    return data


def test_main():
    reload(cart)
    data = readdata()
    tree = cart.cart_tree()
    print tree.create_tree(data)


if __name__ == '__main__':
    test_main()
