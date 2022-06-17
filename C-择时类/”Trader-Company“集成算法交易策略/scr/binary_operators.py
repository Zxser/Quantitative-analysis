'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-06-12 21:10:12
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-06-17 18:11:39
Description: 符号系统
'''

from typing import Union

import numpy as np


def add(x: Union[int, np.ndarray],
        y: Union[int, np.ndarray] = None) -> Union[int, np.ndarray]:

    return np.add(x, y)


def diff(x: Union[int, np.ndarray],
         y: Union[int, np.ndarray] = None) -> Union[int, np.ndarray]:

    return np.subtract(x, y)


def multiple(x: Union[int, np.ndarray],
             y: Union[int, np.ndarray] = None) -> Union[int, np.ndarray]:

    return np.multiply(x, y)


def get_x(x: Union[int, np.ndarray],
          y: Union[int, np.ndarray] = None) -> Union[int, np.ndarray]:
    return x


def get_y(x: Union[int, np.ndarray],
          y: Union[int, np.ndarray] = None) -> Union[int, np.ndarray]:
    return y


def x_is_greater_than_y(
        x: Union[int, np.ndarray],
        y: Union[int, np.ndarray] = None) -> Union[int, np.ndarray]:
    return np.greater(x, y) * 1.0


def 
