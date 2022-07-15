'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-07-15 09:13:32
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-07-15 14:54:47
Description: 
'''
import functools
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd


# 如果有N个交易员的第I个股票用create_formulae生成一组formulae
# 量化化
def create_vector_formulae(M: int,
                           A: List[Callable],
                           O: List[Callable],
                           stock_num: int,
                           max_lag: int = 9,
                           l: int = 1,
                           seed: int = None) -> List[Callable]:
    """构造$\Theta$
       $\Theta=\sum^{M}_{j}w_{j}A_{j}(O_{j}(r_{P_{j}}[t-D_{j}],r_{Q_{j}}[t-F_{j}]))$
       
    Args:
        M (int): 每位交易员表达式最大项数
        A (List[Callable]): 激活函数列表
        O (List[Callable]): 二元操作符函数列表
        stock_num (int): 股票个数
        max_lag (int, optional): 数据延迟最大取值. Defaults to 9.
        l (int, optional): 交易延迟量,即观察到数据后不可能立马进行交易,需要等待l时间. Defaults to 1.
        seed (int, optional): 随机数种子. Defaults to None.

    Returns:
        List: _description_
    """
    def _formula_func(data: np.ndarray, active_func: Callable,
                      binary_oper: Callable, P: int, Q: int, F: int,
                      D: int) -> float:
        """公式生成

        Args:
            data (np.ndarray): 传入数据1轴为股票 0为时间
            active_func (Callable):激活函数
            binary_oper (Callable):二元操作符
            P (int):所选股票下标
            Q (int):所选股票下标
            F (int):滞后期
            D (int):滞后期
        """

        t = data.shape[0]  # 获取时间长度
        if t < (max_lag + l):
            raise ValueError(
                f'数据时间序列长度(t={t})不能小于max_lag + l(max_lag+l={max_lag + l})!')

        indices: np.ndarray = np.arange(max_lag + l, t)

        x: np.ndarray = np.take(data[:, P], indices - D)

        y: np.ndarray = np.take(data[:, Q], indices - F)

        # return active_func(binary_oper(data[P][t - D], data[Q][t - F]))
        return active_func(binary_oper(x, y))

    if seed:

        np.random.seed(seed)

    m: int = np.random.choice(M)  # 根据M最大项数选择个数

    # 向量化
    # 构建延迟数
    a: np.ndarray = np.arange(1, max_lag + l)  # 最少为1天
    D: np.ndarray = np.random.choice(a, m)
    F: np.ndarray = np.random.choice(a, m)

    # 随机选择股票
    P: np.ndarray = np.random.choice(stock_num, m)
    Q: np.ndarray = np.random.choice(stock_num, m)

    # 列表中存放构成公式的"算子"
    formulae: np.ndarray = np.array(
        [np.random.choice(A, m),
         np.random.choice(O, m), P, Q, D, F],
        dtype='object').T

    # 列表中的func后续近仅需要传入data及对应的T即可得到返回值
    formulae_ls_func: List[Callable] = [
        functools.partial(_formula_func,
                          active_func=row[0],
                          binary_oper=row[1],
                          P=row[2],
                          Q=row[3],
                          F=row[4],
                          D=row[5]) for row in formulae
    ]

    return formulae_ls_func


# 如果有N个交易员的第I个股票用create_formulae生成一组formulae
def create_formulae(M: int,
                    A: List[Callable],
                    O: List[Callable],
                    stock_num: int,
                    max_lag: int = 9,
                    l: int = 1,
                    seed: int = None) -> List[Callable]:
    """构造$\Theta$
       $\Theta=\sum^{M}_{j}w_{j}A_{j}(O_{j}(r_{P_{j}}[t-D_{j}],r_{Q_{j}}[t-F_{j}]))$
       
    Args:
        M (int): 每位交易员表达式最大项数
        A (List[Callable]): 激活函数列表
        O (List[Callable]): 二元操作符函数列表
        stock_num (int): 股票个数
        max_lag (int, optional): 数据延迟最大取值. Defaults to 9.
        l (int, optional): 交易延迟量,即观察到数据后不可能立马进行交易,需要等待l时间. Defaults to 1.
        seed (int, optional): 随机数种子. Defaults to None.

    Returns:
        List: _description_
    """
    def _formula_func(data: np.ndarray, active_func: Callable,
                      binary_oper: Callable, P: int, Q: int, F: int,
                      D: int) -> float:
        """公式生成

        Args:
            data (np.ndarray): 传入数据1轴为股票 0为时间
            active_func (Callable):激活函数
            binary_oper (Callable):二元操作符
            P (int):所选股票下标
            Q (int):所选股票下标
            F (int):滞后期
            D (int):滞后期
        """

        t = data.shape[0]  # 获取时间长度
        if t < (max_lag + l):
            raise ValueError(
                f'数据时间序列长度(t={t})不能小于max_lag + l(max_lag+l={max_lag + l})!')

        x: np.ndarray = data[t - D, P]

        y: np.ndarray = data[t - F, Q]

        # return active_func(binary_oper(data[P][t - D], data[Q][t - F]))
        return active_func(binary_oper(x, y))

    if seed:

        np.random.seed(seed)

    m: int = np.max([np.random.choice(M + 1), 1])  # 根据M最大项数选择个数,至少有一个交易员

    # 向量化
    # 构建延迟数
    a: np.ndarray = np.arange(1, max_lag + l)  # 最少为1天
    D: np.ndarray = np.random.choice(a, m)
    F: np.ndarray = np.random.choice(a, m)

    # 随机选择股票
    P: np.ndarray = np.random.choice(stock_num, m)
    Q: np.ndarray = np.random.choice(stock_num, m)

    # 列表中存放构成公式的"算子"
    formulae: np.ndarray = np.array(
        [np.random.choice(A, m),
         np.random.choice(O, m), P, Q, D, F],
        dtype='object').T

    # 列表中的func后续近仅需要传入data及对应的T即可得到返回值
    formulae_ls_func: List[Callable] = [
        functools.partial(_formula_func,
                          active_func=row[0],
                          binary_oper=row[1],
                          P=row[2],
                          Q=row[3],
                          F=row[4],
                          D=row[5]) for row in formulae
    ]

    return formulae_ls_func


def _calc_ols_func(exog: np.ndarray, endog: np.ndarray) -> np.ndarray:
    """ols函数"""
    A: np.ndarray = np.c_[np.ones(len(exog)), exog]

    return np.linalg.lstsq(A, endog, rcond=None)[0]


def rolling_window(data: np.ndarray, window: int) -> List:
    """获取滚动窗口期内的数据

    Args:
        data (np.ndarray): axis 0为日期 1为股票
        window (int): 窗口期

    Returns:
        np.ndarray
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    shape = (data.shape[0] - window + 1, window) + data.shape[1:]
    strides = (data.strides[0], ) + data.strides
    slice_arr = np.squeeze(
        np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides))

    if slice_arr.ndim == 1:
        slice_arr = np.atleast_2d(slice_arr)
    return slice_arr