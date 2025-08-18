from abc import abstractmethod

import numpy as np

from pymoo.core.operator import Operator


class Selection(Operator):

    def __init__(self, **kwargs) -> None:
        """
        This class is used to select parents for the mating or other evolutionary operators.
        Several strategies can be used to increase the selection pressure.
        """
        super().__init__(**kwargs)

    def do(self, problem, pop, n_select, n_parents, to_pop=True, **kwargs):
        """
        从种群中选择新个体（通常作为父代），用于交叉操作。

        参数说明
        ----------
        problem: class
            当前优化问题对象，提供变量范围、可行性等信息，部分交叉算子可能用到。
        pop : Population
            当前种群对象，选择操作将在此基础上进行。
        n_select : int
            需要选择的个体数量。
        n_parents : int
            每组交叉操作所需的父代个体数。
        to_pop : bool
            如果选择结果是索引，是否自动转换为个体对象。
        返回
        -------
        parents : list
            选择出的父代个体（或索引）。
        """

        # 调用实际的选择实现（由子类实现），返回个体索引或个体对象
        ret = self._do(problem, pop, n_select, n_parents, **kwargs)

        # 如果返回的是索引数组，并且需要转换为个体对象，则进行转换
        if to_pop and isinstance(ret, np.ndarray) and np.issubdtype(ret.dtype, np.integer):
            ret = pop[ret]

        # 返回最终选择的父代（个体对象或索引）
        return ret

    @abstractmethod
    def _do(self, problem, pop, n_select, n_parents, **kwargs):
        pass


