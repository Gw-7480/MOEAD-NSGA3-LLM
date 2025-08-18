import numpy as np

from pymoo.core.duplicate import NoDuplicateElimination
from pymoo.core.population import Population
from pymoo.core.repair import NoRepair
from pymoo.util.misc import at_least_2d_array


class Initialization:

    def __init__(self,
                 sampling,
                 repair=None,
                 eliminate_duplicates=None) -> None:

        super().__init__()
        self.sampling = sampling
        self.eliminate_duplicates = eliminate_duplicates if eliminate_duplicates else NoDuplicateElimination()
        self.repair = repair if repair is not None else NoRepair()


    def do(self, problem, n_samples, **kwargs):
        # 如果 self.sampling 已经是一个 Population 对象，直接使用它作为种群
        if isinstance(self.sampling, Population):
            pop = self.sampling

        else:
            # 如果 self.sampling 是一个 numpy 数组，转换为二维数组并包装为 Population
            if isinstance(self.sampling, np.ndarray):
                sampling = at_least_2d_array(self.sampling)  # 保证采样数据至少二维
                pop = Population.new(X=sampling)  # 用采样数据生成种群对象
            else:
                # 否则，self.sampling 应该是一个采样函数，调用它生成种群
                pop = self.sampling(problem, n_samples, **kwargs)

        # 找出所有还没有被评估过的个体（即目标函数值未计算）
        not_eval_yet = [k for k in range(len(pop)) if len(pop[k].evaluated) == 0]
        if len(not_eval_yet) > 0:
            # 对这些未评估的个体进行修复（如边界修正、可行性修正等）
            pop[not_eval_yet] = self.repair(problem, pop[not_eval_yet], **kwargs)

        # 对种群中的个体进行去重，移除重复解
        pop = self.eliminate_duplicates.do(pop)

        # 返回最终生成并处理好的种群对象
        return pop
