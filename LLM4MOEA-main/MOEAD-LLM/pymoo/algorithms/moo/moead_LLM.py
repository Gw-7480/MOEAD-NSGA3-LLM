import numpy as np
from scipy.spatial.distance import cdist

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.algorithm import LoopwiseAlgorithm
from pymoo.core.duplicate import NoDuplicateElimination
from pymoo.core.population import Population
from pymoo.core.selection import Selection
from pymoo.core.variable import Real, get
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.pntx import SinglePointCrossover
from pymoo.operators.crossover.gpt import GPT_interface
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.reference_direction import default_ref_dirs


class NeighborhoodSelection(Selection):

    def __init__(self, prob=1.0) -> None:
        super().__init__()  # 调用父类构造函数
        self.prob = Real(prob, bounds=(0.0, 1.0))  # 选择邻域的概率，范围[0,1]

    def _do(self, problem, pop, n_select, n_parents, neighbors=None, **kwargs):
        assert n_select == len(neighbors)  # 保证选择数量与邻居数量一致
        P = np.full((n_select, n_parents), -1)  # 初始化父代索引矩阵，全部为-1

        prob = get(self.prob, size=n_select)  # 获取每个个体的邻域选择概率

        for k in range(n_select):  # 遍历每个需要选择的个体
            if np.random.random() < prob[k]:  # 如果随机数小于概率，则从邻居中选择父代 默认为0.9的概率
                P[k] = np.random.choice(neighbors[k], n_parents, replace=False)  # 从邻居中随机选择父代
            else:
                P[k] = np.random.permutation(len(pop))[:n_parents]  # 否则从整个种群中随机选择父代

        return P  # 返回父代索引矩阵


# =========================================================================================================
# Implementation
# =========================================================================================================

class MOEAD_LLM(LoopwiseAlgorithm, GeneticAlgorithm):

    def __init__(self,
                 ref_dirs=None,  # 参考方向
                 n_neighbors=20,  # 邻居数量
                 decomposition=None,  # 分解方法
                 prob_neighbor_mating=0.9,  # 邻域交配概率
                 sampling=FloatRandomSampling(),  # 采样方法
                 #crossover=SinglePointCrossover(prob=1.0),
                 crossover=GPT_interface(),  # 交叉算子，这里用GPT接口
                 mutation=PM(prob_var=None, eta=20),  # 变异算子
                 output=MultiObjectiveOutput(),  # 输出方式
                 debug_mode = False,  # 调试模式
                 model_LLM = None,  # LLM模型
                 endpoint = None,  # LLM接口地址
                 key = None,  # LLM接口密钥
                 out_file = None,  # 输出文件
                 **kwargs):

        self.debug_mode = debug_mode  # 是否调试
        self.endpoint = endpoint  # LLM接口地址
        self.model_LLM = model_LLM  # LLM模型
        self.key = key  # LLM密钥
        self.out_file = out_file  # 输出文件
        self.ref_dirs = ref_dirs  # 参考方向

        self.decomposition = decomposition  # 分解方法

        self.n_neighbors = n_neighbors  # 邻居数量

        self.neighbors = None  # 邻居索引

        self.selection = NeighborhoodSelection(prob=prob_neighbor_mating)  # 邻域选择算子

        self.pop_update = np.zeros(len(ref_dirs))  # 种群更新标记

        super().__init__(pop_size=len(ref_dirs),  # 种群规模等于参考方向数量
                         sampling=sampling,  # 采样方法
                         crossover=crossover,  # 交叉算子
                         mutation=mutation,  # 变异算子
                         eliminate_duplicates=NoDuplicateElimination(),  # 不去重
                         output=output,  # 输出方式
                         advance_after_initialization=False,  # 初始化后不自动推进
                         **kwargs)

    def _setup(self, problem, **kwargs):
        assert not problem.has_constraints(), "This implementation of MOEAD does not support any constraints."  # 不支持约束

        if self.ref_dirs is None:  # 如果没有提供参考方向
            self.ref_dirs = default_ref_dirs(problem.n_obj)  # 自动生成
        self.pop_size = len(self.ref_dirs)  # 种群规模等于参考方向数量

        self.neighbors = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort')[:, :self.n_neighbors]  # 计算每个参考方向的最近邻

        if self.decomposition is None:  # 如果没有指定分解方法
            self.decomposition = default_decomp(problem)  # 使用默认分解

    def _initialize_advance(self, infills=None, **kwargs):
        super()._initialize_advance(infills, **kwargs)  # 调用父类方法
        self.ideal = np.min(self.pop.get("F"), axis=0)  # 初始化理想点（目标最小值）

    def _next(self):
        pop = self.pop  # 当前种群

        self.pop_update = np.zeros(len(pop))  # 重置种群更新标记
        for k in np.random.permutation(len(pop)):  # 随机遍历每个个体

            P = self.selection.do(self.problem, pop, 1, self.mating.crossover.n_parents, neighbors=[self.neighbors[k]])  # 选择父代

            # FDP为每个父代计算分解目标值 用PBI分解 FDP作用：用于对父代进行排序，为LLM提供更好的输入信息，分解值反应了每个父代在当前搜索方向上的优劣程度，且排序后的父代信息将成为LLM交叉算子的输入
            FDP = [self.decomposition.do(P[0][i].get("F"), weights=self.ref_dirs[k, :], ideal_point=self.ideal) for i in range(self.mating.crossover.n_parents) ]

            # 进行交叉和变异，生成子代（只取第一个）
            off = np.random.choice(self.mating.do(self.problem, pop,  1,parents=P, n_max_iterations=1,Y=FDP,\
                                                  debug_mode=self.debug_mode, model_LLM=self.model_LLM, endpoint =self.endpoint, key = self.key,out_filename=self.out_file,parents_obj=P))

            off = yield off  # 评估子代

            self.ideal = np.min(np.vstack([self.ideal, off.F]), axis=0)  # 更新理想点

            self._replace(k, off)  # 用子代替换父代（如果更优）

    def _replace(self, k, off):
        pop = self.pop  # 当前种群

        N = self.neighbors[k]  # 获取第k个个体的邻居索引
        FV = self.decomposition.do(pop[N].get("F"), weights=self.ref_dirs[N, :], ideal_point=self.ideal)  # 计算邻居的分解目标值
        off_FV = self.decomposition.do(off.F[None, :], weights=self.ref_dirs[N, :], ideal_point=self.ideal)  # 计算子代的分解目标值

        # 最多替换2个邻居
        n_up = 0
        for i in range(len(FV)):
            if off_FV[i]<FV[i]:  # 如果子代更优
                pop[N[i]] = off  # 用子代替换邻居
                n_up += 1  # 替换计数加1
                self.pop_update[i] = 1  # 标记已更新
            if n_up >= 2:  # 最多替换2个
                break



class ParallelMOEAD(MOEAD_LLM):

    def __init__(self, ref_dirs, **kwargs):
        super().__init__(ref_dirs, **kwargs)  # 调用父类MOEAD_LLM的构造函数，初始化参考方向等参数
        self.indices = None  # 用于存储当前批次生成的个体索引

    def _infill(self):
        pop_size, cross_parents, cross_off = self.pop_size, self.mating.crossover.n_parents, self.mating.crossover.n_offsprings
        # 获取种群规模、交叉所需父代数、每次交叉产生的子代数

        # do the mating in a random order
        indices = np.random.permutation(len(self.pop))[:self.n_offsprings]
        # 随机打乱种群索引，选取n_offsprings个个体作为本轮交配的目标

        # get the parents using the neighborhood selection
        P = self.selection.do(self.problem, self.pop, self.n_offsprings, cross_parents,
                              neighbors=self.neighbors[indices])
        # 对每个目标个体，使用邻域选择算子选出交配父代

        # do not any duplicates elimination - thus this results in exactly pop_size * n_offsprings offsprings
        off = self.mating.do(self.problem, self.pop, 1e12, n_max_iterations=1, parents=P)
        # 进行交叉和变异操作，生成大量子代（不去重）

        # select a random offspring from each mating
        off = Population.create(*[np.random.choice(pool) for pool in np.reshape(off, (self.n_offsprings, -1))])
        # 对每组交配结果，从中随机选择一个子代作为最终的子代个体

        # store the indices because of the neighborhood matching in advance
        self.indices = indices  # 保存本轮交配的目标个体索引，便于后续替换

        return off  # 返回生成的子代个体

    def _advance(self, infills=None, **kwargs):
        assert len(self.indices) == len(infills), "Number of infills must be equal to the one created beforehand."
        # 检查生成的子代数量和索引数量是否一致

        # update the ideal point before starting to replace
        self.ideal = np.min(np.vstack([self.ideal, infills.get("F")]), axis=0)
        # 更新理想点（所有目标的最小值）

        # now do the replacements as in the loop-wise version
        for k, off in enumerate(infills):
            self._replace(self.indices[k], off)
            # 用子代替换父代（如果更优），索引与之前保存的indices对应

def default_decomp(problem):
    if problem.n_obj <= 2:
        from pymoo.decomposition.tchebicheff import Tchebicheff
        return Tchebicheff()  # 2目标问题用Tchebicheff分解
    else:
        from pymoo.decomposition.pbi import PBI
        return PBI()  # 多于2目标问题用PBI分解

parse_doc_string(MOEAD_LLM.__init__)  # 解析MOEAD_LLM构造函数的文档字符串
