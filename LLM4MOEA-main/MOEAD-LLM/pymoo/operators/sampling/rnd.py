import numpy as np

from pymoo.core.sampling import Sampling
from MODTSRA.Utils import N_n, get_argument_parser, N_service
from MODTSRA.fitness import NSGA2
# 全局变量保存DAG依赖信息
PREDECESSORS_LIST = None

class CustomRandomSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        global PREDECESSORS_LIST

        args = get_argument_parser()
        Nij = args.Nij
        K = args.K
        len_order = N_n * Nij
        len_location = N_n * Nij
        len_A = N_n
        n_var = len_order + len_location + len_A

        # 获取DAG结构
        fitness_evaluator = NSGA2(problem.instance_name, args)
        workflow_set = fitness_evaluator.workflow_set

        # 提取每个UAV的DAG依赖信息
        predecessors_list = []
        for workflow in workflow_set:
            predecessors = {}
            for task in workflow.taskSet:
                predecessors[task.id] = set(task.preTaskSet)
            predecessors_list.append(predecessors)
        PREDECESSORS_LIST = predecessors_list  # 保存到全局变量

        X = np.zeros((n_samples, n_var))

        for i in range(n_samples):
            order_flat = []
            for j in range(N_n):
                workflow = workflow_set[j]
                # 用已有的随机拓扑序生成函数，返回整数排列
                topo_order = fitness_evaluator.initializeWorkflowSequence(workflow)
                order_flat.extend(topo_order)
            order_flat = np.array(order_flat)
            # location部分
            location = np.random.randint(1, K+2, size=len_location)
            # A部分
            A = np.random.dirichlet(np.ones(len_A))
            # 拼接
            X[i, :len_order] = order_flat
            X[i, len_order:len_order+len_location] = location
            X[i, len_order+len_location:] = A

        return X

def random(problem, n_samples=1):
    X = np.random.random((n_samples, problem.n_var))

    if problem.has_bounds():
        xl, xu = problem.bounds()
        assert np.all(xu >= xl)
        X = xl + (xu - xl) * X

    return X


class FloatRandomSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.random.random((n_samples, problem.n_var))

        if problem.has_bounds():
            xl, xu = problem.bounds()
            assert np.all(xu >= xl)
            X = xl + (xu - xl) * X

        return X


class BinaryRandomSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        val = np.random.random((n_samples, problem.n_var))
        return (val < 0.5).astype(bool)


class IntegerRandomSampling(FloatRandomSampling):

    def _do(self, problem, n_samples, **kwargs):
        n, (xl, xu) = problem.n_var, problem.bounds()
        return np.column_stack([np.random.randint(xl[k], xu[k] + 1, size=n_samples) for k in range(n)])


class PermutationRandomSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), 0, dtype=int)
        for i in range(n_samples):
            X[i, :] = np.random.permutation(problem.n_var)
        return X



