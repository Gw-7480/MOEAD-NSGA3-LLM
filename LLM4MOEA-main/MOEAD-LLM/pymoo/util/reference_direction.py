import sys

import numpy as np
from scipy import special

from pymoo.util.misc import find_duplicates, cdist


# =========================================================================================================
# Model
# =========================================================================================================


def default_ref_dirs(m):
    if m == 1:
        return np.array([[1.0]])
    elif m == 2:
        return UniformReferenceDirectionFactory(m, n_partitions=99).do()
    elif m == 3:
        return UniformReferenceDirectionFactory(m, n_partitions=12).do()
    else:
        raise Exception("No default reference directions for more than 3 objectives. Please provide them directly:"
                        "https://pymoo.org/misc/reference_directions.html")


def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
    """
    Das-Dennis递归算法的具体实现。

    参数:
        ref_dirs: list
            存储所有生成的权重向量的列表
        ref_dir: ndarray
            当前递归路径上的权重向量（部分填充）
        n_partitions: int
            总分区数
        beta: int
            剩余可分配的分区数
        depth: int
            当前递归深度（对应权重向量的第几个分量）
    """
    if depth == len(ref_dir) - 1:
        # 递归终止条件：到达最后一个分量
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])  # 添加当前权重向量的副本
    else:
        # 递归继续：枚举当前分量的所有可能取值
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            # 递归调用处理下一个分量
            das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1)


class ReferenceDirectionFactory:

    def __init__(self, n_dim, scaling=None, lexsort=True, verbose=False, seed=None, **kwargs) -> None:
        super().__init__()
        self.n_dim = n_dim
        self.scaling = scaling
        self.lexsort = lexsort
        self.verbose = verbose
        self.seed = seed

    def __call__(self):
        return self.do()

    def do(self):

        # 如果设置了随机种子，则设置 numpy 的随机种子，保证可复现
        if self.seed is not None:
            np.random.seed(self.seed)

        # 如果是一维目标，直接返回 [1.0]
        if self.n_dim == 1:
            return np.array([[1.0]])
        else:
            # 调用子类的 _do() 方法生成权重向量
            val = self._do()
            if isinstance(val, tuple):
                ref_dirs, other = val[0], val[1:]
            else:
                ref_dirs = val

            # 如果设置了缩放参数，对权重向量进行缩放
            if self.scaling is not None:
                ref_dirs = scale_reference_directions(ref_dirs, self.scaling)

            # 如果需要按字典序排序，对权重向量排序
            if self.lexsort:
                I = np.lexsort([ref_dirs[:, j] for j in range(ref_dirs.shape[1])][::-1])
                ref_dirs = ref_dirs[I]

            # 返回最终的权重向量
            return ref_dirs

    def _do(self):
        return None


# =========================================================================================================
# Das Dennis Reference Directions (Uniform)
# =========================================================================================================


class UniformReferenceDirectionFactory(ReferenceDirectionFactory):

    def __init__(self, n_dim, n_partitions=None, n_points=None, **kwargs) -> None:
        super().__init__(n_dim, **kwargs)

        # 检查参数：必须提供 n_partitions 或 n_points 之一
        if n_partitions is None and n_points is None:
            raise Exception("Either provide number of partitions or number of points.")

        if n_partitions is not None:
            self.n_partitions = n_partitions
        else:
            # 如果提供 n_points，计算对应的 n_partitions
            self.n_partitions = get_partition_closest_to_points(n_points, n_dim)

    def _do(self):
        # 调用 das_dennis 算法生成均匀分布的权重向量
        return das_dennis(self.n_partitions, self.n_dim)


def get_number_of_uniform_points(n_partitions, n_dim):
    """
    计算在 n_dim 维空间，将 [0,1] 区间等分为 n_partitions 份时，
    可以均匀生成多少个参考方向（权重向量）。

    这等价于求非负整数解的个数，满足 k_1 + k_2 + ... + k_n = n_partitions。
    组合数公式为 C(n_dim + n_partitions - 1, n_partitions)。

    参数:
        n_partitions: int
            分区数，将 [0,1] 区间等分为 n_partitions 份
        n_dim: int
            目标数量（权重向量的维度）

    返回:
        int
            可以均匀生成的权重向量（参考方向）个数
    """
    # 使用 scipy.special.binom 计算组合数 也就是n_dim等于3 n_paititions等于12 C(12,10)=66个权重向量
    return int(special.binom(n_dim + n_partitions - 1, n_partitions))


def get_partition_closest_to_points(n_points, n_dim):
    """
    Returns the corresponding partition number which create the desired number of points
    or less!
    """

    if n_dim == 1:
        return 0

    n_partitions = 1
    _n_points = get_number_of_uniform_points(n_partitions, n_dim)
    while _n_points <= n_points:
        n_partitions += 1
        _n_points = get_number_of_uniform_points(n_partitions, n_dim)
    return n_partitions - 1


def das_dennis(n_partitions, n_dim):
    if n_partitions == 0:
        # 如果分区数为0，只能均匀分配，每个目标权重为 1/n_dim
        return np.full((1, n_dim), 1 / n_dim)
    else:
        ref_dirs = []  # 用于存储所有生成的权重向量
        ref_dir = np.full(n_dim, np.nan)  # 临时数组，存储当前递归路径上的分配
        # 递归枚举所有分配方式
        das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        # 合并所有结果为一个二维数组返回
        return np.concatenate(ref_dirs, axis=0)


class MultiLayerReferenceDirectionFactory:

    def __init__(self, *args) -> None:
        self.layers = []
        self.layers.extend(args)

    def __call__(self):
        return self.do()

    def add_layer(self, *args):
        self.layers.extend(args)

    def do(self):
        ref_dirs = []
        for factory in self.layers:
            ref_dirs.append(factory)
        ref_dirs = np.concatenate(ref_dirs, axis=0)
        is_duplicate = find_duplicates(ref_dirs)
        return ref_dirs[np.logical_not(is_duplicate)]


# =========================================================================================================
# Util
# =========================================================================================================

def get_rng(seed=None):
    if seed is None or type(seed) == int:
        rng = np.random.default_rng(seed)
    return rng


def sample_on_unit_simplex(n_points, n_dim, unit_simplex_mapping="kraemer", seed=None):
    if unit_simplex_mapping == "sum":
        rnd = map_onto_unit_simplex(get_rng(seed).random((n_points, n_dim)), "sum")

    elif unit_simplex_mapping == "kraemer":
        rnd = map_onto_unit_simplex(get_rng(seed).random((n_points, n_dim)), "kraemer")

    elif unit_simplex_mapping == "das-dennis":
        n_partitions = get_partition_closest_to_points(n_points, n_dim)
        rnd = UniformReferenceDirectionFactory(n_dim, n_partitions=n_partitions).do()

    else:
        raise Exception("Please define a valid sampling on unit simplex strategy!")

    return rnd


def map_onto_unit_simplex(rnd, method):
    n_points, n_dim = rnd.shape

    if method == "sum":
        ret = rnd / rnd.sum(axis=1)[:, None]

    elif method == "kraemer":
        M = sys.maxsize

        rnd *= M
        rnd = rnd[:, :n_dim - 1]
        rnd = np.column_stack([np.zeros(n_points), rnd, np.full(n_points, M)])

        rnd = np.sort(rnd, axis=1)

        ret = np.full((n_points, n_dim), np.nan)
        for i in range(1, n_dim + 1):
            ret[:, i - 1] = rnd[:, i] - rnd[:, i - 1]
        ret /= M

    else:
        raise Exception("Invalid unit simplex mapping!")

    return ret


def scale_reference_directions(ref_dirs, scaling):
    return ref_dirs * scaling + ((1 - scaling) / ref_dirs.shape[1])


def get_reference_directions(method="uniform", n_dim=3, n_partitions=12, **kwargs):
    """
    获取参考方向的便捷函数。

    参数:
        method: str - 生成方法 ("uniform", "das-dennis")
        n_dim: int - 目标数量
        n_partitions: int - 分区数

    返回:
        ndarray - 参考方向矩阵
    """
    if method in ["uniform", "das-dennis"]:
        return UniformReferenceDirectionFactory(n_dim, n_partitions=n_partitions, **kwargs).do()
    else:
        raise ValueError(f"Unknown reference direction method: {method}")


def select_points_with_maximum_distance(points, n_select, method="farthest_point"):
    """
    从给定点集中选择具有最大距离的点子集。

    参数:
        points: ndarray, shape (n_points, n_dim)
            候选点集
        n_select: int
            要选择的点数量
        method: str
            选择方法，默认"farthest_point"

    返回:
        ndarray: 选中点的索引
    """
    n_points, n_dim = points.shape

    if n_select >= n_points:
        return np.arange(n_points)

    if method == "farthest_point":
        # 最远点采样算法
        selected = []
        remaining = list(range(n_points))

        # 随机选择第一个点
        first_idx = np.random.randint(n_points)
        selected.append(first_idx)
        remaining.remove(first_idx)

        # 迭代选择距离已选点最远的点
        for _ in range(n_select - 1):
            if not remaining:
                break

            max_min_dist = -1
            next_idx = None

            for idx in remaining:
                # 计算到所有已选点的最小距离
                min_dist = float('inf')
                for sel_idx in selected:
                    dist = np.linalg.norm(points[idx] - points[sel_idx])
                    min_dist = min(min_dist, dist)

                # 选择最小距离最大的点
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    next_idx = idx

            if next_idx is not None:
                selected.append(next_idx)
                remaining.remove(next_idx)

        return np.array(selected)

    elif method == "random":
        # 随机选择
        return np.random.choice(n_points, n_select, replace=False)

    else:
        raise ValueError(f"Unknown selection method: {method}")