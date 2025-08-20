import numpy as np
import itertools
import os
import copy
import time
import pandas as pd
import plotly.graph_objs as go
from MODTSRA.NSGA import NSGA2
from deepseek_api import deepseek_call
from MODTSRA.llm_operator import LLMSearchOperator


class NSGA3(NSGA2):
    """NSGA-III算法类，继承自NSGA2，并集成了LLM搜索算子"""

    def __init__(self, instance_name, args, num_objectives=3, divisions=12, use_llm=True, llm_caller=None):
        """
        初始化NSGA3算法

        Args:
            instance_name: 实例名称
            args: 算法参数
            num_objectives: 目标函数数量，默认3个目标
            divisions: 参考点划分数，用于生成均匀分布的参考点
            use_llm: 是否使用LLM算子，默认True
            llm_caller: LLM调用函数，用于与深度学习模型交互
        """
        # 调用父类NSGA2的初始化方法，继承基本的遗传算法框架
        super().__init__(instance_name, args)
        # 设置目标函数的数量
        self.num_objectives = num_objectives
        # 设置参考点的划分精度
        self.divisions = divisions
        # 生成均匀分布的参考点，用于NSGA3的选择机制
        self.reference_points = generate_reference_points(num_objectives, divisions)
        # === 新增：LLM 黑盒算子配置 ===
        # 标记是否启用LLM搜索算子
        self.use_llm = use_llm
        # 创建LLM搜索算子实例，用于智能化的个体变异和交叉
        self.llm_operator = LLMSearchOperator(
            K=self.K,  # 从父类继承的参数K
            Nij=self.Nij,  # 从父类继承的邻接矩阵参数
            use_llm=use_llm,  # 是否使用LLM的标志
            llm_caller=llm_caller,  # 外部提供的LLM调用函数（如deepseek接口）
            llm_temperature=0.2,  # LLM生成的温度参数，控制随机性
            p_apply_llm=1.0  # 应用LLM算子的概率（1.0表示总是使用）
        )

    def environmental_selection(self, population, N):
        """
        环境选择方法，从合并种群中选择N个个体进入下一代
        这是NSGA3的核心选择机制，基于非支配排序和参考点关联

        Args:
            population: 当前合并的种群（包含父代和子代）
            N: 需要选择的个体数量

        Returns:
            选择后的新种群列表
        """
        # 对整个种群进行快速非支配排序，得到不同的前沿层级
        self.fast_non_dominated_sort(population)
        # 初始化新种群列表
        new_population = []
        # 从第一层前沿开始选择个体
        i = 1
        # 逐层添加完整的前沿层，直到无法添加整层为止
        while len(new_population) + len(self.F_rank[i]) <= N:
            # 将当前层的所有个体加入新种群
            new_population.extend(self.F_rank[i])
            i += 1  # 移动到下一层前沿
        # 获取最后一层需要部分选择的前沿
        last_layer = self.F_rank[i]
        # 计算还需要选择的个体数量
        K = N - len(new_population)
        # 使用参考点选择机制从最后一层中选择K个个体
        selected = self.reference_point_selection(last_layer, K)
        # 将选择的个体添加到新种群中
        new_population.extend(selected)
        return new_population

    def reference_point_selection(self, last_layer, K):
        """
        基于参考点的选择方法，这是NSGA3区别于NSGA2的关键机制
        通过将个体与参考点关联来维持种群的多样性

        Args:
            last_layer: 最后一层前沿的个体列表
            K: 需要从该层选择的个体数量

        Returns:
            选择的K个个体列表
        """
        # 提取该层所有个体的目标函数值
        objs = np.array([ind.fitness for ind in last_layer])
        # 计算目标空间的最小值和最大值，用于归一化
        min_vals = objs.min(axis=0)
        max_vals = objs.max(axis=0)
        # 将目标函数值归一化到[0,1]范围，避免不同量级的影响
        # 添加小的常数1e-12防止除零错误
        norm_objs = (objs - min_vals) / (max_vals - min_vals + 1e-12)
        # 计算每个个体与所有参考点的关联关系
        associations = []
        for i, obj in enumerate(norm_objs):
            # 计算当前个体到所有参考点的欧氏距离
            dists = np.linalg.norm(self.reference_points - obj, axis=1)
            # 存储个体索引、最近参考点索引、最小距离
            associations.append((i, np.argmin(dists), np.min(dists)))
        # 统计每个参考点关联的个体数量
        ref_count = {i: 0 for i in range(len(self.reference_points))}
        for _, ref_idx, _ in associations:
            ref_count[ref_idx] += 1
        # 开始选择过程
        selected = []  # 存储选中的个体
        used = set()  # 记录已使用的个体索引
        # 循环直到选够K个个体
        while len(selected) < K:
            # 找到关联个体数最少的参考点（保持多样性）
            min_ref = min(ref_count, key=lambda x: ref_count[x])
            # 找到与该参考点关联且未被使用的个体候选
            candidates = [(i, dist) for i, ref_idx, dist in associations
                          if ref_idx == min_ref and i not in used]
            # 如果没有可用候选，标记该参考点为无穷大，跳过
            if not candidates:
                ref_count[min_ref] = float('inf')
                continue
            # 选择距离该参考点最近的个体
            idx, _ = min(candidates, key=lambda x: x[1])
            # 将选中的个体加入结果列表
            selected.append(last_layer[idx])
            # 标记该个体已被使用
            used.add(idx)
            # 增加该参考点的关联计数
            ref_count[min_ref] += 1
        return selected

    def run(self):
        """
        NSGA3算法的主运行流程
        包含初始化、迭代进化、最终结果返回等步骤

        Returns:
            最终的外部存档（Pareto前沿）列表
        """
        # === 算法初始化阶段 ===
        # 初始化种群，生成初始的个体
        self.initializePopulation()
        # 对初始种群进行快速非支配排序
        self.fast_non_dominated_sort(self.P_population)
        # 用第一层前沿（最优个体）初始化外部存档EP
        self.initializeEP(self.F_rank[1])
        # 为除第一层外的其他层计算拥挤距离（保持NSGA2兼容性）
        for i in range(1, len(self.F_rank)):
            self.crowding_distance_assignment(self.F_rank[i])
        # === 开始进化迭代 ===
        # 通过选择、交叉、变异生成初始子代种群Q
        self.Q_population = self.make_new_population(self.P_population)
        # 更新外部存档，加入新发现的非支配解
        self.update_EP_FromSet(self.EP, self.F_rank[1])
        # 初始化代数计数器
        t = 1
        # 主进化循环，直到达到最大代数
        while t <= self.maxGen:
            # 打印当前进化进度
            print(f"=== Generation {t}/{self.maxGen} ===")
            # 合并父代P和子代Q，形成候选种群R
            self.R_population = self.combine_Pt_and_Qt(self.P_population, self.Q_population)
            # 使用NSGA3环境选择从R中选择下一代P
            self.P_population = self.environmental_selection(self.R_population, self.popSize)
            # 基于新的父代P生成子代Q（使用LLM算子）
            self.Q_population = self.make_new_population(self.P_population)
            # 持续更新外部存档，保存发现的所有非支配解
            self.update_EP_FromSet(self.EP, self.F_rank[1])
            # 递增代数
            t += 1

        # === 算法结束，整理最终结果 ===
        # 为外部存档中的每个个体设置临时适应度（用第一个目标值）
        for ep in self.EP:
            ep.temp_fitness = ep.fitness[0]
        # 按第一个目标值排序（可选的后处理步骤）
        test_fast = sorted(self.EP, key=lambda Individual: Individual.temp_fitness)
        # 提取所有个体的适应度值，形成最终的Pareto前沿
        EP_list = [copy.deepcopy(ind.fitness) for ind in test_fast]
        return EP_list

    def make_new_population(self, population):
        """
        生成新种群的方法，集成了LLM搜索算子
        替代传统的交叉和变异操作，使用智能化的LLM来生成子代

        Args:
            population: 当前父代种群

        Returns:
            生成的子代种群列表
        """
        # 初始化新种群列表
        new_population = []
        # 使用锦标赛选择从父代中选择用于繁殖的个体
        selected = self.tournamentSelectionOperator(population)
        # 两两配对生成子代（父代1 + 父代2 -> 子代1 + 子代2）
        for i in range(0, self.popSize, 2):
            # 选择第一个父代
            p1 = selected[i]
            # 选择第二个父代，如果索引超界则使用第一个个体
            p2 = selected[i + 1] if i + 1 < len(selected) else selected[0]
            # 使用LLM算子生成一对子代个体
            # 这里调用LLM的智能搜索功能，基于两个父代生成改进的子代
            c1, c2 = self.llm_operator.propose_offspring_pair(p1, p2)
            # === 处理B染色体的特殊逻辑 ===
            # 如果个体包含B类型的染色体（可能是特定问题的编码）
            if 'B' in p1.chromosome and 'B' in p2.chromosome:
                # 深拷贝父代的B染色体
                cb1 = copy.copy(p1.chromosome['B'])
                cb2 = copy.copy(p2.chromosome['B'])
                # 执行简单的单点交叉操作
                # 随机选择交叉点（确保至少为1）
                cpt = np.random.randint(1, len(cb1)) if len(cb1) > 1 else 1
                # 在交叉点前进行基因交换
                for j in range(cpt):
                    cb1[j], cb2[j] = cb2[j], cb1[j]
                # 将交叉后的B染色体赋给子代
                c1.chromosome['B'] = cb1
                c2.chromosome['B'] = cb2
            # 将第一个子代加入新种群
            new_population.append(c1)
            # 如果新种群还没满，加入第二个子代
            if len(new_population) < self.popSize:
                new_population.append(c2)
        # 计算新生成种群中所有个体的适应度值
        self.calculatePopulationFitness(new_population)
        return new_population


def generate_reference_points(num_objectives, divisions):
    """
    生成均匀分布的参考点
    这些参考点用于NSGA3的选择机制，帮助维持种群在目标空间中的多样性分布

    Args:
        num_objectives: 目标函数的数量
        divisions: 每个目标轴上的划分数量

    Returns:
        标准化后的参考点数组，每个参考点都在单位超平面上
    """

    def recursive_gen(current, left, depth):
        """
        递归生成参考点的辅助函数
        使用组合数学生成所有可能的整数组合

        Args:
            current: 当前正在构建的点的坐标列表
            left: 剩余可分配的总和
            depth: 当前递归的深度（对应目标维度）
        """
        # 如果到达最后一个维度，直接分配剩余的值
        if depth == num_objectives - 1:
            points.append(current + [left])
        else:
            # 递归地尝试所有可能的分配方案
            for i in range(left + 1):
                # 分配i给当前维度，剩余(left-i)继续递归
                recursive_gen(current + [i], left - i, depth + 1)
    # 存储所有生成的参考点
    points = []
    # 从空列表开始，总和为divisions，深度为0开始递归
    recursive_gen([], divisions, 0)
    # 将整数坐标转换为numpy数组并标准化到[0,1]范围
    # 除以divisions使所有坐标和为1（位于单位超平面上）
    ref_points = np.array(points) / divisions
    return ref_points


# ================= 主流程入口 =================
if __name__ == '__main__':
    from Utils import get_argument_parser, print_info, project_path

    args = get_argument_parser()
    print_info('************************  NSGA3-LLM ************************')
    print_info(
        '\n---------------------- Test instance:  [' + str(args.Nij) + ',' + str(args.K) + ']------------------------')

    instance_name = '[' + str(args.Nij) + ',' + str(args.K) + ']'
    args.save_dir = os.path.join(args.save_dir, instance_name, 'NSGA3-LLM')
    os.makedirs(args.save_dir, exist_ok=True)

    extra = "Running time: {} | popSize: {} | maxGen: {}".format(args.runTime, args.popSize, args.maxGen)
    print_info(extra, '\n')

    alg_dir = os.path.join(project_path, 'ExperimentResult', instance_name, 'NSGA3-LLM')
    PF_ref = None

    for I in range(1, args.runTime + 1):
        run_dir = os.path.join(alg_dir, str(I))
        if not os.path.isdir(run_dir):
            os.makedirs(run_dir)
        startTime = time.time()

        # 启用 LLM 版本
        nsgaiii = NSGA3(instance_name, args, use_llm=True, llm_caller=deepseek_call)

        if PF_ref:
            nsgaiii.PF_ref = PF_ref
        else:
            nsgaiii.PF_ref = None

        EP_list = nsgaiii.run()
        igd_save_path = os.path.join(run_dir, 'IGD_value.txt')
        with open(igd_save_path, 'w') as f:
            if hasattr(nsgaiii, 'final_IGD') and nsgaiii.final_IGD is not None:
                f.write(f'Final IGD value: {nsgaiii.final_IGD:.6f}\n')
            else:
                f.write('PF_ref (reference Pareto front) is not set, IGD not computed.\n')
        pd.DataFrame({'Time': np.array(EP_list)[:, 0],
                      'Energy': np.array(EP_list)[:, 1],
                      'Cost': np.array(EP_list)[:, 2]}).to_csv(os.path.join(run_dir, 'PF.csv'), index=False)
        fig = go.Figure(data=[go.Scatter3d(
            x=np.array(EP_list)[:, 0],
            y=np.array(EP_list)[:, 1],
            z=np.array(EP_list)[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=np.array(EP_list)[:, 2],
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title='Cost')
            )
        )])
        fig.update_layout(
            title='Pareto Frontier (NSGA-III)',
            scene=dict(
                xaxis_title='Time',
                yaxis_title='Energy',
                zaxis_title='Cost'
            ),
            width=800,
            height=600
        )
        fig.show()