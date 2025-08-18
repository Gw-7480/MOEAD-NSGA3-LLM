'''''
Gong Y, Bian K, Hao F, et al.
Dependent tasks offloading in mobile edge computing:
a multi-objective evolutionary optimization strategy[J].
Future Generation Computer Systems, 2023, 148: 314-325
'''''
import math

from scipy.special import jv  # 用于Bessel函数J_0
import pandas as pd
import os
import cmath
import numpy as np
import math
import plotly.graph_objects as go
from Utils import *
import os, time, random
from pandas import DataFrame

np.random.seed(10)
random.seed(10)

class MOEAD_1:
    def __init__(self, instance_name, args):
        self.popSize = args.popSize
        self.maxGen = args.maxGen
        self.T = args.T     # 邻居数量
        self.pc = args.pc   # 交叉概率
        self.pmApp = args.pmApp # 对应用的变异概率
        self.pmBit = args.pmBit # 对每个基因位的变异概率

        self.VT = {}  # 权重向量集合
        self.B = {}  # 权向量的邻居
        self.population = []
        self.Z = []  # 参考点
        self.F_rank = []  # 将种群非支配排序分层, 用种群中的个体的下标来表示，一个元素表示第一层,下标从1开始
        self.PF_history = []            #每一代的历史最优Pareto Front
        self.EP = []                    #保存当前代的历史非支配解

        #---------------------------------------Problem Notation----------------------------------------
        self.instance_name = instance_name
        self.Nij = args.Nij  # 每个依赖任务的子任务数量
        self.K = args.K  # 核的数量
        self.d_i_list = self.load_task_input_data_size(instance_name)
        self.num_objectives = 3  # 或3，视你的目标数而定

        self.workflow_set = []
        self.get_workflow_set(project_path +'/instance/' + instance_name)
        self.UAV_transmission_rate_set = self.get_UAV_transmission_rate()

        self.a = [0.2, 0.4, 0.6, 0.8, 1]  # The frequency scaling factors
        self.M = len(self.a)  # M different frequency levels.
        self.IGDValue = None
        self.IGD_list = []     # 保存100代的IGD值
        # self.PF_ref = self.get_PF_ref()

        self.M = args.num_satellites  # 卫星数量（文档中M颗）
        self.satellites = []
        # 初始化每颗卫星的基础参数（文档中“每颗卫星配备云服务器”）
        for m in range(self.M):
            # 每颗卫星轨道高度略有差异（800±50km），体现多卫星分布
            orbit_height = 800 + np.random.uniform(-50, 50)
            # 卫星运行速度（近地轨道约7.8km/s）
            v_l = 7.8 + np.random.uniform(-0.2, 0.2)
            self.satellites.append({
                "id": m,
                "orbit_height": orbit_height,
                "v_l": v_l,  # 运行速度（km/s）
                "f_sat": 1.0e9  # 计算频率（文档中卫星CPU频率）
            })


    def run(self):
        self.initializeWeightVectorAndNeighbor()
        self.initializePopulation()
        self.initializeReferencePoint()
        self.fast_non_dominated_sort(self.population)
        self.initializeEP(self.F_rank[1])

        t = 1
        # 新增：保存每一代PF（Pareto Front，帕累托前沿）的路径
        pf_gen_dir = os.path.join(project_path, 'ExperimentResult', self.instance_name, 'MOEAD_1_multisat', '1', 'PF_gen')
        os.makedirs(pf_gen_dir, exist_ok=True)  # 如果目录不存在则创建

        # 主进化循环，迭代直到达到最大代数
        while (t <= self.maxGen):
            # 对种群中的每一个个体进行操作
            for i in range(self.popSize):
                y_ = self.reproduction(i)  # 产生新个体（交叉、变异等）
                self.updateNeighborSolutions(i, y_)  # 用新个体更新邻居解
                self.updateReferencePoint(y_)        # 更新参考点
                self.update_EP_FromElement(self.EP, y_)  # 用新个体更新EP（外部种群/精英集）

            # PF_know = [copy.deepcopy(ind.fitness) for ind in self.EP]  # 可选：保存当前EP的适应度

            # 新增：保存每一代EP的归一化适应度（用于后续性能指标计算，如IGD）
            pf_gen = [copy.deepcopy(ind.fitness) for ind in self.EP]  # 当前EP中所有个体的适应度
            # 归一化处理（与IGD.py一致，需传入PF_ref作为参考帕累托前沿）
            if hasattr(self, 'PF_ref') and self.PF_ref:
                from MODTSRA.IGD import normalize_pf
                all_points = np.array(self.PF_ref + pf_gen)  # 合并参考前沿和当前前沿
                min_vals = np.min(all_points, axis=0)        # 每个目标的最小值
                max_vals = np.max(all_points, axis=0)        # 每个目标的最大值
                norm_pf_gen, _, _ = normalize_pf(pf_gen, min_vals, max_vals)  # 归一化
            else:
                norm_pf_gen = pf_gen  # 若无参考前沿，则不归一化

            # 保存当前代的归一化帕累托前沿到CSV文件
            import pandas as pd
            pd.DataFrame(norm_pf_gen, columns=['Time', 'Energy', 'Cost']).to_csv(
                os.path.join(pf_gen_dir, f'PF_gen_{t}.csv'), index=False
            )

            # 每20代输出一次进度信息
            if t % 20 == 0:
                print('Generation ', t)
            t += 1  # 进入下一代

        # 进化结束后，对EP中的个体按第一个目标值（如Time）升序排序
        for ep in self.EP:
            ep.temp_fitness = ep.fitness[0]
        test_fast = sorted(self.EP, key=lambda Individual: Individual.temp_fitness)
        EP_list = [copy.deepcopy(ind.fitness) for ind in test_fast]  # 提取排序后的适应度列表
        return EP_list # 返回最终的非支配解集（帕累托前沿）

    """
        **********************************************run**********************************************
    """
    def initializeEP(self, F_rank):
        for ind in F_rank:
            self.EP.append(copy.deepcopy(ind))


    def initializeWeightVectorAndNeighbor(self):
        # # 可选：从文件读取所有权重向量（已注释掉）
        # all_weights = pd.read_csv(project_path + '/instance/all_weights.csv').values  # DataFrame转换成numpy.array
        # for i in range(self.popSize):
        #     self.VT[i] = list(all_weights[i])

        H = self.popSize - 1  # H为权重向量的数量减一
        # 随机生成popSize个权重向量，每个向量的元素和为1
        for i in range(0, H + 1):
            w = []          # 当前权重向量
            remain = 1.0    # 剩余可分配的权重
            # 依次为每个目标分配权重，最后一个目标分配剩余权重
            for j in range(self.num_objectives - 1):
                wj = np.random.rand() * remain  # 随机分配一部分权重
                w.append(wj)
                remain -= wj
            w.append(remain)  # 最后一个目标分配剩余权重
            self.VT[i] = w    # 保存权重向量

        # 计算每个权重向量的T个最近邻
        for i in self.VT.keys():
            distance = []
            for j in self.VT.keys():
                if(i != j):
                    tup = (j, self.getDistance(self.VT[i], self.VT[j]))  # 计算i和j之间的欧氏距离
                    distance.append(tup)
            distance = sorted(distance, key=lambda x: x[1])  # 按距离升序排序
            neighbor = []
            for j in range(self.T):
                neighbor.append(distance[j][0])  # 取最近的T个邻居的索引
            self.B[i] = neighbor  # 保存邻居索引列表


    def initializePopulation(self):
        for i in range(self.popSize):
            ind = Individual()
            ratio_list = self.get_random_resource_allocation_ratio(N_uav)
            uav_set = []
            for j in range(N_uav):
                uav = UAV(self.K)
                uav.workflow = copy.deepcopy(self.workflow_set[j])
                uav.workflow.order = self.initializeWorkflowSequence(uav.workflow)
                uav.workflow.location = [random.randint(1, self.K + 1) for _ in range(self.Nij)]
                uav_set.append(uav)
                uav.ratio = ratio_list[j]
                uav.R_i = self.UAV_transmission_rate_set[j]
            B = [random.randint(0, 1) for _ in range(N_service)]
            ind.chromosome = {'S': uav_set, 'A': ratio_list, 'B': B}
            self.calculateFitness(ind)
            self.population.append(ind)


    def initializeReferencePoint(self):
        self.Z = []
        num_objectives = self.num_objectives  # 假设你在 __init__ 里定义了 self.num_objectives = 3
        for i in range(num_objectives):
            fitness_i = [ind.fitness[i] for ind in self.population]
            self.Z.append(min(fitness_i))


    def reproduction(self, i):
        # 从个体i的邻居中随机选择两个邻居索引k和l，用于生成新个体
        k = random.choice(self.B[i])
        l = random.choice(self.B[i])

        # 创建两个新个体ind_k和ind_l，分别作为交叉和变异的父代
        ind_k = Individual()
        ind_l = Individual()

        # 构建ind_k的UAV基因集合
        uav_k_set = []
        for gene in self.population[k].chromosome['S']:
            uav_k = copy.deepcopy(gene)  # 深拷贝父代k的UAV基因，避免原始个体被修改
            self.reInitialize_WorkflowTaskSet_Schedule(uav_k)  # 重新初始化UAV的任务调度
            uav_k_set.append(uav_k)
        # 组装ind_k的染色体，包括UAV集合、资源分配向量A、服务缓存向量B
        ind_k.chromosome = {
            'S': uav_k_set,
            'A': copy.copy(self.population[k].chromosome['A']),
            'B': copy.copy(self.population[k].chromosome['B'])
        }

        # 构建ind_l的UAV基因集合
        uav_l_set = []
        for gene in self.population[l].chromosome['S']:
            uav_l = copy.deepcopy(gene)  # 深拷贝父代l的UAV基因
            self.reInitialize_WorkflowTaskSet_Schedule(uav_l)  # 重新初始化UAV的任务调度
            uav_l_set.append(uav_l)
        # 组装ind_l的染色体
        ind_l.chromosome = {
            'S': uav_l_set,
            'A': copy.copy(self.population[l].chromosome['A']),
            'B': copy.copy(self.population[l].chromosome['B'])
        }

        # 对两个新个体进行交叉操作，产生新的基因组合
        self.crossoverOperator(ind_k, ind_l)
        # 对两个新个体分别进行变异操作，增加多样性
        self.mutantOperator(ind_k)
        self.mutantOperator(ind_l)
        # 计算两个新个体的适应度
        self.calculateFitness(ind_k)
        self.calculateFitness(ind_l)
        # 在ind_k和ind_l中选择表现更优的个体作为最终的后代返回
        return self.select_best_individual(ind_k, ind_l)


    def select_best_individual(self, ind_1, ind_2):
        if self.isDominated(ind_1.fitness, ind_2.fitness):
            return ind_1
        elif self.isDominated(ind_2.fitness, ind_1.fitness):
            return ind_2
        else:
            if random.randint(0,1) == 0:
                return ind_1
            else:
                return ind_2


    def crossoverOperator(self, ind_k, ind_l):
        # 先交叉UAV的执行顺序和执行位置向量
        for i in range(N_uav):
            gene_1 = ind_k.chromosome['S'][i]
            gene_2 = ind_l.chromosome['S'][i]
            cpt = random.randint(0, len(gene_1.workflow.location) - 1)
            cPart_1 = []  # 保存第一个个体的执行顺序的从开始到交叉点的片段
            cPart_2 = []  # 保存第二个个体的执行顺序的从开始到交叉点的片段
            # 执行位置交叉
            for j in range(0, cpt):
                gene_1.workflow.location[j], gene_2.workflow.location[j] = gene_2.workflow.location[j], gene_1.workflow.location[j]
                cPart_1.append(gene_1.workflow.order[j])
                cPart_2.append(gene_2.workflow.order[j])
            # 执行顺序交叉
            for j in range(len(cPart_1)):
                gene_2.workflow.order.remove(cPart_1[j])  # 在个体二中移除第一个个体的交叉片段
                gene_1.workflow.order.remove(cPart_2[j])  # 在个体一中移除第二个个体的交叉片段
            gene_1.workflow.order = cPart_2 + gene_1.workflow.order
            gene_2.workflow.order = cPart_1 + gene_2.workflow.order

        # 交叉资源分配向量
        cpt = random.randint(1, len(ind_k.chromosome['A']))
        for j in range(0, cpt):
            u_j = random.random()
            if u_j <= 0.5:
                r_j = pow(2*u_j, 1./2)
            else:
                r_j = pow(1/(2*(1-u_j)), 1./2)
            ind_k.chromosome['A'][j] = 0.5 * ((1 + r_j)*ind_k.chromosome['A'][j]+
                                              (1 - r_j)*ind_l.chromosome['A'][j])
            ind_l.chromosome['A'][j] = 0.5 * ((1 - r_j) * ind_k.chromosome['A'][j] +
                                              (1 + r_j) * ind_l.chromosome['A'][j])
        ind_k.chromosome['A'] = np.abs(ind_k.chromosome['A']) / np.linalg.norm(ind_k.chromosome['A'], ord=1) # 使得比例加起来等于1

        # 交叉服务缓存向量
        cpt = random.randint(1, len(ind_k.chromosome['B']))
        for j in range(0, cpt):
            ind_k.chromosome['B'][j], ind_l.chromosome['B'][j] = ind_l.chromosome['B'][j], ind_k.chromosome['B'][j]

    def mutantOperator(self, ind):
        # 先变异UAV的执行顺序和执行位置向量
        for gene in ind.chromosome['S']:
            rnd_UAV = random.random()
            if (rnd_UAV < 1.0 / N_uav):  # 针对每一个基因（SMD）判断是否变异
                for i in range(gene.workflow.location.__len__()):
                    rnd_bit = random.random()
                    if (rnd_bit < 1.0 / (self.Nij)):
                        pos = gene.workflow.location[i]
                        rand = list(np.arange(1,self.K+2))
                        rand.remove(pos)
                        gene.workflow.location[i] = random.choice(rand)

                r = random.randint(1, gene.workflow.order.__len__() - 2)  # 随机选择一个变异位置
                formerSetPoint = []
                rearSetPoint = []
                for i in range(0, gene.workflow.order.__len__() - 1):  # 从前往后直到所有的前驱任务都被包含在formerSetPoint中
                    formerSetPoint.append(gene.workflow.order[i])
                    if set(gene.workflow.taskSet[r].preTaskSet).issubset(set(formerSetPoint)):
                        break
                for j in range(gene.workflow.order.__len__() - 1, -1, -1):  # 从后往前直到所有的后继任务都被包含在rearSetPoint中
                    rearSetPoint.append(gene.workflow.order[j])
                    if set(gene.workflow.taskSet[r].sucTaskSet).issubset(set(rearSetPoint)):
                        break
                rnd_insert_pt = random.randint(i + 1, j - 1)  # 从i+1到j-1之间随机选一个整数
                gene.workflow.order.remove(r)  # 移除变异任务
                gene.workflow.order.insert(rnd_insert_pt, r)  # 在随机生成的插入点前插入r

        # 变异资源分配比例
        for j in range(len(ind.chromosome['A'])):
            r = random.random()
            if r <= 1./ len(ind.chromosome['A']):
                ind.chromosome['A'][j] = ind.chromosome['A'][j] + 0.8 * (1-0)*random.random()
        ind.chromosome['A'] = np.abs(ind.chromosome['A']) / np.linalg.norm(ind.chromosome['A'], ord=1)

        # 变异服务缓存决策
        for j in range(len(ind.chromosome['B'])):
            r = random.random()
            if r <= 1. / len(ind.chromosome['B']):
                ind.chromosome['B'][j] = 1 - ind.chromosome['B'][j]

    def updateReferencePoint(self, y_):
        for j in range(len(y_.fitness)):
            if(self.Z[j] > y_.fitness[j]):
                self.Z[j] = y_.fitness[j]


    def updateNeighborSolutions(self, i, y_):
        for j in self.B[i]:
            y_g_te = self.getTchebycheffValue(j, y_)
            neig_g_te = self.getTchebycheffValue(j, self.population[j])
            if(y_g_te <= neig_g_te):
                self.population[j] = y_


    def update_EP_FromElement(self, EP, ind):  #用新解ind来更新EP
        if EP == []:
            EP.append(copy.deepcopy(ind))
        else:
            i = 0
            while (i < len(EP)):  # 判断ind是否支配EP中的非支配解，若支配，则删除它所支配的解
                if (self.isDominated(ind.fitness, EP[i].fitness) == True):
                    EP.remove(EP[i])
                    i -= 1
                i += 1
            for ep in EP:
                if (self.isDominated(ep.fitness, ind.fitness) == True):
                    return None
            if (self.isExist(ind, EP) == False):
                EP.append(copy.deepcopy(ind))


    def isExist(self, ind, EP):   #判断个体ind的适应度是否与EP中某个个体的适应度相对，若相等，则返回True
        for ep in EP:
            if ind.fitness == ep.fitness: # 判断两个列表对应元素的值是否相等
                return True
        return False


    def getTchebycheffValue(self, index, ind):  #index是fitness个体的索引，用来获取权重向量
        g_te = []
        for i in range(len(ind.fitness)):
            temp = self.VT[index][i] * abs(ind.fitness[i] - self.Z[i])
            g_te.append(temp)
        return max(g_te)


    def fast_non_dominated_sort(self, population):
        for p in population:
            p.S_p = []
            p.rank = None
            p.n = 0

        self.F_rank = []
        F1 = []  # 第一个非支配解集前端
        self.F_rank.append(None)
        for p in population:
            for q in population:
                if self.isDominated(p.fitness, q.fitness):
                    p.S_p.append(q)
                elif self.isDominated(q.fitness, p.fitness):
                    p.n += 1
            if (p.n == 0):
                p.rank = 1
                F1.append(p)
        self.F_rank.append(F1)

        i = 1
        while (self.F_rank[i] != []):
            Q = []
            for p in self.F_rank[i]:
                for q in p.S_p:
                    q.n -= 1
                    if (q.n == 0):
                        q.rank = i + 1
                        Q.append(q)

            if(Q != []):
                i += 1
                self.F_rank.append(Q)
            else:
                break


    def isDominated(self, fitness_1, fitness_2):  # 前者是否支配后者
        flag = -1
        for i in range(len(fitness_1)):
            if fitness_1[i] < fitness_2[i]:
                flag = 0
            if fitness_1[i] > fitness_2[i]:
                return False
        if flag == 0:
            return True
        else:
            return False


    def get_random_resource_allocation_ratio(self, dim):
        w_kept = np.random.randn(dim)  # 从标准正太分布中抽取reward_size个随机数
        w_kept = np.abs(w_kept) / np.linalg.norm(w_kept, ord=1)  # 求1范数
        return np.round(w_kept, 8)

    def initializeWorkflowSequence(self, workflow):
        S = []  # 待排序的任务集合
        R = []  # 已排序任务
        T = []
        R.append(workflow.entryTask)
        for task in workflow.taskSet:
            T.append(task.id)
        T.remove(workflow.entryTask)

        while T != []:
            for t in T:
                if set(workflow.taskSet[t].preTaskSet).issubset(set(R)):  #判断t的前驱节点集是否包含在R中
                    if t not in S:
                        S.append(t)
            ti = random.choice(S) #随机从S中选择一个元素
            S.remove(ti)
            T.remove(ti)
            R.append(ti)
        return R

    def calculateLocalEnergy(self, instance_name, task_idx):
        """
        计算本地设备执行单个任务的能耗
        参数:
            instance_name (str): 实例名称
            task_idx (int): 任务索引
        返回:
            float: 单个任务的本地能耗（焦耳）
        """
        F_M = 0.6 * 1e9  # 本地设备计算能力 (Hz)
        KAPPA = pow(10, -26)  # 有效电容系数

        # 1. 读取所有子任务的CPU周期数
        cpu_cycles_list = self.load_task_cpu_cycles(instance_name, self.Nij)
        task_number = len(cpu_cycles_list)
        # 2. 索引修正，防止越界
        if task_idx >= task_number:
            raise IndexError(f"task_idx {task_idx} 超出任务数量 {task_number}")

        F_i_G = cpu_cycles_list[task_idx]  # 当前子任务的CPU周期数，单位：Gcycles

        # 3. 计算能耗
        energy = KAPPA * (F_i_G * 1e9) * (F_M ** 2)
        return float(energy)

    def calculate_all_local_tasks_total_time(self, workflow, cpu_cycles_list, md_frequency=0.6):
        """
        计算 workflow 中所有子任务在本地设备串行执行的总完成时间（考虑依赖关系）
        :param workflow: Workflow 对象（包含所有任务）
        :param cpu_cycles_list: 所有任务的CPU周期数列表（单位：Gcycles）
        :param md_frequency: MD的计算能力（GHz），默认为0.6
        :return: 总完成时间（float）
        """
        # 假设只有一个本地核心，维护核心的可用时间线
        core_time_point = [0.0]

        # 遍历 workflow 的调度顺序
        for task_id in workflow.order:
            task = workflow.taskSet[task_id]

            # 1. 计算就绪时间（所有前驱任务的完成时间最大值）
            ready_times = []
            for pre_id in task.preTaskSet:
                pre_task = workflow.taskSet[pre_id]
                ready_times.append(pre_task.FT_i_l if pre_task.FT_i_l is not None else 0)
            RT_i_l = max(ready_times) if ready_times else 0

            # 2. 实际开始时间 = max(就绪时间, 上一个任务完成时间)
            start_time = max(RT_i_l, core_time_point[-1])

            # 3. 执行时间
            c_i = cpu_cycles_list[task.id]  # 单位：Gcycles
            exec_time = c_i / md_frequency  # (Gcycles / GHz) = 秒

            # 4. 完成时间
            finish_time = start_time + exec_time

            # 5. 更新任务对象
            task.RT_i_l = RT_i_l
            task.ST_i_l = start_time
            task.FT_i_l = finish_time

            # 6. 更新核心时间线
            core_time_point.append(finish_time)

        # 总完成时间为最后一个任务的完成时间
        exit_task = workflow.taskSet[workflow.exitTask]
        return exit_task.FT_i_l

    def load_task_cpu_cycles(self, instance_name, task_number):
        """从指定实例文件夹中读取任务CPU周期数，单位是G
        参数:
            instance_name (str): 实例名称，对应实例文件夹名。
            task_number (int): 任务数量。
        返回:
            list of float: 长度为 task_number 的任务 CPU 周期数列表，索引对应任务 ID。
        """
        # 构建文件路径
        project_root = os.path.dirname(os.path.abspath(__file__))
        cpu_cycles_path = os.path.join(project_root, 'instance', instance_name, 'task_CPU_cycles_number.csv')

        cpu_cycles_df = pd.read_csv(cpu_cycles_path)

        # 提取 CPU 周期数并存入列表
        cpu_cycles_list = []
        for i in range(task_number):
            cpu_cycles_list.append(float(cpu_cycles_df.values[i][0]))

        return cpu_cycles_list
        ######## 单例读取的数据大小 后期如果需要之后在修改 ####

    def load_task_input_data_size(self, instance_name):
        """
        读取指定实例下所有任务的输入数据大小（D_i）。其中是以KB为单位
        :param instance_name: 例如 '[20,6]'
        :return: d_i_list，每个任务的输入数据大小（float）组成的列表
        """
        # 获取当前脚本的目录
        project_root = os.path.dirname(os.path.abspath(__file__))
        # 拼接完整路径
        path = os.path.join(project_root, 'instance', instance_name, 'task_input_data_size.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)
        d_i_list = [float(val[0]) for val in df.values]
        return d_i_list

        ######## 单例读取的数据大小 后期如果需要之后在修改 ####

    def calculate_ground_to_uav_transmission_energy(self, d_mn, task_idx, K=10, B=0.001, p_G=0.1, sigma_2=1e-6):
        """
        计算从地面用户到无人机的传输能耗 E_Gn 和传输时间 T_mn
        参数:
            d_mn (float): 地面用户与无人机的距离（单位：千米）
            task_idx (int): 任务索引（用于查找对应的 D_i）
            K (float): Rician 因子，默认10
            B (float): 信道带宽，单位：GHz（如0.001为1MHz）
            p_G (float): 地面用户发射功率 (W)
            sigma_2 (float): 噪声功率
        返回:
            tuple: (传输能耗 E_Gn, 传输时间 T_mn)
        """
        # 获取对应任务的输入数据大小 D_i
        D_i_KB = self.d_i_list[task_idx]
        D_i_bit = D_i_KB * 1024  # 转换为 bit

        # 距离单位转换：千米 -> 米
        d_mn_m = d_mn * 1000

        # 随机选择信道增益（单位：dB）
        gain_db_choices = [-18, -16, -14, -12, -10, -8]
        gain_db = np.random.choice(gain_db_choices)
        g_xi = 10 ** (gain_db / 10)  # dB转线性

        # 计算传输速率 R_Gn
        B = 0.6e6  # 0.6 MHz
        p_G = 0.5  # W
        sigma_2 = 0.6e-6  # W

        R_Gn = B * np.log2(1 + (p_G * g_xi) / sigma_2)

        # 计算传输时间 T_mn
        T_mn = D_i_bit / R_Gn

        # 计算传输能耗 E_Gn
        E_Gn = p_G * T_mn

        return E_Gn, T_mn

    def calculatePopulationFitness(self, population):
        for ind in population:
            self.calculateFitness(ind)

        # 调用 self.calculateFitness(ind) 方法来计算个体 ind 的适应度。
        # 适应度通常是根据个体的性能指标（如时间、能量消耗等）进行评估的

    def calculate_total_energy_consumption(self, U_n, tasks):
        total_energy_consumption = 0
        for task in tasks:
            # 修正：如果exePosition为None，尝试从workflow中获取
            if task.exePosition is None and hasattr(U_n, 'workflow'):
                try:
                    idx = U_n.workflow.order.index(task.id)
                    task.exePosition = U_n.workflow.location[idx]
                except Exception:
                    raise ValueError(f"Cannot determine exePosition for task {task.id}")
            if task.exePosition is None:
                raise ValueError(f"Task {task.id} exePosition is None!")
            f_max = U_n.ratio * U_n.coreCC[task.exePosition]
            p_max = U_n.pws_i
            p_min = U_n.pwr_i
            cycles = task.M_i_j * 1e9  # 转换为周期数
            freq_hz = f_max * 1e9  # 转换为 Hz
            time_sec = cycles / freq_hz  # 得到秒
            energy_consumption = time_sec * (p_max + p_min)  # 焦耳
            total_energy_consumption += energy_consumption
        return total_energy_consumption

    def calculate_total_execution_delay(self, U_n, tasks, workflow):
        """
        计算无人机 U_n 在执行所有子任务时的总延时（多核心调度）
        :param U_n: 无人机对象
        :param tasks: 任务列表
        :param workflow: Workflow对象（包含order、exitTask等）
        :return: 计算得到的总延时
        """
        # 初始化每个核心的时间线
        core_time_point = {core: [0.0] for core in U_n.coreCC.keys()}
        finish_times = {}

        for task_id in workflow.order:
            task = workflow.taskSet[task_id]
            core = task.exePosition  # 任务分配到的核心编号

            # 1. 计算就绪时间（所有父任务的完成时间最大值）
            if task.preTaskSet:
                ready_times = [finish_times[pre_id] for pre_id in task.preTaskSet]
                RT_i_uav = max(ready_times)
            else:
                RT_i_uav = 0.0

            # 2. 实际开始时间 = max(就绪时间, 该核心上上一个任务完成时间)
            start_time = max(RT_i_uav, core_time_point[core][-1])

            # 3. 执行时间 D_{n,j,k} = M_i / (δ_j * f_{n,j,k}^{max})
            exec_time = task.M_i_j / (U_n.ratio * U_n.coreCC[core])

            # 4. 完成时间
            finish_time = start_time + exec_time

            # 5. 记录
            finish_times[task_id] = finish_time
            core_time_point[core].append(finish_time)

        # 返回出口任务的完成时间
        return finish_times[workflow.exitTask]

    def calculate_individual_energy(self, ind, index):
        total_energy = 0

        # 1. 本地设备能耗（遍历所有任务） #########不知道是否正确，还需再仔细检查
        for task_idx in range(self.Nij):
            local_energy = self.calculateLocalEnergy(self.instance_name, task_idx)
            total_energy += local_energy

        # 2. 地面到无人机传输能耗
        transmission_energy = 0
        for idx in range(self.Nij):
            d_mn = 0.03  # 获取地面到无人机的距离（单位：千米）
            E_Gn, _ = self.calculate_ground_to_uav_transmission_energy(d_mn,
                                                                       idx)  # 修正：应接收两个返回值########在这里的ricial因子需要再查看一下
            transmission_energy += E_Gn
        total_energy += transmission_energy

        # 3. 无人机执行子任务的计算能耗总和（严格按照公式）
        # 2. 只计算 ind.chromosome['S'][uav_index] 这个无人机的所有子任务能耗
        uav = ind.chromosome['S'][index]
        tasks = [uav.workflow.taskSet[task_id] for task_id in uav.workflow.order]
        uav_energy = self.calculate_total_energy_consumption(uav, tasks)
        total_energy += uav_energy

        return total_energy

    def calculate_ground_execution_delay(self, ind, uav, workflow):
        """
        计算地面用户卸载到无人机的传输时间 + 无人机执行所有子任务的延时
        :param uav: UAV对象
        :param workflow: Workflow对象
        :return: 总延时（float）
        """
        self.calculateWorkflowTimeEnergy(ind, uav, workflow)
        total_delay = 0
        # 1. 地面到无人机的传输时间（所有子任务累加）
        for task_id in workflow.order:
            d_mn = 0.03
            _, T_mn = self.calculate_ground_to_uav_transmission_energy(d_mn, task_id)
            total_delay += T_mn
        # 2. 无人机执行所有子任务的延时
        uav_delay = self.calculate_total_execution_delay(uav, [workflow.taskSet[tid] for tid in workflow.order],
                                                         workflow)
        total_delay += uav_delay

        return total_delay
        ########   卫星  #########

    def calculate_angle_and_service_time(self):
        """
        随机生成无人机到卫星的直线距离，计算仰角和服务时间
        :return: l_GS（无人机到卫星的直线距离）, phi（仰角，弧度）, theta（中心角，弧度）, service_time
        """
        I_E = 6370  # 地球半径 (km)
        I_o = 800  # 卫星轨道高度 (km)
        min_phi = math.radians(40)  # 最小仰角 40°
        # 随机生成无人机到卫星的直线距离 l_GS
        l_GS = np.random.uniform(800, 1250)
        # 由余弦定理反推仰角 phi
        # l_GS^2 = (I_E + I_o)^2 + I_E^2 - 2*I_E*(I_E + I_o)*cos(phi)
        cos_phi = ((I_E + I_o) ** 2 + I_E ** 2 - l_GS ** 2) / (2 * I_E * (I_E + I_o))
        cos_phi = np.clip(cos_phi, -1, 1)
        phi = math.acos(cos_phi)
        # 保证仰角不小于最小仰角
        if phi < min_phi:
            phi = min_phi
            l_GS = math.sqrt((I_E + I_o) ** 2 + I_E ** 2 - 2 * I_E * (I_E + I_o) * math.cos(phi))
        # 计算地心中心角 theta
        theta = math.acos((I_E + I_o) / l_GS * math.sin(phi))
        # 服务时间可根据实际业务逻辑定义，这里假设与中心角成正比
        service_time = theta  # 或根据实际需求调整
        return l_GS, phi, theta, service_time

    def calculate_effective_communication_time(self, phi, v_l):
        """
        计算无人机与近地轨道卫星之间的有效通信时间 T_{U_n S_l}
        参数:
            phi: 卫星与无人机之间的仰角（弧度）
            v_l: 近地轨道卫星的运行速度 (km/s)
        返回:
            有效通信时间 T_{U_n S_l}
        """
        I_E = 6370  # 地球半径 (km)
        I_o = 800  # 卫星轨道高度 (km)
        # 计算覆盖弧长
        L_S = 2 * (I_E + I_o) * phi
        T_Un_Sl = L_S / v_l
        return T_Un_Sl

    def get_available_satellites(self, current_time):
        """筛选当前时刻处于覆盖窗口内的卫星，确保返回的卫星信息包含f_sat"""
        available_sats = []
        for sat in self.satellites:
            l_GS, phi, theta, service_time = self.calculate_angle_and_service_time()
            T_Un_Sl = self.calculate_effective_communication_time(phi, sat["v_l"])
            if "f_sat" not in sat:
                sat["f_sat"] = 1.0e9  # 默认频率
            if T_Un_Sl > 0:
                available_sats.append({
                    "satellite": sat,
                    "distance": l_GS,
                    "effective_time": T_Un_Sl,
                    "phi": phi
                })
        return available_sats

    def get_max_doppler_shift_ka(self, d_nl):
        """
        根据d_nl（km）返回Ka频段最大多普勒频移（Hz）
        """
        if d_nl < 900:
            return 662e3  # 800 km
        elif d_nl < 1100:
            return 635e3  # 1000 km
        elif d_nl < 1350:
            return 610e3  # 1200 km
        else:
            return 576e3  # 1500 km

        # 现在R_nl容易偏小 需要后期再重新看 因为B_n传参一直是0.01 所以就会导致R_nl偏小，再运行两遍看看是否一直正确 现在R_nl大概是几百万或者几千万，

    def calculate_sat_uav_channel_and_rate(
            self,
            delta_l=25,  # 卫星波束增益，dB → 需转线性
            wavelength=0.01,  # 信号波长，单位：米（保持SI单位）
            d_nl=None,  # 无人机到卫星距离，单位：千米 → 转米
            phi_angle=None,  # 相位偏移，弧度（无量纲）
            f_hat_n=None,  # 最大多普勒频率，单位：Hz（原GHz转Hz）
            T_nl=None,  # 传输时延，单位：秒（保持SI单位）
            B_n=0.01,  # 信道带宽，单位：GHz
            P_U=1.0,  # 无人机发射功率，单位：W（保持SI单位）
            g_nl=None,
            interference_h_list=None,
            sigma_l_sq=None
    ):
        # === 单位转换预处理 ===
        # 1. 卫星波束增益 dB转线性
        delta_l_linear = 10 ** (delta_l / 10)
        # 25dB → 316.23
        B_n_z = B_n * 1e9  # 将GHz转换为Hz

        # 2. 距离单位转换（千米→米）
        if d_nl is not None:
            d_nl_m = d_nl * 1000  # 千米转米
        else:
            d_nl_m = np.random.uniform(800e3, 1250e3)  # 800-1250 km → 800000-1250000米
        if phi_angle is None:
            phi_angle = np.random.uniform(0, 2 * np.pi)  # 随机一个相位偏移，单位弧度
        if T_nl is None:
            T_nl = d_nl_m / 3e8  # 光速c=3e8 m/s

        # 3. 频率单位转换（GHz→Hz已在参数输入层处理）
        # f_hat_n = 0.1 GHz → 1e8 Hz
        # B_n = 0.01 GHz → 1e7 Hz

        # 4. 噪声功率计算（使用SI单位）（在原论文中 sigma也可计算为-100dB）
        if sigma_l_sq is None:
            k_B = 1.38e-23  # 玻尔兹曼常数，J/K
            zeta = 354.81  # 噪声温度，K
            sigma_l_sq = k_B * zeta * B_n_z  # 单位：W (J/s)

        f_hat_n = self.get_max_doppler_shift_ka(d_nl)
        # === 信道计算（使用SI单位） ===
        # 1. 平均信道系数计算
        phase_shift = cmath.exp(1j * phi_angle)
        h_nl_bar = (np.sqrt(delta_l_linear) * wavelength) / (4 * np.pi * d_nl_m) * phase_shift

        if g_nl is None:
            # 让g_nl的方差与h_nl_bar一致
            var = np.abs(h_nl_bar) ** 2 * 5
            std = np.sqrt(var / 2)  # 实部和虚部各占一半
            g_nl = np.random.normal(0, std) + 1j * np.random.normal(0, std)

        # 2. 多普勒效应计算（使用Hz单位）
        epsilon = jv(0, 2 * np.pi * f_hat_n * T_nl)  # f_hat_n已转Hz

        # 3. 时延CSI信道系数
        h_nl = epsilon * h_nl_bar + np.sqrt(1 - epsilon ** 2) * g_nl

        # 用 h_nl 近似 interference_h_list
        interference_h_list = [h_nl]

        # 4. 信号与干扰功率计算
        signal_power = P_U * np.abs(h_nl) ** 2  # 单位：W
        interference_power = sum([P_U * np.abs(h) ** 2 for h in interference_h_list])

        # 5. 速率计算（使用Hz单位带宽）
        SINR = signal_power / (sigma_l_sq + interference_power)
        R_nl = B_n_z * np.log2(1 + SINR)  # 单位：bit/s

        R_nl = max(R_nl, 1e7)  # 不低于10 Mbps
        R_nl = min(R_nl, 1e8)  # 不高于1Gp

        return h_nl_bar, epsilon, h_nl, R_nl

        #### 代码在转卫星的逻辑还没有实现  ######!!!!!

    def calculate_offloading_transmission_time(
            self,
            mu_m, mu_n, mu_l,  # μ_m, μ_n, μ_l
            task_idx,  # 新增
            d_nl=None,  # 无人机到卫星距离，单位：千米
            b=1.2,  # 传输开销因子
            W_i=None,  # 任务数据量，单位：bit
            # 以下为信道参数，均为GHz、千米等标准单位
            delta_l=25,  # 卫星波束增益，dB
            wavelength=0.01,  # 信号波长，米
            phi_angle=None,  # 相位偏移，弧度
            f_hat_n=0.1,  # 最大多普勒频率，GHz
            T_nl=None,  # 传输时延，秒
            B_n=0.01,  # 信道带宽，GHz
            P_U=1.0,  # 无人机发射功率，W
            g_nl=None,  # 复高斯随机变量
            interference_h_list=None,  # 干扰信道系数列表
            sigma_l_sq=None,  # 卫星AWGN噪声方差
            c=3e8  # 光速，m/s
    ):
        """
        计算子任务从无人机 U_n 卸载到低轨卫星 S_l 上的传输时间 T_{n,l}^T
        自动调用链路速率计算，所有参数单位已标准化
        """
        # 距离标准化：千米 -> 米
        if d_nl is None:
            d_nl = np.random.uniform(800, 1250)  # 800~1250 km
        d_nl_m = d_nl * 1000  # 米

        # 数据量标准化：Kb -> bit
        if W_i is None:
            W_i = self.d_i_list[task_idx]  # 单位：Kb
        W_i_bit = W_i * 1024  # Kb -> bit

        mu_n_l_up = mu_n
        mu_n_l_down = mu_m

        # 频率、带宽标准化：GHz -> Hz
        f_hat_n_hz = f_hat_n * 1e9
        B_n_hz = B_n * 1e9
        # 自动计算上行链路速率（单位：Gbit/s）

        # 分贝线性化
        delta_l_linear = 10 ** (delta_l / 10)

        # 调用链路速率计算，确保 R_nl 单位为 bit/s   R_nl也是偏小
        _, _, _, R_nl = self.calculate_sat_uav_channel_and_rate(
            delta_l=delta_l,  # 线性化
            wavelength=wavelength,
            d_nl=d_nl,
            phi_angle=phi_angle,
            f_hat_n=f_hat_n,
            T_nl=T_nl,
            B_n=B_n,
            P_U=P_U,
            g_nl=g_nl,
            # interference_h_list=interference_h_list,
            sigma_l_sq=sigma_l_sq
        )

        propagation_delay = d_nl_m / c  # 单位：秒
        mu_n_l_up = mu_n
        mu_n_l_down = mu_m
        data_transmission_time = (b * (1 - mu_n_l_up) * (1 - mu_n_l_down) * W_i_bit) / R_nl  # 秒

        total_time = propagation_delay + data_transmission_time
        return total_time

        #### 代码在转卫星的逻辑还没有实现  ######!!!!!也就是卸载因子还没有给出一个具体值

    def calculate_transmission_energy(
            self,
            mu_m, mu_n, mu_l,  # μ_m, μ_n, μ_l
            task_idx,  # 新增
            P_U=1.0,
            b=1.2,
            W_i=None,
            # 下面为R_nl所需参数
            delta_l=25,
            wavelength=0.01,
            d_nl=None,
            phi_angle=None,
            f_hat_n=0.1,
            T_nl=None,
            B_n=0.01,
            g_nl=None,
            # interference_h_list=None,
            sigma_l_sq=None
    ):
        """
        计算无人机 U_n 和近地轨道卫星 S_l 之间传输的能耗 E_{n,l,i}^T
        参数:
            P_U: 无人机发射功率，单位：W
            b: 传输开销因子，默认1.2
            mu_n_i: 无人机卸载比例 μ_n^i
            mu_m_i: 另一卸载比例 μ_m^i
            W_i: 子任务 v_i 的输入数据大小
            其余参数：用于链路速率计算
        返回:
            传输能耗 E_{n,l,i}^T，单位：焦耳（J）
        """
        if d_nl is None:
            d_nl = np.random.uniform(800, 1250)  # 800~1250 km
        d_nl_m = d_nl * 1000  # m

        if W_i is None:
            W_i = self.d_i_list[task_idx]  # Kb
        W_i_bit = W_i * 1024

        # 分贝线性化
        delta_l_linear = 10 ** (delta_l / 10)

        # 频率、带宽标准化：GHz -> Hz
        f_hat_n_hz = f_hat_n * 1e9
        B_n_hz = B_n * 1e9

        # 调用链路速率计算，确保所有参数为SI单位，，####### 在这一块 因为有一些带入的值已经是变大的，然后再带入进去变得会更大，这里也需要修改，
        _, _, _, R_nl = self.calculate_sat_uav_channel_and_rate(
            delta_l=delta_l,
            wavelength=wavelength,
            d_nl=d_nl,
            phi_angle=phi_angle,
            f_hat_n=f_hat_n,
            T_nl=T_nl,
            B_n=B_n,
            P_U=P_U,
            g_nl=g_nl,
            # interference_h_list=interference_h_list,
            sigma_l_sq=sigma_l_sq
        )

        numerator = b * (1 - mu_n) * (1 - mu_m) * W_i_bit
        energy = P_U * numerator / R_nl  # 焦耳（J）
        return energy

        #### 代码在转卫星的逻辑还没有实现  ######!!!!!也就是卸载因子还没有给出一个具体值

    def calculate_satellite_computation_time(self, mu_m, mu_n, mu_l, instance_name, task_idx):
        """
        计算子任务 v_i 在卫星 S_l 处的计算执行时间 T_{l,i}^C
        参数:
            mu_l_i: 卫星卸载比例 μ_l^i
            mu_n_i: 无人机卸载比例 μ_n^i
            mu_m_i: 其他卸载比例 μ_m^i
            instance_name: 实例名称，对应实例文件夹名
            task_idx: 子任务索引（int）
        返回:
            计算执行时间 T_{l,i}^C，单位：秒
        """

        # 1. 读取所有子任务的CPU周期数
        cpu_cycles_list = self.load_task_cpu_cycles(instance_name, self.Nij)
        task_number = len(cpu_cycles_list)
        F_i_G = cpu_cycles_list[task_idx]  # 当前子任务的CPU周期数，单位：Gcycles

        # 2. 转换为实际周期数
        F_i = F_i_G * 1e9  # cycles

        f_max_S = 1.0e9  # 1 GHz = 1e9 Hz
        f_l_i = f_max_S

        # 3. 计算执行时间
        numerator = mu_l * (1 - mu_n) * (1 - mu_m) * F_i  # cycles
        exec_time_sec = numerator / f_l_i  # 秒
        return exec_time_sec

        #### 代码在转卫星的逻辑还没有实现  ######!!!!!也就是卸载因子还没有给出一个具体值

    def calculate_satellite_computation_energy(self, mu_l_i, mu_n_i, mu_m_i, instance_name, task_idx, f_sat):
        """
        计算子任务 v_i 在卫星 S_l 处的计算能耗 E_{l,i}^C
        参数:
            mu_l_i: 卫星卸载比例 μ_l^i
            mu_n_i: 无人机卸载比例 μ_n^i
            mu_m_i: 其他卸载比例 μ_m^i
            instance_name: 实例名称，对应实例文件夹名
            task_idx: 子任务索引（int）
        返回:
            计算能耗 E_{l,i}^C
        """
        i_coeff = 1e-25  # 能耗系数
        # 读取所有子任务的CPU周期数（单位：Gcycles）
        cpu_cycles_list = self.load_task_cpu_cycles(instance_name, self.Nij)
        F_i_G = cpu_cycles_list[task_idx]  # 当前子任务的CPU周期数，单位：Gcycles

        # 转换为实际周期数
        F_i = F_i_G * 1e9  # cycles

        f_max_S = 1.0e9  # 1 GHz
        f_l_i = f_max_S

        return i_coeff * mu_l_i * (1 - mu_n_i) * (1 - mu_m_i) * F_i * (f_l_i ** 2)

        #### 代码在转卫星的逻辑还没有实现  ######!!!!!也就是卸载因子还没有给出一个具体值，然后这里的数据量是说是再0.6-1.2MB中均匀分布，
        # 可以先暂时这样设置，此处还没有设置

    def calculate_satellite_offloading_metrics(
            self,
            mu_i_l, mu_i_n, mu_i_m,
            W_i,  # bit
            instance_name,  # 实例名称
            task_idx,  # 子任务索引
            f_j_i=3,  # 设为3Ghz 设置把云服务器的资源全部分配出去
            P_s=1.0,  # W, 发射功率
            d_ij=None,  # km, 卫星间距离
            b=1.2,  # 传输开销因子
            G_max=20,  # #这里是改变速率的重要的点
            B_ij=1.0,  # GHz, 卫星间链路带宽
            zeta=354.81,  # K, 接收端系统等效噪声系数
            k=1.38e-23,  # 玻尔兹曼常数
            f_s=30.0,  # GHz
            c=3e8,  # m/s, 光速
            t_i=1e-25  # 能耗系数
    ):
        """
        计算卫星间链路的传输速率、传输时间、传输能耗、计算执行时间和计算能耗（多卫星模型）
        返回:
            R_ij: 传输速率 (Gbit/s)
            T_j_i_T: 传输时间 (s)
            E_j_i_T: 传输能耗 (J)
            T_j_i_C: 计算执行时间 (s)
            E_j_i_C: 计算能耗 (J)
        """
        # 动态生成卫星间距离
        if d_ij is None or d_ij == 0:
            d_ij = np.random.uniform(100, 500)  # km
        d_ij_m = d_ij * 1000  # m

        if W_i is None:
            W_i = self.d_i_list[task_idx]  # Kb
        W_i_bit = W_i * 1024

        # 单位换算
        B_ij_Hz = B_ij * 1e9  # GHz -> Hz
        f_s_Hz = f_s * 1e9  # GHz -> Hz

        # 读取该任务的CPU周期数
        project_root = os.path.dirname(os.path.abspath(__file__))
        cpu_cycles_path = os.path.join(project_root, 'instance', instance_name, 'task_CPU_cycles_number.csv')
        cpu_cycles_df = pd.read_csv(cpu_cycles_path)
        task_number = len(cpu_cycles_df)
        cpu_cycles_list = [float(cpu_cycles_df.values[i][0]) for i in range(task_number)]
        F_i_G = cpu_cycles_list[task_idx]
        F_i = F_i_G * 1e9  # cycles

        # 计算传输速率 R_ij
        numerator = P_s * (G_max ** 2)
        denominator = zeta * k * B_ij_Hz * ((4 * math.pi * d_ij_m * f_s_Hz / c) ** 2)  # d_ij km -> m
        # R_ij太小导致延时不对 能耗都不对 不过现在归一化令G_max为1 这样算出来就差不多是一万多比特，周四弄一下最后一个R_lk 最后运行看一下结果，如果可以的话就找其他指标试着作比较
        R_ij = B_ij_Hz * np.log2(1 + (numerator / denominator))

        # 计算传输时间 T_j_i_T
        propagation_delay = d_ij_m / c  # m / (m/s) = s
        data_transmission_time = (b * (1 - mu_i_l) * (1 - mu_i_n) * (1 - mu_i_m) * W_i_bit) / R_ij  # bit / (bit/s) = s
        T_j_i_T = propagation_delay + data_transmission_time

        # 计算传输能耗 E_j_i_T
        E_j_i_T = P_s * data_transmission_time

        # 计算计算执行时间 T_j_i_C
        numerator_C = (1 - mu_i_l) * (1 - mu_i_n) * (1 - mu_i_m) * F_i
        T_j_i_C = numerator_C / (f_j_i * 1e9)  # F_i: cycles, f_j_i: GHz -> cycles/s

        # 计算计算能耗 E_j_i_C
        E_j_i_C = t_i * numerator_C * (f_j_i * 1e9) ** 2  # cycles * (Hz)^2

        return R_ij, T_j_i_T, E_j_i_T, T_j_i_C, E_j_i_C

        ### 信道增益为未知数 ####

    def calculate_cloud_server_metrics(
            self,
            mu_l_i, mu_n_i, mu_m_i,
            instance_name,  # 实例名称
            task_idx,  # 子任务索引
            f_k_i,  # GHz, 云服务器分配给任务的计算资源 (<=3)
            d_lk=None,  # km, 卫星到云服务器距离
            b=1.2,  # 传输开销因子
            P_s=1.0,  # W, 发射功率
            B_lk=1.0,  # GHz, 卫星与云服务器间带宽
            h_lk=None,  # 信道增益复数（复数模长的平方）
            W_i=None,
            sigma_l=None,  # 噪声功率（如为None则自动计算，单位W）
            t_i=1e-25  # 能耗系数
    ):
        """
        计算云服务器卸载任务的传输速率、传输时间、传输能耗、计算执行时间和计算能耗。
        参数:
            mu_l_i, mu_n_i, mu_m_i: 任务卸载比例相关参数 (0~1)
            instance_name: 实例名称
            task_idx: 子任务索引
            f_k_i: 云服务器分配给任务的计算资源 (GHz, <=3)
            d_lk: 卫星 S_l 到云服务器 CS_k 的距离 (km, 50~100)
            b: 传输开销因子 (默认1.2)
            P_s: 发射功率 (W, 默认1)
            B_lk: 带宽 (GHz, 默认1)
            h_lk: 信道增益复数（如未知则随机生成Rayleigh信道）
            sigma_l: 噪声功率 (dBm, 默认-100)
            t_i: 能耗系数 (默认1e-25)
        返回:
            R_lk: 传输速率 (Gbit/s)
            T_k_i_T: 传输时间 (s)
            E_l_k_i_T: 传输能耗 (J)
            T_k_i_C: 计算执行时间 (s)
            E_k_i_C: 计算能耗 (J)
        """
        # 距离动态生成
        if d_lk is None:
            d_lk = np.random.uniform(50, 100)  # km
        d_lk_m = d_lk * 1000  # 转换为米

        # 数据量标准化：如为None则读取，单位为Kb，需转bit
        if W_i is None:
            W_i = self.d_i_list[task_idx]  # Kb
        W_i_bit = W_i * 1024  # Kb -> bit

        # 带宽单位转换 GHz -> Hz
        B_lk_Hz = B_lk * 1e9

        # 信道增益处理（如未知则Rayleigh分布）因为在论文中暂未找到出处 所以先暂时这样计算。
        if h_lk is None:
            # 云服务器到地面用户中点的距离（km）
            d_cloud_ground = np.random.uniform(50, 100)
            # 地面用户到无人机距离（km）
            d_ground_uav = 0.03
            # 无人机到卫星距离（km）
            d_uav_sat = np.random.uniform(800, 1250)
            # 近似云服务器到卫星的距离（km），假设三点共线
            d_lk = d_cloud_ground + d_ground_uav + d_uav_sat
            d_lk_m = d_lk * 1e3  # 转为米

            # 自由空间损耗模型
            c = 3e8  # 光速 m/s
            f_c = 30e9  # 载波频率，30GHz（Ka频段）
            # 增大天线增益
            G_t = 17  # 发射天线增益（线性值，建议2~5）
            G_r = 17  # 接收天线增益（线性值，建议2~5）
            channel_gain_factor = 1  # 经验放大系数，建议1~5

            # 计算信道增益
            L_lk = (c / (4 * np.pi * d_lk_m * f_c)) ** 2
            h_lk = np.sqrt(L_lk) * G_t * G_r * channel_gain_factor  # 复数信道增益，假设无相位

        h_abs2 = h_lk ** 2

        if sigma_l is None:
            k_B = 1.38e-23  # 玻尔兹曼常数，J/K
            zeta = 354.81  # 噪声温度，K
            B_lk_Hz = B_lk * 1e9  # 将GHz转换为Hz
            sigma_l = k_B * zeta * B_lk_Hz  # 单位：W (J/s)

        # 读取该任务的CPU周期数
        # 先获取任务总数
        project_root = os.path.dirname(os.path.abspath(__file__))
        cpu_cycles_path = os.path.join(project_root, 'instance', instance_name, 'task_CPU_cycles_number.csv')
        cpu_cycles_df = pd.read_csv(cpu_cycles_path)
        F_i_G = float(cpu_cycles_df.values[task_idx][0])
        F_i = F_i_G * 1e9  # cycles

        # 计算信噪比 SNR
        snr = (P_s * h_abs2) / sigma_l

        # 传输速率 R_lk (Gbit/s)
        R_lk = B_lk_Hz * np.log2(1 + snr)  # 单位：Gbit/s

        # 传输时间 T_k_i_T (s)
        c = 3e8  # 光速 m/s
        propagation_delay = d_lk_m / c  # s
        data_size = b * (1 - mu_l_i) * (1 - mu_n_i) * (1 - mu_m_i) * W_i_bit  # bit
        T_k_i_T = propagation_delay + data_size / R_lk  # s

        # 传输能耗 E_l_k_i_T
        E_l_k_i_T = P_s * data_size / R_lk  # J

        # 计算执行时间 T_k_i_C
        cpu_cycles = (1 - mu_l_i) * (1 - mu_n_i) * (1 - mu_m_i) * F_i
        T_k_i_C = cpu_cycles / (f_k_i * 1e9)  # s, f_k_i为GHz

        # 计算能耗 E_k_i_C (J)
        E_k_i_C = t_i * cpu_cycles * (f_k_i * 1e9) ** 2  # cycles * (Hz)^2

        return R_lk, T_k_i_T, E_l_k_i_T, T_k_i_C, E_k_i_C

    def calculate_total_system_energy(self, ind, instance_name, task_indices, index, current_time=0):
        """
        多卫星适配：通过参数传递支持多卫星，兼容单卫星/多卫星场景
        """
        total_energy = 0

        # 1. 获取当前可用卫星
        available_sats = self.get_available_satellites(current_time)
        if not available_sats:
            return float("inf")  # 无可用卫星时返回无穷大

        # 2. 选择主卫星
        primary_sat = max(available_sats, key=lambda x: x["effective_time"])
        primary_distance = primary_sat["distance"]
        primary_f_sat = primary_sat["satellite"]["f_sat"]

        # 3. 地面设备能耗
        total_energy += self.calculate_individual_energy(ind, index)

        # 4. 无人机到卫星的传输能耗  这里有问题导致计算出来的能耗巨大无比
        for idx in task_indices:
            mu_m = ind.offloading_factors['mu_m']
            mu_n = ind.offloading_factors['mu_n']
            mu_l = ind.offloading_factors['mu_l']
            W_i = self.d_i_list[idx]
            E_nl_T = self.calculate_transmission_energy(
                mu_m, mu_n, mu_l, idx,
                W_i=W_i,
                d_nl=primary_distance
            )
            total_energy += E_nl_T

        # 5. 卫星计算能耗
        for idx in task_indices:
            mu_m = ind.offloading_factors['mu_m']
            mu_n = ind.offloading_factors['mu_n']
            mu_l = ind.offloading_factors['mu_l']
            E_l_C = self.calculate_satellite_computation_energy(
                mu_l, mu_n, mu_m, instance_name, idx,
                f_sat=primary_f_sat
            )
            total_energy += E_l_C

        # 6. 星间链路能耗（多卫星时才计算）
        if len(available_sats) >= 2:
            secondary_sat = available_sats[1]
            secondary_f_sat = secondary_sat["satellite"]["f_sat"]
            for idx in task_indices:
                mu_l_i = ind.offloading_factors['mu_l']
                mu_n_i = ind.offloading_factors['mu_n']
                mu_m_i = ind.offloading_factors['mu_m']
                W_i = self.d_i_list[idx]
                _, _, E_l_k_i_T, _, E_k_i_C = self.calculate_cloud_server_metrics(
                    mu_l_i, mu_n_i, mu_m_i, instance_name, idx,
                    f_k_i=1.0,
                    d_lk=primary_distance * 0.2  # 可根据实际场景调整
                )
                total_energy += E_l_k_i_T + E_k_i_C

        return total_energy

    def calculate_task_delay(self, ind, task_idx, current_time=0):
        """
        计算单个任务的最大延时，适配多卫星参数传递
        """
        uav = ind.chromosome['S'][task_idx]
        workflow = uav.workflow

        # 1. 地面到无人机的传输时间
        total_T_mn = 0.0
        for task_id in workflow.order:
            d_mn = 0.03
            _, T_mn = self.calculate_ground_to_uav_transmission_energy(d_mn, task_id)
            total_T_mn += T_mn

        # 2. 本地直接计算延时
        md_frequency = 0.6  # GHz
        cpu_cycles_list = self.load_task_cpu_cycles(self.instance_name, self.Nij)
        local_exec_time = self.calculate_all_local_tasks_total_time(
            workflow, cpu_cycles_list, md_frequency=md_frequency
        )

        # 3. 地面→无人机→无人机执行延时
        ground_delay = self.calculate_ground_execution_delay(ind, uav, workflow)

        # 4. 卫星相关延时
        available_sats = self.get_available_satellites(current_time)
        if not available_sats:
            sat_delay = float("inf")
            sat_isl_delay = float("inf")
            cloud_delay = float("inf")
        else:
            primary_sat = max(available_sats, key=lambda x: x["effective_time"])
            primary_f_sat = primary_sat["satellite"]["f_sat"]

            # 地面->无人机->卫星延时
            total_T_nl_T = 0.0
            total_T_l_i_C = 0.0
            mu_m = ind.offloading_factors['mu_m']
            mu_n = ind.offloading_factors['mu_n']
            mu_l = ind.offloading_factors['mu_l']
            for task_id in workflow.order:
                T_nl_T = self.calculate_offloading_transmission_time(mu_m, mu_n, mu_l, task_id)
                T_l_i_C = self.calculate_satellite_computation_time(
                    mu_m, mu_n, mu_l, self.instance_name, task_id
                )
                total_T_nl_T += T_nl_T
                total_T_l_i_C += T_l_i_C
            sat_delay = total_T_mn + total_T_nl_T + total_T_l_i_C

            # 地面->无人机->卫星i->星间链路->卫星j延时
            total_T_j_i_T = 0.0
            total_T_j_i_C = 0.0
            if len(available_sats) >= 2:
                secondary_sat = available_sats[1]
                secondary_f_sat = secondary_sat["satellite"]["f_sat"]
                f_j_i = secondary_f_sat
                for task_id in workflow.order:
                    _, T_j_i_T, _, T_j_i_C, _ = self.calculate_satellite_offloading_metrics(
                        mu_l, mu_n, mu_m, None, self.instance_name, task_id, f_j_i
                    )
                    total_T_j_i_T += T_j_i_T
                    total_T_j_i_C += T_j_i_C
                sat_isl_delay = total_T_mn + total_T_nl_T + total_T_j_i_T + total_T_j_i_C
            else:
                sat_isl_delay = float("inf")

            # 地面->无人机->卫星->云服务器延时
            f_k_i = 3.0
            total_T_k_i_T = 0.0
            total_T_k_i_C = 0.0
            for task_id in workflow.order:
                _, T_k_i_T, _, T_k_i_C, _ = self.calculate_cloud_server_metrics(
                    mu_l, mu_n, mu_m, self.instance_name, task_id, f_k_i
                )
                total_T_k_i_T += T_k_i_T
                total_T_k_i_C += T_k_i_C
            cloud_delay = total_T_mn + total_T_nl_T + total_T_k_i_T + total_T_k_i_C

        # 取最大延时
        return max(local_exec_time, ground_delay, sat_delay, sat_isl_delay, cloud_delay)

        # 计算个体 ind 的适应度，包括时间和能量消耗

    def calculateFitness(self, ind):

        # 以卸载因子作为参数传递
        mu_m = ind.offloading_factors['mu_m']
        mu_n = ind.offloading_factors['mu_n']
        mu_l = ind.offloading_factors['mu_l']

        # 确保个体的服务缓存决策是可行的。如果缓存决策不符合约束条件，该方法会进行调整。
        self.repair_infeasible_caching_decision(ind)
        ind.fitness = []
        time = []
        energy = []
        cost = []

        for index in range(N_n):
            # 对于每个子任务索引 index 在范围 N_n 表示无人机数量内：
            # 从个体的染色体中获取对应的无人机 uav，即 ind.chromosome['S'][index]。
            # 调用 self.calculateWorkflowTimeEnergy(ind, uav, uav.workflow) 方法，计算该无人机在其工作流中的总执行时间和能量消耗，并更新相关属性。
            # 将无人机的总执行时间 uav.workflow.schedule.T_total 添加到 time 列表中。
            # 将无人机的总能量消耗 uav.workflow.schedule.E_total 添加到 energy 列表中。

            # 对于每个子任务索引 index 在范围 N_n（无人机数量）内：
            # 以多跳卸载的最大任务时延为目标
            task_delay = self.calculate_task_delay(ind, index)
            time.append(task_delay)

            uav = ind.chromosome['S'][index]  # 一个gene就是一个UAV
            # 先调度，赋值所有任务的调度属性
            self.calculateWorkflowTimeEnergy(ind, uav, uav.workflow)

            task_indices = list(range(self.Nij))
            total_energy = self.calculate_total_system_energy(ind, self.instance_name, task_indices, index)
            energy.append(total_energy)

            f_xi = uav.coreCC[uav.workflow.location[index]]
            T_si = uav.workflow.schedule.T_total
            alpha = 1  # 设置 alpha 的值
            beta = 2.5e-12  # 设置 beta 的值
            cpu_cycles = T_si * f_xi * 1e9  # 总CPU周期数
            total_cost = alpha * beta * cpu_cycles  # 美元
            # unit_cost = np.exp(-alpha) * (np.exp(f_xi) - 1) * beta
            # total_cost = unit_cost * T_si
            cost.append(total_cost)

        # 计算并将平均时间和平均能量分别添加到个体的适应度列表 ind.fitness 中。
        ind.fitness.append(np.average(time))
        ind.fitness.append(np.average(energy))
        ind.fitness.append(np.average(cost))

        # 修复个体 ind 的服务缓存决策，使其符合约束条件

    def repair_infeasible_caching_decision(self, ind):
        # 无限循环
        while True:
            # 初始化变量 s 为 0，用于累加缓存决策的总开销。
            s = 0
            all_1_set = []  # 保存决策为1的索引和程序规模
            for j in range(N_service):
                # 算当前服务的开销 s，通过将服务的缓存决策 ind.chromosome['B'][j] 与其对应的规模 h_s_list[j] 相乘并累加到 s 中。
                # 如果该服务的缓存决策为 1（即被缓存），则将其索引和规模作为元组 (j, h_s_list[j]) 添加到 all_1_set 列表中
                s += ind.chromosome['B'][j] * h_s_list[j]
                if ind.chromosome['B'][j] == 1:
                    all_1_set.append((j, h_s_list[j]))
            if s <= h_hap:  # 检查总开销 s 是否小于等于最大允许开销 h_hap。如果满足条件，则退出循环
                # 如果总开销 s 超过了 h_hap，则对 all_1_set 列表进行排序，按规模从大到小排列。
                break
            all_1_set = sorted(all_1_set, key=itemgetter(1), reverse=True)
            # 将 all_1_set 中第一个元素（即规模最大的服务）的缓存决策设置为 0，表示取消该服务的缓存。
            ind.chromosome['B'][all_1_set[0][0]] = 0

    def calculateWorkflowTimeEnergy(self, ind, uav, workflow):
        # 初始化工作流调度的时间和能量列表
        workflow.schedule.TimeEnergy = []
        workflow.schedule.T_total = None  # 总时间初始化为 None
        workflow.schedule.E_total = 0  # 总能量初始化为 0
        workflow.schedule.C_total = 0  # 总成本初始化为 0

        # 遍历工作流中的每个任务
        for i in range(len(workflow.order)):
            taskId = workflow.order[i]  # 获取当前任务的 ID
            pos = workflow.location[i]  # 获取当前任务的执行位置
            task = workflow.taskSet[taskId]  # 获取当前任务对象
            task.exePosition = pos  # 设置任务的执行位置

            if pos == self.K + 1:  # 如果任务在 HAP 上执行
                task.islocal = False  # 标记任务为非本地执行
                if task.id == workflow.entryTask:  # 如果是入口任务
                    # 初始化准备时间、开始时间和完成时间
                    task.RT_i_l = task.ST_i_l = task.FT_i_l = 0
                    task.RT_i_ws = task.ST_i_ws = 0.0
                    task.FT_i_ws = task.ST_i_ws + task.I_i_j / uav.R_i  # 计算完成时间
                    task.RT_i_c = task.ST_i_c = task.FT_i_ws  # 更新云服务器的准备时间

                    # 判断 HAP 是否缓存了任务请求的服务
                    if ind.chromosome['B'][task.S_i_j] == 1:
                        task.FT_i_c = task.ST_i_c + task.M_i_j / (uav.ratio * uav.coreCC[pos])
                    else:
                        task.FT_i_c = task.ST_i_c + task.M_i_j / (uav.ratio * uav.coreCC[pos]) + D_ij_dow

                    task.RT_i_wr = task.ST_i_wr = task.FT_i_c  # 更新无线接收通道的准备时间
                    task.FT_i_wr = task.ST_i_wr + task.O_i_j / uav.R_i  # 计算完成时间
                    # 更新调度时间点
                    workflow.schedule.wsTP.append(task.FT_i_ws)
                    workflow.schedule.MECTP.append(task.FT_i_c)
                    workflow.schedule.wrTP.append(task.FT_i_wr)
                else:
                    # 处理非入口任务
                    task.RT_i_ws = self.get_RT_i_ws(task, workflow)  # 获取无线发送通道的准备时间
                    task.ST_i_l = float("inf")  # 本地开始时间初始化为无穷大
                    task.FT_i_l = float("inf")  # 本地完成时间初始化为无穷大

                    # 计算无线发送通道的开始和完成时间
                    if workflow.schedule.wsTP[-1] < task.RT_i_ws:
                        task.ST_i_ws = task.RT_i_ws
                        task.FT_i_ws = task.ST_i_ws + task.I_i_j / uav.R_i
                    else:
                        task.ST_i_ws = workflow.schedule.wsTP[-1]
                        task.FT_i_ws = task.ST_i_ws + task.I_i_j / uav.R_i
                    workflow.schedule.wsTP.append(task.FT_i_ws)

                    # 计算云服务器的开始和完成时间
                    task.RT_i_c = self.get_RT_i_c(task, workflow)
                    if workflow.schedule.MECTP[-1] < task.RT_i_c:
                        task.ST_i_c = task.RT_i_c
                        if ind.chromosome['B'][task.S_i_j] == 1:
                            task.FT_i_c = task.ST_i_c + task.M_i_j / (uav.ratio * uav.coreCC[pos])
                        else:
                            task.FT_i_c = task.ST_i_c + task.M_i_j / (uav.ratio * uav.coreCC[pos]) + D_ij_dow
                    else:
                        task.ST_i_c = workflow.schedule.MECTP[-1]
                        if ind.chromosome['B'][task.S_i_j] == 1:
                            task.FT_i_c = task.ST_i_c + task.M_i_j / (uav.ratio * uav.coreCC[pos])
                        else:
                            task.FT_i_c = task.ST_i_c + task.M_i_j / (uav.ratio * uav.coreCC[pos]) + D_ij_dow

                    workflow.schedule.MECTP.append(task.FT_i_c)

                    # 计算无线接收通道的开始和完成时间
                    task.RT_i_wr = task.FT_i_c
                    if workflow.schedule.wrTP[-1] < task.RT_i_wr:
                        task.ST_i_wr = task.RT_i_wr
                        task.FT_i_wr = task.ST_i_wr + task.O_i_j / uav.R_i
                    else:
                        task.ST_i_wr = workflow.schedule.wrTP[-1]
                        task.FT_i_wr = task.ST_i_wr + task.O_i_j / uav.R_i
                    workflow.schedule.wrTP.append(task.FT_i_wr)

                    # 计算任务的能量消耗
                    task.energy += uav.pws_i * (task.FT_i_ws - task.ST_i_ws)
                    task.energy += uav.pwr_i * (task.FT_i_wr - task.ST_i_wr)
                    workflow.schedule.E_total += task.energy  # 更新总能量

            else:  # 如果任务在本地核心上执行
                task.islocal = True  # 标记任务为本地执行
                task.RT_i_ws = task.RT_i_c = task.RT_i_wr = 0.0  # 初始化准备时间
                task.ST_i_ws = task.ST_i_c = task.ST_i_wr = 0.0  # 初始化开始时间
                task.FT_i_ws = task.FT_i_c = task.FT_i_wr = 0.0  # 初始化完成时间

                if task.id == workflow.entryTask:  # 如果是入口任务
                    task.RT_i_l = task.ST_i_l = 0  # 初始化准备和开始时间
                    task.FT_i_l = task.ST_i_l + task.M_i_j / uav.coreCC[pos]  # 计算完成时间
                else:
                    task.RT_i_l = self.get_RT_i_l(task, workflow)  # 获取本地准备时间
                    if task.RT_i_l > workflow.schedule.coreTP[pos][-1]:
                        task.ST_i_l = task.RT_i_l  # 更新开始时间
                    else:
                        task.ST_i_l = workflow.schedule.coreTP[pos][-1]  # 使用上一个任务的完成时间
                    task.FT_i_l = task.ST_i_l + task.M_i_j / uav.coreCC[pos]  # 计算完成时间
                workflow.schedule.coreTP[pos].append(task.FT_i_l)  # 更新核心时间点
                task.energy = uav.pcc_i[pos] * (task.FT_i_l - task.ST_i_l)  # 计算能量消耗
                workflow.schedule.E_total += task.energy  # 更新总能量

            workflow.schedule.S[pos].append(task.id)  # 将任务 ID 添加到执行位置的任务集合中

        # 根据出口任务的执行情况更新总时间
        if workflow.taskSet[workflow.exitTask].islocal:
            workflow.schedule.T_total = workflow.taskSet[workflow.exitTask].FT_i_l
        else:
            workflow.schedule.T_total = workflow.taskSet[workflow.exitTask].FT_i_wr

        # 将总时间和总能量添加到时间能量列表中
        workflow.schedule.TimeEnergy.append(workflow.schedule.T_total)
        workflow.schedule.TimeEnergy.append(workflow.schedule.E_total)

    def calculateComputingCost(self, f, T_e_i):  # m为卸载的服务器索引, T_e_i为在该服务器上的计算时间
        unit_cost = np.exp(-epsilon_1) * (np.exp(f) - 1) * epsilon_2
        return unit_cost * T_e_i


    def get_RT_i_ws(self, task, workflow):
        if task.id == workflow.entryTask:
            return 0.0
        else:
            pre_max = []
            for pre_taskId in task.preTaskSet:
                if workflow.taskSet[pre_taskId].islocal == True:
                    pre_max.append(workflow.taskSet[pre_taskId].FT_i_l)
                else:
                    pre_max.append(workflow.taskSet[pre_taskId].FT_i_ws)
            return max(pre_max)


    def get_RT_i_c(self, task, workflow):
        pre_max = []
        for pre_taskId in task.preTaskSet:
            pre_max.append(workflow.taskSet[pre_taskId].FT_i_c)
        return max(task.FT_i_ws, max(pre_max))


    def get_RT_i_l(self, task, workflow):
        if task.id == workflow.entryTask:
            return 0.0
        else:
            pre_max = []
            for pre_taskId in task.preTaskSet:
                if workflow.taskSet[pre_taskId].islocal == True:
                    pre_max.append(workflow.taskSet[pre_taskId].FT_i_l)
                else:
                    pre_max.append(workflow.taskSet[pre_taskId].FT_i_wr)
            return max(pre_max)


    def reInitialize_WorkflowTaskSet_Schedule(self, smd):
        for task in smd.workflow.taskSet:
            self.reInitializeTaskSet(task)
        self.reInitializeSchedule(smd.workflow.schedule)


    def reInitializeTaskSet(self, task):
        task.islocal = None
        task.exePosition = None
        task.RT_i_l = task.ST_i_l = task.FT_i_l = None
        task.RT_i_ws = task.RT_i_c = task.RT_i_wr = None
        task.ST_i_ws = task.ST_i_c = task.ST_i_wr = None
        task.FT_i_ws = task.FT_i_c = task.FT_i_wr = None
        task.energy = 0
        task.cost = 0


    def reInitializeSchedule(self, schedule):
        if schedule.K == 3:
            schedule.S = {1:[], 2:[], 3:[], 4:[]}
            schedule.coreTP = {1:[0], 2:[0], 3:[0]}
        else:
            schedule.S = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
            schedule.coreTP = {1: [0], 2: [0], 3: [0], 4: [0], 5: [0], 6: [0]}
        schedule.wsTP = [0]
        schedule.MECTP = [0]
        schedule.wrTP = [0]
        schedule.T_total = None
        schedule.E_total = 0
        schedule.TimeEnergy = []


    def get_UAV_transmission_rate(self):
        UAV_transmission_rate_set = []
        for i in range(N_uav):
            d_i_UH = self.getDistance(uav_coordinate_set[i], [100, 100, z_H])
            g_i = pow(10, g_0 / 10) / d_i_UH
            R_i = (Psi * pow(10, 6)) * np.log2(1 + (p_i_upl * g_i)/sigma_2)
            UAV_transmission_rate_set.append(R_i)
        return UAV_transmission_rate_set


    def get_workflow_set(self, filename):
        for i in range(1, N_uav+1):
            wf = Workflow(self.K)
            with open(filename+'/DAG/'+str(i)+'.txt', 'r') as readFile:
                for line in readFile:
                    task = Task()
                    task.I_i_j = round(random.uniform(5000, 6000), 2) * 1024
                    task.O_i_j = round(random.uniform(500, 1000), 2) * 1024
                    task.M_i_j = round(random.uniform(0.1, 0.5), 2)
                    task.S_i_j = round(random.randint(0, 9))

                    s = line.splitlines()
                    s = s[0].split(':')
                    predecessor = s[0]
                    id = s[1]
                    successor = s[2]
                    if (predecessor != ''):
                        predecessor = predecessor.split(',')
                        for pt in predecessor:
                            task.preTaskSet.append(int(pt))
                    else:
                        wf.entryTask = int(id)
                    task.id = int(id)
                    if (successor != ''):
                        successor = successor.split(',')
                        for st in successor:
                            task.sucTaskSet.append(int(st))
                    else:
                        wf.exitTask = int(id)
                    wf.taskSet.append(task)
            self.workflow_set.append(wf)


    def getIGDValue(self, PF_ref, PF_know):
        sum = []
        for v in PF_ref:
            distance = self.d_v_PFSet(v, PF_know)
            sum.append(distance)
        return np.average(sum)


    def d_v_PFSet(self, v, PFSet):  # 求v和PFSet中最近的距离
        dList = []
        for pf in PFSet:
            distance = self.getDistance(v, pf)
            dList.append(distance)
        return min(dList)


    def getDistance(self, point1, point2):
        return np.sqrt(np.sum(np.square([point1[i] - point2[i] for i in range(len(point1))])))

class Individual:
    def __init__(self):
        self.chromosome = {}      #基因位是UAV类型
        self.fitness = []
        self.isFeasible = True    #判断该个体是否合法
        self.temp_fitness = None  #临时适应度，计算拥挤距离的时候，按每个目标值来对类列表进行升序排序
        self.distance = 0.0
        self.rank = None
        self.S_p = []  #种群中此个体支配的个体集合
        self.n = 0  #种群中支配此个体的个数
        mu_m = random.uniform(0, 1)
        mu_n = random.uniform(0, 1 - mu_m)
        mu_l = 1 - mu_m - mu_n
        self.offloading_factors = {
            'mu_m': 0.33,
            'mu_n': 0.33,
            'mu_l': 0.33
        }

class UAV:
    def __init__(self, K):
        self.coordinate = []    # 无人机的位置坐标
        self.workflow = Workflow(K)      # 无人机对应的工作流对象
        self.ratio = None       # 资源分配比例（如CPU频率分配比例）
        self.channel = None     # 信道索引（用于通信）
        self.R_i = None         # 无人机的数据传输速率
        self.K = K              # 无人机核心数量

        # 根据核心数量K初始化核心的计算能力（coreCC），单位可为GHz或相对值
        # K=3时，4个核心，最后一个为HAP（高空平台）核心
        if K == 3:
            self.coreCC = {1:1, 2:0.8, 3:0.6, 4:f_sat}        # 三个本地核心和一个sat核心的计算能力
        else:
            # K>3时，7个核心，最后一个为sat核心
            self.coreCC = {1:1.6, 2:1.4, 3:1.2, 4:1, 5:0.8, 6:0.6, 7:f_sat}

        # 根据核心数量K初始化各核心最大频率下的功耗（pcc_i），单位为瓦特
        if K == 3:  # 三核心时的功耗
            self.pcc_i = {1:4, 2:2, 3:1}
        else:       # 多核心时的功耗
            self.pcc_i = {1:10, 2:8, 3:6, 4:4, 5:2, 6:1}

        self.pws_i = p_i_upl  # 无人机发送数据时的功耗（单位：瓦特）
        self.pwr_i = p_i_rec  # 无人机接收数据时的功耗（单位：瓦特）

class Workflow:
    def __init__(self, K):
        self.entryTask = None      #开始任务
        self.exitTask = None       #结束任务
        self.order = []         #执行顺序
        self.location = []         #执行位置
        self.taskNumber = None
        self.taskSet = []          #列表的索引值就是任务的id值
        self.schedule = Schedule(K)

class Schedule:
    def __init__(self, K):
        self.K = K
        self.taskSet = {}
        if K == 3:
            self.S = {1:[], 2:[], 3:[], 4:[]} # Record the set of task that is executed certain execution unit selection. eg. S[3]=[v1,v3,v5,v7,v9,v10]
            self.coreTP = {1:[0], 2:[0], 3:[0]}  # Index is core number, its element denotes the current time point on the core.
        else:
            self.S = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
            self.coreTP = {1: [0], 2: [0], 3: [0], 4: [0], 5: [0], 6: [0]}
        self.wsTP = [0]  # The current time point on the wireless sending channel.
        self.MECTP = [0]  # The current time point on the cloud.
        self.wrTP = [0]  # The current time point on the wireless receiving channel.
        self.T_total = None
        self.E_total = 0
        self.C_total = 0
        self.TimeEnergy = []

class Task:
    def __init__(self):
        self.id = None
        self.islocal = None    # Denote the task is executed locally or on cloud.
        self.preTaskSet = []   #The set of predecessor task (element is Task class).
        self.sucTaskSet = []   #The set of successor task (element is Task class).
        self.exePosition = None  # it denotes execution position (i.e., [1,2,3,4])of the task.
        self.actualFre = 1    # The actual frequency scaling factors.

        self.I_i_j = None  # The data size of the task.
        self.O_i_j = None  # The output data size of the task.
        self.M_i_j = None  # The number of CPU cycles required to perform task
        self.S_i_j = None  # The number of service programs supported by the mEC system

        self.RT_i_l = None     # The ready time of task vi on a local core.
        self.RT_i_ws = None    # The ready time of task vi on the wireless sending channel.
        self.RT_i_c = None     # The ready time of task vi on the [10,20] server.
        self.RT_i_wr = None    # The ready time for the cloud to transmit back the results of task vi

        self.ST_i_l = None     # The start time of task vi on a local core.
        self.ST_i_ws = None    # The start time of task vi on the wireless sending channel.
        self.ST_i_c = None     # The start time of task vi on the [10,20] server.
        self.ST_i_wr = None    # The start time for the cloud to transmit back the results of task vi

        self.FT_i_l = None     # The finish time of task vj on a local core.
        self.FT_i_ws = None    # The finish time of task vj on the wireless sending channel.
        self.FT_i_c = None     # The finish time of task vj on the [10,20] server.
        self.FT_i_wr = None    # The finish time of task vj on the wireless receiving channel.
        self.energy = 0
        self.cost = 0

def get_combined_pf_ref(project_path, instance_name):
    """
    合并MOEAD和NSGA2第一轮的PF前沿，作为参考帕累托前沿
    """
    import os
    import pandas as pd

    moead_pf_path = os.path.join(
        project_path,
        'ExperimentResult',
        instance_name,
        'MOEAD_1',
        '1',
        'PF.csv'
    )
    nsgaii_pf_path = os.path.join(
        project_path,
        'ExperimentResult',
        instance_name,
        'NSGA2',
        '1',
        'PF.csv'
    )
    PF_ref = []
    # 读取MOEAD PF
    if os.path.exists(moead_pf_path):
        moead_pf_df = pd.read_csv(moead_pf_path)
        moead_pf = moead_pf_df[['Time', 'Energy', 'Cost']].values.tolist()
        PF_ref.extend(moead_pf)
    else:
        print(f"Warning: MOEAD PF.csv not found at {moead_pf_path}.")
    # 读取NSGA2 PF
    if os.path.exists(nsgaii_pf_path):
        nsgaii_pf_df = pd.read_csv(nsgaii_pf_path)
        nsgaii_pf = nsgaii_pf_df[['Time', 'Energy', 'cost']].values.tolist()
        PF_ref.extend(nsgaii_pf)
    else:
        print(f"Warning: NSGA2 PF.csv not found at {nsgaii_pf_path}.")

    # 去重（可选，若PF前沿可能有重复点）
    PF_ref = [list(x) for x in set(tuple(row) for row in PF_ref)]
    return PF_ref


def MOEAD_1_run(instance_name, args):
    alg_dir = os.path.join(project_path, 'ExperimentResult', instance_name, 'MOEAD_1_multisat')
    each_run_ExeTime_list = []
    all_offloading_factors = []

    for I in range(1, args.runTime + 1):
        if not os.path.isdir(alg_dir + str(I)):
            os.makedirs(alg_dir + str(I))

        startTime = time.time()
        moead = MOEAD_1(instance_name, args)
        # 新增：设置参考帕累托前沿  这里的前沿只有NSGA和MOEAD的
        PF_ref = get_combined_pf_ref(project_path, instance_name)
        moead.PF_ref = PF_ref
        EP_list = moead.run()

        # 保存帕累托前沿到CSV
        import pandas as pd
        pd.DataFrame(EP_list, columns=['Time', 'Energy', 'Cost']).to_csv(alg_dir + str(I) + '/PF.csv', index=False)

        # === 可视化三目标帕累托前沿（可交互3D） ===
        EP_array = np.array(EP_list)
        if EP_array.shape[1] >= 3:  # 确保有三目标
            fig = go.Figure(data=[go.Scatter3d(
                x=EP_array[:, 0],
                y=EP_array[:, 1],
                z=EP_array[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=EP_array[:, 2],  # 用 Cost 作为颜色
                    colorscale='Viridis',
                    opacity=0.8
                )
            )])
            fig.update_layout(
                title='MOEAD Pareto Frontier (Run {})'.format(I),
                scene=dict(
                    xaxis_title='Time',
                    yaxis_title='Energy',
                    zaxis_title='Cost'
                ),
                width=800,
                height=600
            )
            fig.show()
        else:
            print("Warning: EP_list does not have 3 objectives, skipping 3D plot.")

        endTime = time.time()
        each_run_ExeTime_list.append(endTime - startTime)



if __name__ == '__main__':
    args = get_argument_parser()
    print_info('************************  MOEA/D-1 ************************')
    print_info(
        '\n---------------------- Test instance:  [' + str(args.Nij) + ',' + str(args.K) + ']------------------------')

    '''创建测试集目录'''
    instance_name = '[' + str(args.Nij) + ',' + str(args.K) + ']'
    args.save_dir = os.path.join(args.save_dir, instance_name, 'MOEAD_1_multisat')
    os.makedirs(args.save_dir, exist_ok=True)

    extra = "Running time: {} | popSize: {} | maxGen: {}"\
        .format(args.runTime, args.popSize, args.maxGen)
    print_info(extra, '\n')

    MOEAD_1_run(instance_name, args)



