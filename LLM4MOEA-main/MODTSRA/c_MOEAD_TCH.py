'''''
Gong Y, Bian K, Hao F, et al.
Dependent tasks offloading in mobile edge computing:
a multi-objective evolutionary optimization strategy[J].
Future Generation Computer Systems, 2023, 148: 314-325
'''''
from Utils import *
import os, time, random
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import DataFrame

np.random.seed(1)
random.seed(1)

class MOEAD_TCH:
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

        self.workflow_set = []
        self.get_workflow_set(project_path +'/instance/' + instance_name)
        self.UAV_transmission_rate_set = self.get_UAV_transmission_rate()

        self.a = [0.2, 0.5, 0.8, 1]  # The frequency scaling factors
        self.M = len(self.a)  # M different frequency levels.
        self.IGDValue = None
        self.IGD_list = []     # 保存100代的IGD值
        # self.PF_ref = self.get_PF_ref()




    def run(self):
        self.initializeWeightVectorAndNeighbor()
        self.initializePopulation()
        self.initializeReferencePoint()
        self.fast_non_dominated_sort(self.population)
        self.initializeEP(self.F_rank[1])

        t = 1
        while (t <= self.maxGen):
            for i in range(self.popSize):
                y_ = self.reproduction(i)
                self.updateNeighborSolutions(i, y_)
                self.updateReferencePoint(y_)
                self.update_EP_FromElement(self.EP, y_)
            # PF_know = [copy.deepcopy(ind.fitness) for ind in self.EP]
            if t % 20 == 0:
                print('Generation ', t)
            t += 1

        for ep in self.EP:
            ep.temp_fitness = ep.fitness[0]
        test_fast = sorted(self.EP, key=lambda Individual: Individual.temp_fitness)
        EP_list = [copy.deepcopy(ind.fitness) for ind in test_fast]
        return EP_list # 返回最终的非支配解集

    """
        **********************************************run**********************************************
    """
    def initializeEP(self, F_rank):
        for ind in F_rank:
            self.EP.append(copy.deepcopy(ind))


    def initializeWeightVectorAndNeighbor(self):
        # all_weights = pd.read_csv(project_path + '/instance/all_weights.csv').values  # DataFrame转换成numpy.array
        # for i in range(self.popSize):
        #     self.VT[i] = list(all_weights[i])
        H = self.popSize - 1
        for i in range(0, H + 1):
            w = []
            w1 = i / H - 0.0
            w2 = 1.0 - i / H
            w.append(w1)
            w.append(w2)
            self.VT[i] = w

        for i in self.VT.keys():
            distance = []
            for j in self.VT.keys():
                if(i != j):
                    tup = (j, self.getDistance(self.VT[i], self.VT[j]))
                    distance.append(tup)
            distance= sorted(distance, key=lambda x:x[1])
            neighbor = []
            for j in range(self.T):
                neighbor.append(distance[j][0])
            self.B[i] = neighbor


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
        fitness_1 = [] #存储所有个体的第一个适应度值
        fitness_2 = [] #存储所有个体的第二个适应度值
        # fitness_3 = [] #存储所有个体的第三个适应度值
        for ind in self.population:
            fitness_1.append(ind.fitness[0])
            fitness_2.append(ind.fitness[1])
            # fitness_3.append(ind.fitness[2])
        self.Z.append(min(fitness_1))
        self.Z.append(min(fitness_2))
        # self.Z.append(min(fitness_3))


    def reproduction(self, i):
        k = random.choice(self.B[i])
        l = random.choice(self.B[i])
        ind_k = Individual()
        ind_l = Individual()
        uav_k_set = []
        for gene in self.population[k].chromosome['S']:
            smd_k = copy.deepcopy(gene)
            self.reInitialize_WorkflowTaskSet_Schedule(smd_k)
            uav_k_set.append(smd_k)
        ind_k.chromosome = {'S': uav_k_set,
                            'A': copy.copy(self.population[k].chromosome['A']),
                            'B': copy.copy(self.population[k].chromosome['B'])}

        uav_l_set = []
        for gene in self.population[l].chromosome['S']:
            smd_l = UAV(self.K)
            smd_l.workflow.order = copy.copy(gene.workflow.order)
            smd_l.workflow.location = copy.copy(gene.workflow.location)
            uav_l_set.append(smd_l)
        ind_l.chromosome = {'S': uav_l_set,
                            'A': copy.copy(self.population[l].chromosome['A']),
                            'B': copy.copy(self.population[l].chromosome['B'])}
        self.crossoverOperator(ind_k, ind_l)
        self.mutantOperator(ind_k)
        self.calculateFitness(ind_k)
        return ind_k


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


    def calculatePopulationFitness(self, population):
        for ind in population:
            self.calculateFitness(ind)


    def calculateFitness(self, ind):
        self.repair_infeasible_caching_decision(ind)
        ind.fitness = []
        time = []
        energy = []
        # cost = []
        for index in range(N_uav):
            uav = ind.chromosome['S'][index]  #一个gene就是一个UAV
            self.calculateWorkflowTimeEnergy(ind, uav, uav.workflow)
            time.append(uav.workflow.schedule.T_total)
            energy.append(uav.workflow.schedule.E_total)
            # cost.append(uav.workflow.schedule.C_total)
        ind.fitness.append(np.average(time))
        ind.fitness.append(np.average(energy))
        # ind.fitness.append(np.average(cost))


    def repair_infeasible_caching_decision(self, ind):
        while True:
            s = 0
            all_1_set = [] # 保存决策为1的索引和程序规模
            for j in range(N_service):
                s += ind.chromosome['B'][j] * h_s_list[j]
                if ind.chromosome['B'][j] == 1:
                    all_1_set.append((j, h_s_list[j]))
            if s <= h_hap:
                break
            all_1_set = sorted(all_1_set, key=itemgetter(1), reverse=True)
            ind.chromosome['B'][all_1_set[0][0]] = 0


    def calculateWorkflowTimeEnergy(self, ind, uav, workflow):
        workflow.schedule.TimeEnergy = []
        workflow.schedule.T_total = None
        workflow.schedule.E_total = 0
        workflow.schedule.C_total = 0


        for i in range(len(workflow.order)):
            taskId = workflow.order[i]
            pos = workflow.location[i]
            task = workflow.taskSet[taskId]
            task.exePosition = pos
            if pos == self.K+1:   # The task is executed on the HAP.
                task.islocal = False
                if task.id == workflow.entryTask:
                    task.RT_i_l = task.ST_i_l = task.FT_i_l = 0
                    task.RT_i_ws = task.ST_i_ws = 0.0
                    task.FT_i_ws = task.ST_i_ws + task.I_i_j / uav.R_i
                    task.RT_i_c = task.ST_i_c = task.FT_i_ws

                    if ind.chromosome['B'][task.S_i_j] == 1: # 判断HAP是否缓存了任务请求的服务
                        task.FT_i_c = task.ST_i_c + task.M_i_j / (uav.ratio * uav.coreCC[pos])
                    else:
                        task.FT_i_c = task.ST_i_c + task.M_i_j / (uav.ratio * uav.coreCC[pos]) + D_ij_dow

                    task.RT_i_wr = task.ST_i_wr = task.FT_i_c
                    task.FT_i_wr = task.ST_i_wr + task.O_i_j / uav.R_i
                    workflow.schedule.wsTP.append(task.FT_i_ws)
                    workflow.schedule.MECTP.append(task.FT_i_c)
                    workflow.schedule.wrTP.append(task.FT_i_wr)
                else:
                    task.RT_i_ws = self.get_RT_i_ws(task, workflow)
                    task.ST_i_l = float("inf")
                    task.FT_i_l = float("inf")
                    if workflow.schedule.wsTP[-1] < task.RT_i_ws:
                        task.ST_i_ws = task.RT_i_ws
                        task.FT_i_ws = task.ST_i_ws + task.I_i_j / uav.R_i
                    else:
                        task.ST_i_ws = workflow.schedule.wsTP[-1]
                        task.FT_i_ws = task.ST_i_ws + task.I_i_j / uav.R_i
                    workflow.schedule.wsTP.append(task.FT_i_ws)

                    task.RT_i_c = self.get_RT_i_c(task, workflow)
                    if workflow.schedule.MECTP[-1] < task.RT_i_c:
                        task.ST_i_c = task.RT_i_c
                        if ind.chromosome['B'][task.S_i_j] == 1: # 判断HAP是否缓存了任务请求的服务
                            task.FT_i_c = task.ST_i_c + task.M_i_j / (uav.ratio * uav.coreCC[pos])
                        else:
                            task.FT_i_c = task.ST_i_c + task.M_i_j / (uav.ratio * uav.coreCC[pos]) + D_ij_dow
                    else:
                        task.ST_i_c = workflow.schedule.MECTP[-1]
                        if ind.chromosome['B'][task.S_i_j] == 1: # 判断HAP是否缓存了任务请求的服务
                            task.FT_i_c = task.ST_i_c + task.M_i_j / (uav.ratio * uav.coreCC[pos])
                        else:
                            task.FT_i_c = task.ST_i_c + task.M_i_j / (uav.ratio * uav.coreCC[pos]) + D_ij_dow

                    workflow.schedule.MECTP.append(task.FT_i_c)

                    task.RT_i_wr = task.FT_i_c
                    if workflow.schedule.wrTP[-1] < task.RT_i_wr:
                        task.ST_i_wr = task.RT_i_wr
                        task.FT_i_wr = task.ST_i_wr + task.O_i_j / uav.R_i
                    else:
                        task.ST_i_wr = workflow.schedule.wrTP[-1]
                        task.FT_i_wr = task.ST_i_wr + task.O_i_j / uav.R_i
                    workflow.schedule.wrTP.append(task.FT_i_wr)
                task.energy += uav.pws_i * (task.FT_i_ws - task.ST_i_ws)
                task.energy += uav.pwr_i * (task.FT_i_wr - task.ST_i_wr)
                workflow.schedule.E_total += task.energy
                workflow.schedule.C_total += self.calculateComputingCost(uav.ratio * uav.coreCC[pos],
                                                                         task.M_i_j / (uav.ratio * uav.coreCC[pos]))
            else:          # The task is executed on a local core.
                task.islocal = True
                task.RT_i_ws = task.RT_i_c = task.RT_i_wr = 0.0
                task.ST_i_ws = task.ST_i_c = task.ST_i_wr = 0.0
                task.FT_i_ws = task.FT_i_c = task.FT_i_wr = 0.0
                if task.id == workflow.entryTask:
                    task.RT_i_l = task.ST_i_l = 0
                    task.FT_i_l = task.ST_i_l + task.M_i_j / uav.coreCC[pos]
                else:
                    task.RT_i_l = self.get_RT_i_l(task, workflow)
                    if task.RT_i_l > workflow.schedule.coreTP[pos][-1]:
                        task.ST_i_l = task.RT_i_l
                    else:
                        task.ST_i_l = workflow.schedule.coreTP[pos][-1]
                    task.FT_i_l = task.ST_i_l + task.M_i_j / uav.coreCC[pos]
                workflow.schedule.coreTP[pos].append(task.FT_i_l)
                task.energy = uav.pcc_i[pos] * (task.FT_i_l - task.ST_i_l)
                workflow.schedule.E_total += task.energy
            workflow.schedule.S[pos].append(task.id)
        if workflow.taskSet[workflow.exitTask].islocal == True:
            workflow.schedule.T_total = workflow.taskSet[workflow.exitTask].FT_i_l
        else:
            workflow.schedule.T_total = workflow.taskSet[workflow.exitTask].FT_i_wr
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

class UAV:
    def __init__(self, K):
        self.coordinate = []    # The position coordination of the UAV
        self.workflow = Workflow(K)      #The workflow of the UAV
        self.ratio = None
        self.channel = None     # Gaining channel index
        self.R_i = None       # The data transmission rate of the UAV
        self.K = K
        #The UAV is modeled as a 3-tuple
        if K == 3:
            self.coreCC = {1:1, 2:0.8, 3:0.6, 4:f_hap}        # The computing capacity of three core.
        else:
            self.coreCC = {1:1.6, 2:1.4, 3:1.2, 4:1, 5:0.8, 6:0.6, 7:f_hap}

        if K == 3:# The power consumption of the three cores under the maximum operating frequency.
            self.pcc_i = {1:4, 2:2, 3:1}
        else:
            self.pcc_i = {1:10, 2:8, 3:6, 4:4, 5:2, 6:1}


        self.pws_i = p_i_upl  # The send data power (w) of the UAV
        self.pwr_i = p_i_rec  # The receive data power (w) of the UAV

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



def MOEAD_TCH_run(instance_name, args):
    alg_dir = project_path+'/ExperimentResult/' + instance_name + '/MOEAD_TCH/'
    each_run_ExeTime_list = []
    for I in range(1, args.runTime + 1):
        if os.path.isdir(alg_dir + str(I)) == False:
            os.makedirs(alg_dir + str(I))

        startTime = time.time()
        moead = MOEAD_TCH(instance_name, args)
        EP_list = moead.run()
        endTime = time.time()

        CT = CT = endTime - startTime
        each_run_ExeTime_list.append(CT)
        print_info("\n\n " + str(I) + "-th time run", '| Exe. Time: ', CT)

        DataFrame({'Time': np.array(EP_list)[:, 0],
                   'Energy': np.array(EP_list)[:, 1]}).to_csv(alg_dir + str(I) + '/PF.csv', index=False)






if __name__ == '__main__':
    args = get_argument_parser()
    print_info('************************  MOEA/D-TCH ************************')
    print_info(
        '\n---------------------- Test instance:  [' + str(args.Nij) + ',' + str(args.K) + ']------------------------')

    '''创建测试集目录'''
    instance_name = '[' + str(args.Nij) + ',' + str(args.K) + ']'
    args.save_dir = os.path.join(args.save_dir, instance_name, 'MOEAD_TCH')
    os.makedirs(args.save_dir, exist_ok=True)

    extra = "Running time: {} | popSize: {} | maxGen: {}"\
        .format(args.runTime, args.popSize, args.maxGen)
    print_info(extra, '\n')

    MOEAD_TCH_run(instance_name, args)



