import random

import numpy as np
import copy
import matplotlib.pyplot as plt
from operator import itemgetter
import argparse
random.seed(1)
project_path = 'D:/post programs/LLM4MOEA-main/LLM4MOEA-main/MODTSRA'
# 新增
G_m = 12 #地面用户的数量
F_m = 0.6 # 地面用户设备的计算能力(GHz)
k_hw = 1e-26
h_los = 0 # 直射路径LoS的信道系数
h_Nlos = 0 # 多径散射的信道系数

# :param p_G: 移动设备的发射功率
# :param T_mn: 传输时间
p_G = 0  # 从其他部分获取或传入
T_mn = 0  # 从其他部分获取或传入


B_mn = 0 # 地面用户和无人机之间的信道宽带，
h_2 = 0 # 无线信道功率增益（表征信道质量）（地面用户到无人机）
sigma_squared = 0.1 # 地面用户到无人机的上行传输速率中的  “噪声功率”
p_u = 0 # 地面用户发射功率
D_i = 0 # 子任务v_i的数据量的大小。
delta_t = 0 # 指的是第k个处理器上的计算频率
P_nk_max = 0 # 第k个核心上的最大计算频率执行子任务的最小执行延迟
f_sat = 1.0 ####假设卫星计算能力为1.0GHz
###### 令卫星为f_Sat

f_x_i_s = 2.0
T_i_s = 5.0
alpha = 0.5
beta=10



# #
N_uav = 3
N_n = 3 # 无人机数量
N_service = 10 # 服务程序的数量
h_hap = 250 # HAP的存储空间
h_s_list = [random.randint(20,100) for _ in range(N_service)] # 每个服务程序的规模

F = 5 # 计算频率水平数量
f_hap = 10 # HAP计算能力（GHz）
obj_num = 2 # 优化目标数量
g_0 = -30  # 信道功率增益（dB）
p_i_upl = 10 # 无人机的上传功率（W）
p_i_rec = 2 # 无人机的接收功率（W）
z_i_U = 30 # 无人机的飞行高度
z_H = 20000 # HAP的漂浮高度
Psi = 20  # 信道带宽（MHz）
sigma_2 = pow(10,-6) # 噪声功率
kappa_i = pow(10, -26) # 有效电容系数
Lambda = 2
uav_coordinate_set = [[50,  134, z_i_U],
                      [150, 134, z_i_U],
                      [100, 67,  z_i_U]]# 三个无人机的坐标
D_ij_dow = 1 #从云中心下载服务的时延
epsilon_1 = 1 # 计算费用参数
epsilon_2 = 2.5 * pow(10, -3) # 计算费用参数




def get_argument_parser():
    parser = argparse.ArgumentParser()

    '''***************************问题相关参数***************************'''
    parser.add_argument('--Nij', type=int, default=20, metavar='N_task',
                        help='Number of tasks')

    parser.add_argument('--K', type=int, default=6, metavar='N_core',
                        help='Number of cores')
    '''***************************问题相关参数***************************'''


    '''***************************算法相关参数***************************'''
    parser.add_argument('--popSize', type=int, default=100, metavar='N_pop',
                        help='Size of population')

    parser.add_argument('--maxGen', type=int, default=100, metavar='max_gen',
                        help='Maximum number of generations')

    parser.add_argument('--runTime', type=int, default=20,
                        help='running time of the algorithm')

    parser.add_argument('--T', type=int, default=10, metavar='num_neighbor',
                        help='Number of neighbor')

    parser.add_argument('--pc', type=int, default=0.8, metavar='pc',
                        help='Crossover probability')

    parser.add_argument('--pmApp', type=int, default=0.03, metavar='pm_app',
                        help='mutation probability of application')

    parser.add_argument('--pmBit', type=int, default=0.01, metavar='pm_bit',
                        help='mutation probability of bit')

    parser.add_argument('--num_satellites', type=int, default=3,
                       help='Number of satellites in the system (default: 3)')  # 新增卫星数量参数


    parser.add_argument('--save-dir', default='D:/post programs//LLM4MOEA-main/MODTSRA/ExperimentResult',
                        help='directory to save agent logs')
    '''***************************算法相关参数***************************'''

    args = parser.parse_args()
    return args









def set_weight(eps_idx, reward_size):
    np.random.seed(eps_idx)
    w_kept = np.random.randn(reward_size)  # 从标准正太分布中抽取reward_size个随机数
    w_kept = (np.abs(w_kept) / np.linalg.norm(w_kept, ord=1))  # 求1范数
    return np.round(w_kept, 8)


def print_info(*message):
    print('\033[96m', *message, '\033[0m')

def update_EP_History(EP_current, EP_history):  # 用当前运行后的非支配解集EP_Current来更新历史非支配解集EP_History
    if (EP_history == []):
        for epc in EP_current:
            EP_history.append(copy.copy(epc))
    else:
        for epc in EP_current:
            if (isExist(epc, EP_history) == False):  # 先判断ep是否在EP_History中，若不在，则返回False。
                if (isEP_Dominated_ind(EP_history, epc) == False):  # 然后再判断EP_History是否支配ep
                    i = 0
                    while (i < EP_history.__len__()):  # 判断ep是否支配EP中的非支配解，若支配，则删除它所支配的解
                        if (isDominated(epc, EP_history[i]) == True):
                            EP_history.remove(EP_history[i])
                            i -= 1
                        i += 1
                    EP_history.append(copy.copy(epc))


def isExist(ep, EP_history):  # 判断ep是否与EP中某个支配解相对，若相等，则返回True
    for eph in EP_history:
        if ep == eph:  # 判断两个列对应元素的值是否相等
            return True
    return False


def isEP_Dominated_ind(EP_history, ep):  # 判断EP中的某个非支配解是否支配ep，若支配，则返回True
    for eph in EP_history:
        if isDominated(eph, ep):
            return True
    return False


def isDominated(fitness_1, fitness_2):  # 前者是否支配后者
    flag = -1
    for i in range(2):
        if fitness_1[i] < fitness_2[i]:
            flag = 0
        if fitness_1[i] > fitness_2[i]:
            return False
    if flag == 0:
        return True
    else:
        return False




def remove_duplicate_element(EP_list):  # 去除具有相同的适应度的个体
    temp  = [EP_list[0]]
    for p in EP_list:
        flag = True
        for q in temp:
            if (p == q).all():
                flag = False
                break
        if flag == True:
            temp.append(p)
    return temp


def fast_non_dominated_sort(population):
    F_rank = []
    for p in population:
        p.S_p = []
        p.rank = None
        p.n = 0


    F1 = []  # 第一个非支配解集前端
    F_rank.append(None)
    for p in population:
        for q in population:
            if isDominated(p.fitness, q.fitness):
                p.S_p.append(q)
            elif isDominated(q.fitness, p.fitness):
                p.n += 1
        if (p.n == 0):
            p.rank = 1
            F1.append(p)
    F_rank.append(F1)

    i = 1
    while (F_rank[i] != []):
        Q = []
        for p in F_rank[i]:
            for q in p.S_p:
                q.n -= 1
                if (q.n == 0):
                    q.rank = i + 1
                    Q.append(q)

        if(Q != []):
            i += 1
            F_rank.append(Q)
        else:
            break
    return F_rank


def get_Pareto_Front_each_epsisode(EP):
    F_rank = fast_non_dominated_sort(EP)  # 求出非支配解集
    EP_current = []  # 用列表的方式保存非支配解集
    for ind in F_rank[1]:
        EP_current.append(ind.fitness)
    return remove_duplicate_element(EP_current)


def getIGDValue(PF_ref, PF_know):
    sum = []
    for v in PF_ref:
        distance = d_v_PFSet(v, PF_know)
        sum.append(distance)
    return np.average(sum)


def getGDValue(PF_ref, PF_know):
    sum = []
    for v in PF_know:
        distance = d_v_PFSet(v, PF_ref)
        sum.append(distance)
    return np.sqrt(np.average(sum))


def d_v_PFSet(v, PFSet):  # 求v和PFSet中最近的距离
    dList = []
    for pf in PFSet:
        distance = getDistance(v, pf)
        dList.append(distance)
    return min(dList)


def getDistance(point1, point2):
    return np.sqrt(np.sum(np.square([point1[i] - point2[i] for i in range(2)])))






# 仅仅保留文件名是数字的，即获得所有次数的实验结果
# def get_algorithm_all_run_result(instance_name, algorithm_name):
#     path = '../ExperimentResult/' + instance_name + '/' + algorithm_name
#     all_result = []  # 仅仅保留文件名是数字的，即获得所有次数的实验结果
#     tmp = [str(e) for e in range(1, 101)]
#     for root, dirs, files in os.walk(path):
#         for f in files:
#             if f.split('.')[0] in tmp:
#                 all_result.append(f)
#         return all_result


def get_Pareto_Front_all_episode(EP):
    F_rank = fast_non_dominated_sort(EP)
    for ep in F_rank[1]:
        ep.temp_fitness = ep.fitness[0]
    PF_list = sorted(F_rank[1], key=lambda Individual: Individual.temp_fitness)
    PF_list = [ind.fitness for ind in PF_list]
    return np.array(PF_list)






class Individual:
    def __init__(self, fitness):
        self.fitness = fitness
        self.temp_fitness = None  # 临时适应度，计算拥挤距离的时候，按每个目标值来对类列表进行升序排序
        self.distance = 0.0
        self.rank = None
        self.S_p = []  # 种群中此个体支配的个体集合
        self.n = 0  # 种群中支配此个体的个数

