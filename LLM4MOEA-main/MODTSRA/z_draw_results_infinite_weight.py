import copy
import os

import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import os.path
from Utils import *
import pandas as pd
from operator import itemgetter
from pandas import DataFrame
from neural_network_model import *
from hypervolume import InnerHyperVolume
from envs.modag_env import DelayEnergy


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


args = get_argument_parser()

all_weights = pd.read_csv('../instance/all_weights.csv').values  # DataFrame转换成numpy.array
weight_keys_None = {}
weight_keys_list = {}
for w in all_weights:
    weight_keys_None[str(w)] = None
    weight_keys_list[str(w)] = []


def get_current_directory_filename(path):
    for root, dir_list, files in os.walk(path):
        if len(dir_list) == 0:
            temp = []
            for f in files:
                temp.append([int(f.split('.')[0]), f])
            temp = sorted(temp, key=itemgetter(0))
            return [e[1] for e in temp]
        else:
            dir_list = sorted([int(d) for d in dir_list])
            dir_list = [str(d) for d in dir_list]
        return dir_list

def plot_DR_HDR_loss(instance_list, algorithm_list):

    print('----------> Plot DR, HDR, and loss <----------')
    size = 17
    font = {'size': size}
    Marker = ['<', '+', 'x', 'v', '^', 'd', '>', '*', 'd']
    Color = ['m', 'g', 'orange', 'b', 'k', 'r', 'sienna', 'tan', 'k']

    for instance_name in instance_list:
        ins_dir = '../ExperimentResult/' + instance_name

        all_alg_data = dict.fromkeys(algorithm_list, None)
        for algorithm_name in algorithm_list:
            alg = os.path.join(ins_dir, algorithm_name)
            each_alg_all_running_dirs = get_current_directory_filename(alg)
            each_alg_all_running_data = np.zeros([EPISODE_NUMBER, 3])
            folder_number = 0
            for d in each_alg_all_running_dirs:
                path = os.path.join(alg, str(d) + '/episode.csv')
                data = pd.read_csv(path).values
                each_alg_all_running_data = each_alg_all_running_data + data[:EPISODE_NUMBER,1:4] # 取后三列
                folder_number += 1
            each_alg_all_running_data = each_alg_all_running_data / folder_number
            all_alg_data[algorithm_name] = each_alg_all_running_data

        '''----------------> 绘制DR curve <----------------'''
        plt.figure()
        FG = []
        for i, algorithm_name in enumerate(algorithm_list):
            fg, = plt.plot(np.arange(EPISODE_NUMBER), all_alg_data[algorithm_name][:EPISODE_NUMBER,0], linewidth=2, label=algorithm_name)
            FG.append(fg)
        plt.legend(handles=FG, prop=font, loc='center right')
        plt.tick_params(labelsize=size)
        plt.ylabel('DR value', fontsize=size)
        plt.xlabel('Episode', fontsize=size)

        path = ins_dir + '/DRcurve' + instance_name + '.pdf'
        plt.savefig(path, figsize=(2, 1), bbox_inches='tight')
        plt.close()

        '''----------------> 绘制HDR curve <----------------'''
        plt.figure()
        FG = []
        for i, algorithm_name in enumerate(algorithm_list):
            fg, = plt.plot(np.arange(EPISODE_NUMBER), all_alg_data[algorithm_name][:EPISODE_NUMBER, 1], markersize=10, linewidth=2,
                           label=algorithm_name)
            FG.append(fg)
        plt.legend(handles=FG, prop=font, loc='center right')
        plt.tick_params(labelsize=size)
        plt.ylabel('HDR value', fontsize=size)
        plt.xlabel('Episode', fontsize=size)

        path = ins_dir + '/HDRcurve' + instance_name + '.pdf'
        plt.savefig(path, figsize=(2, 1), bbox_inches='tight')
        plt.close()

        '''----------------> 绘制loss curve <----------------'''
        plt.figure()
        FG = []
        for i, algorithm_name in enumerate(algorithm_list):
            fg, = plt.plot(np.arange(EPISODE_NUMBER), all_alg_data[algorithm_name][:EPISODE_NUMBER, 2], markersize=10, linewidth=2,
                           label=algorithm_name)
            FG.append(fg)
        plt.legend(handles=FG, prop=font, loc='center right')
        plt.tick_params(labelsize=size)
        plt.ylabel('Loss value', fontsize=size)
        plt.xlabel('Episode', fontsize=size)

        path = ins_dir + '/Losscurve' + instance_name + '.pdf'
        plt.savefig(path, figsize=(2, 1), bbox_inches='tight')
        plt.close()


def get_global_w_opt_DR(instance_list, algorithm_list):
    for instance_name in instance_list:
        ins_path = '../ExperimentResult/' + instance_name + '/'
        all_alg_w_opt_DR = DataFrame()

        for I in range(len(algorithm_list)):
            alg_path = ins_path + algorithm_list[I] + '/'
            all_run_w_opt_DR = DataFrame()
            each_alg_all_running_dirs = get_current_directory_filename(alg_path)

            for J in each_alg_all_running_dirs:
                each_run_w_opt_DR = pd.read_csv(alg_path + J +'/w_opt_DR_fitness.csv')

                if J == str(1):
                    all_run_w_opt_DR = copy.deepcopy(each_run_w_opt_DR[['weight']])
                    all_run_w_opt_DR['DR_list'] = [[each_run_w_opt_DR.loc[k, 'DR']] for k in range(each_run_w_opt_DR.shape[0])]
                else:
                    for k in range(each_run_w_opt_DR.shape[0]):
                        all_run_w_opt_DR.loc[k, 'DR_list'].append(each_run_w_opt_DR.loc[k, 'DR'])

            for k in range(all_run_w_opt_DR.shape[0]): # 获得列表中最大的那个DR
                all_run_w_opt_DR.loc[k, 'DR_list'] = max(all_run_w_opt_DR.loc[k, 'DR_list'])
            all_run_w_opt_DR.rename(columns={'DR_list': 'DR'}, inplace=True)
            all_run_w_opt_DR.to_csv(alg_path + 'alg_w_opt_DR.csv', index=False) # 所有次数的最优写入文件

            if I == 0:
                all_alg_w_opt_DR = copy.deepcopy(all_run_w_opt_DR[['weight']])
                all_alg_w_opt_DR['DR_list'] = [[all_run_w_opt_DR.loc[k, 'DR']] for k in range(all_run_w_opt_DR.shape[0])]
            else:
                for k in range(all_run_w_opt_DR.shape[0]):
                    all_alg_w_opt_DR.loc[k, 'DR_list'].append(all_run_w_opt_DR.loc[k, 'DR'])

        for I in range(all_alg_w_opt_DR.shape[0]): # 获得每个算法每次运行后的最优DR
            all_alg_w_opt_DR.loc[I, 'DR_list'] = max(all_alg_w_opt_DR.loc[I, 'DR_list'])
        all_alg_w_opt_DR.rename(columns={'DR_list': 'DR'}, inplace=True)
        all_alg_w_opt_DR.to_csv(ins_path + 'global_w_opt_DR.csv', index=False)  # 所有次数的最优写入文件



def get_each_algorithm_each_episode_regret(instance_list, algorithm_list):
        for instance_name in instance_list:
            ins_path = '../ExperimentResult/' + instance_name + '/'
            global_w_opt_DR = {} # 将dataframe转换成dict， 'weight'作为关键字，'DR'作为其对应的值
            temp = pd.read_csv(ins_path + 'global_w_opt_DR.csv')
            for i in range(temp.shape[0]): global_w_opt_DR[temp.loc[i, 'weight']] = temp.loc[i, 'DR']

            for I in range(len(algorithm_list)):
                alg_path = ins_path + algorithm_list[I] + '/'
                each_alg_all_running_dirs = get_current_directory_filename(alg_path)

                for J in each_alg_all_running_dirs:
                    each_run_episode = pd.read_csv(alg_path + str(J) +'/episode.csv')
                    each_run_episode['regret'] = np.zeros(EPISODE_NUMBER)
                    for eps_idx in range(EPISODE_NUMBER):
                        weight = each_run_episode.loc[eps_idx, 'weight']
                        each_run_episode.loc[eps_idx, 'regret'] = \
                            global_w_opt_DR[weight] - each_run_episode.loc[eps_idx, 'DR']

                    each_run_episode['cum_regret'] = np.zeros(EPISODE_NUMBER) # 计算累计regret
                    cum_value = 0
                    for eps_idx in range(EPISODE_NUMBER):
                        cum_value += each_run_episode.loc[eps_idx, 'regret']
                        each_run_episode.loc[eps_idx, 'cum_regret'] = cum_value

                    each_run_episode.to_csv(alg_path + str(J) +'/episode.csv', index=False) # 重新写入


def get_each_algorithm_average_cumulative_regret(instance_list, algorithm_list):
    for instance_name in instance_list:
        ins_path = '../ExperimentResult/' + instance_name + '/'

        all_algo_CR = pd.DataFrame(np.zeros((EPISODE_NUMBER, len(algorithm_list))), columns=algorithm_list)
        for I in range(len(algorithm_list)):
            alg_path = ins_path + algorithm_list[I] + '/'
            avg_cum_regret = np.zeros(EPISODE_NUMBER)
            each_alg_all_running_dirs = get_current_directory_filename(alg_path)

            for J in each_alg_all_running_dirs:
                each_run_cum_regret = pd.read_csv(alg_path + str(J) + '/episode.csv').loc[:, 'cum_regret'].to_numpy()
                avg_cum_regret += each_run_cum_regret
            avg_cum_regret = avg_cum_regret / len(each_alg_all_running_dirs)
            all_algo_CR[algorithm_list[I]] = avg_cum_regret
        all_algo_CR.to_csv(ins_path + 'all_algo_CR.csv', index=False)


def get_average_episodic_regret(instance_list, algorithm_list):
        for instance_name in instance_list:
            ins_path = '../ExperimentResult/' + instance_name + '/'

            all_alg_AER = []
            for I in range(len(algorithm_list)):
                alg_path = ins_path + algorithm_list[I] + '/'
                each_run_AER = []
                each_alg_all_running_dirs = get_current_directory_filename(alg_path)

                for J in each_alg_all_running_dirs:
                    AER = np.average(pd.read_csv(alg_path + str(J) + '/episode.csv').loc[:, 'regret'])
                    each_run_AER.append(AER)

                all_alg_AER.append(each_run_AER)
                DataFrame(each_run_AER).to_csv(alg_path + 'each_run_AER.csv', index=False)

        # 求出所有算法的average episode regret
        AER = DataFrame()
        for algorithm_name in algorithm_list:
            AER[algorithm_name] = np.zeros(len(instance_list))
            for j in range(len(instance_list)):
                data = pd.read_csv('../ExperimentResult/' + instance_list[j] + '/' + algorithm_name + '/each_run_AER.csv')
                AER.loc[j, algorithm_name] = np.average(data)
        print(AER)
        AER.to_csv('../ExperimentResult/Average episodic regret.csv')

def get_each_algorithm_Pareto_front(instance_list, algorithm_list):
    for instance_name in instance_list:
        ins_dir = '../ExperimentResult/' + instance_name

        for algorithm_name in algorithm_list:
            alg = os.path.join(ins_dir, algorithm_name)
            each_alg_all_running_dirs = get_current_directory_filename(alg)

            '''获得每个算法的Pareto Front'''
            alg_population = []
            for d in each_alg_all_running_dirs:
                path = os.path.join(alg, str(d) + '/PF.csv')
                data_PF = pd.read_csv(path).values
                for ep in data_PF:
                    ind = Individual(list(ep))
                    alg_population.append(ind)
            F_rank = fast_non_dominated_sort(alg_population)
            for ep in F_rank[1]:
                ep.temp_fitness = ep.fitness[0]
            temp = sorted(F_rank[1], key=lambda Pareto: Pareto.temp_fitness)
            PF_list = [np.array(ind.fitness) for ind in temp]
            PF_list = remove_duplicate_element(PF_list)
            DataFrame(PF_list).to_csv(alg + '/algorithmPF.csv', index=False)
            '''获得每个算法的Pareto Front'''


def plot_all_algorithm_Pareto_front_and_referPF(instance_list, algorithm_list):
    print('----------> Plot Pareto front <----------')
    size = 12
    font = {'size': size}
    Marker = ['<', '^', 'o', '>', 'd', '>', '*', 'd']
    Color = ['Green', 'Blue', 'orange', 'Violet', 'Black', 'Black', 'sienna', 'tan', 'k']

    for instance_name in instance_list:
        ins_dir = '../ExperimentResult/' + instance_name

        all_alg_PF = dict.fromkeys(algorithm_list, None)
        population = []
        for algorithm_name in algorithm_list:
            alg = os.path.join(ins_dir, algorithm_name)
            all_alg_PF[algorithm_name] = pd.read_csv(alg+'/algorithmPF.csv').values
            for ep in all_alg_PF[algorithm_name]:
                ind = Individual(list(ep))
                population.append(ind)
        F_rank = fast_non_dominated_sort(population)
        for ep in F_rank[1]:
            ep.temp_fitness = ep.fitness[0]
        temp = sorted(F_rank[1], key=lambda Pareto: Pareto.temp_fitness)
        PF_list = [np.array(ind.fitness) for ind in temp]
        PF_list = remove_duplicate_element(PF_list)
        DataFrame(PF_list).to_csv(ins_dir + '/referPF.csv', index=False)
        PF_list = np.array(PF_list)

        plt.figure()
        FG = []
        algorithm_list_fig = ['MORL-DW', 'MORL-TS', 'MORL-NS', 'MORL-EN', 'MORL-TBER']
        for i, algorithm_name in enumerate(algorithm_list):
            fg, = plt.plot(all_alg_PF[algorithm_name][:, 0], all_alg_PF[algorithm_name][:, 1],
                           marker=Marker[i], markersize=3, color=Color[i],
                           linestyle='', linewidth=2, label=algorithm_list_fig[i])
            FG.append(fg)

        # fg, = plt.plot(PF_list[:, 0], PF_list[:, 1],
        #                    marker='o', markersize=1, color='sienna',
        #                    linestyle='', linewidth=2, label='refer PF')
        # FG.append(fg)

        plt.legend(handles=FG, prop=font, loc='upper right')
        plt.tick_params(labelsize=size)
        plt.ylabel('Energy consumption', fontsize=size)
        plt.xlabel('Application delay', fontsize=size)
        path = 'F:/基金/2024国家自然青年基金/申请书/图片'+'/PF' + instance_name + '.svg'
        plt.savefig(path, figsize=(2, 1), bbox_inches='tight')
        plt.close()



def get_IGD_GD_normalize(instance_list, algorithm_list):
    IGD = pd.DataFrame(np.zeros((len(instance_list), len(algorithm_list))),
                       columns=algorithm_list)  # 每一行是一个instance，每一列是一个算法
    GD = pd.DataFrame(np.zeros((len(instance_list), len(algorithm_list))),
                      columns=algorithm_list)  # 每一行是一个instance，每一列是一个算法
    # ***************************求得最小最大值***************************
    for ins_index, instance_name in enumerate(instance_list):
        ins_dir = '../ExperimentResult/' + instance_name
        PF_ref = pd.read_csv(ins_dir+'/referPF.csv').values
        min_ref = [np.min(PF_ref[:, 0]), np.min(PF_ref[:, 1])]
        max_ref = [np.max(PF_ref[:, 0]), np.max(PF_ref[:, 1])]

        MIN = [copy.deepcopy(min_ref)]
        MAX = [copy.deepcopy(max_ref)]
        for algorithm_name in algorithm_list:
            alg = os.path.join(ins_dir, algorithm_name)
            PF_know = pd.read_csv(alg + '/algorithmPF.csv').values
            min_know = [np.min(PF_know[:, 0]), np.min(PF_know[:, 1])]
            max_know = [np.max(PF_know[:, 0]), np.max(PF_know[:, 1])]
            MIN.append(copy.deepcopy(min_know))
            MAX.append(copy.deepcopy(max_know))

        MIN = np.array(MIN)
        MAX = np.array(MAX)
        MIN = [np.min(MIN[:, 0]), np.min(MIN[:, 1])]
        MAX = [np.max(MAX[:, 0]), np.max(MAX[:, 1])]
    # ***************************求得最小最大值***************************

        for i, pf in enumerate(PF_ref):
            for j in range(2):
                PF_ref[i][j] = (pf[j] - MIN[j]) / (MAX[j] - MIN[j])

        for algorithm_name in algorithm_list:
            alg = os.path.join(ins_dir, algorithm_name)
            PF_know = pd.read_csv(alg + '/algorithmPF.csv').values
            for i, pf in enumerate(PF_know):
                for j in range(2):
                    PF_know[i][j] = (pf[j] - MIN[j]) / (MAX[j] - MIN[j])
            IGD[algorithm_name][ins_index] = getIGDValue(PF_ref, PF_know)
            GD[algorithm_name][ins_index] = getGDValue(PF_ref, PF_know)
    print('************* IGD *************')
    print(IGD, '\n\n')
    IGD.to_csv('../ExperimentResult/Inverted Generational Distance.csv')

    print('************* GD *************')
    print(GD, '\n\n')
    GD.to_csv('../ExperimentResult/Generational Distance.csv')

def get_IGD_GD_no_normalize(instance_list, algorithm_list):
    IGD = pd.DataFrame(np.zeros((len(instance_list), len(algorithm_list))),
                       columns=algorithm_list)  # 每一行是一个instance，每一列是一个算法
    GD = pd.DataFrame(np.zeros((len(instance_list), len(algorithm_list))),
                      columns=algorithm_list)  # 每一行是一个instance，每一列是一个算法

    for ins_index, instance_name in enumerate(instance_list):
        ins_dir = '../ExperimentResult/' + instance_name
        PF_ref = pd.read_csv(ins_dir+'/referPF.csv').values

        for algorithm_name in algorithm_list:
            alg = os.path.join(ins_dir, algorithm_name)
            PF_know = pd.read_csv(alg + '/algorithmPF.csv').values
            IGD[algorithm_name][ins_index] = getIGDValue(PF_ref, PF_know)
            GD[algorithm_name][ins_index] = getGDValue(PF_ref, PF_know)
    print('************* IGD *************')
    print(IGD, '\n\n')
    IGD.to_csv('../ExperimentResult/Inverted Generational Distance.csv')

    print('************* GD *************')
    print(GD, '\n\n')
    GD.to_csv('../ExperimentResult/Generational Distance.csv')


def get_HV_normalize(instance_list, algorithm_list):
    HV = pd.DataFrame(np.zeros((len(instance_list), len(algorithm_list))),
                       columns=algorithm_list)  # 每一行是一个instance，每一列是一个算法

    # ***************************求得最小最大值***************************
    for ins_index, instance_name in enumerate(instance_list):
        ins_dir = '../ExperimentResult/' + instance_name
        PF_ref = pd.read_csv(ins_dir + '/referPF.csv').values
        min_ref = [np.min(PF_ref[:, 0]), np.min(PF_ref[:, 1])]
        max_ref = [np.max(PF_ref[:, 0]), np.max(PF_ref[:, 1])]

        MIN = [copy.deepcopy(min_ref)]
        MAX = [copy.deepcopy(max_ref)]
        for algorithm_name in algorithm_list:
            alg = os.path.join(ins_dir, algorithm_name)
            PF_know = pd.read_csv(alg + '/algorithmPF.csv').values
            min_know = [np.min(PF_know[:, 0]), np.min(PF_know[:, 1])]
            max_know = [np.max(PF_know[:, 0]), np.max(PF_know[:, 1])]
            MIN.append(copy.deepcopy(min_know))
            MAX.append(copy.deepcopy(max_know))

        MIN = np.array(MIN)
        MAX = np.array(MAX)
        MIN = [np.min(MIN[:, 0]), np.min(MIN[:, 1])]
        MAX = [np.max(MAX[:, 0]), np.max(MAX[:, 1])]
        # ***************************求得最小最大值***************************

        for algorithm_name in algorithm_list:
            alg = os.path.join(ins_dir, algorithm_name)
            PF_know = pd.read_csv(alg + '/algorithmPF.csv').values
            for i, pf in enumerate(PF_know):
                for j in range(2):
                    PF_know[i][j] = (pf[j] - MIN[j]) / (MAX[j] - MIN[j])
            HV[algorithm_name][ins_index] = compute_hypervolume(PF_know)

    print('************* HV *************')
    print(HV, '\n\n')
    HV.to_csv('../ExperimentResult/Hyper volume.csv')


def compute_hypervolume(ep_objs_batch):
    n = len(ep_objs_batch[0])
    HV = InnerHyperVolume(np.ones(n))
    return HV.compute(ep_objs_batch)


def get_sparsity_normalize(instance_list, algorithm_list):
    sparsity = pd.DataFrame(np.zeros((len(instance_list), len(algorithm_list))),
                       columns=algorithm_list)  # 每一行是一个instance，每一列是一个算法

    # ***************************求得最小最大值***************************
    for ins_index, instance_name in enumerate(instance_list):
        ins_dir = '../ExperimentResult/' + instance_name
        PF_ref = pd.read_csv(ins_dir + '/referPF.csv').values
        min_ref = [np.min(PF_ref[:, 0]), np.min(PF_ref[:, 1])]
        max_ref = [np.max(PF_ref[:, 0]), np.max(PF_ref[:, 1])]

        MIN = [copy.deepcopy(min_ref)]
        MAX = [copy.deepcopy(max_ref)]
        for algorithm_name in algorithm_list:
            alg = os.path.join(ins_dir, algorithm_name)
            PF_know = pd.read_csv(alg + '/algorithmPF.csv').values
            min_know = [np.min(PF_know[:, 0]), np.min(PF_know[:, 1])]
            max_know = [np.max(PF_know[:, 0]), np.max(PF_know[:, 1])]
            MIN.append(copy.deepcopy(min_know))
            MAX.append(copy.deepcopy(max_know))

        MIN = np.array(MIN)
        MAX = np.array(MAX)
        MIN = [np.min(MIN[:, 0]), np.min(MIN[:, 1])]
        MAX = [np.max(MAX[:, 0]), np.max(MAX[:, 1])]
        # ***************************求得最小最大值***************************

        for algorithm_name in algorithm_list:
            alg = os.path.join(ins_dir, algorithm_name)
            PF_know = pd.read_csv(alg + '/algorithmPF.csv').values
            for i, pf in enumerate(PF_know):
                for j in range(2):
                    PF_know[i][j] = (pf[j] - MIN[j]) / (MAX[j] - MIN[j])
            sparsity[algorithm_name][ins_index] = compute_sparsity(PF_know)

    print('************* sparsity *************')
    print(sparsity, '\n\n')
    sparsity.to_csv('../ExperimentResult/sparsity.csv')



def compute_sparsity(ep_objs_batch):
    if len(ep_objs_batch) < 2:
        return 0.0

    sparsity = 0.0
    m = len(ep_objs_batch[0])
    ep_objs_batch_np = np.array(ep_objs_batch)
    for dim in range(m):
        objs_i = np.sort(copy.deepcopy(ep_objs_batch_np.T[dim]))
        for i in range(1, len(objs_i)):
            sparsity += np.square(objs_i[i] - objs_i[i - 1])
    sparsity /= (len(ep_objs_batch) - 1)

    return sparsity


# 通过保存的神经网络model获得Naive的所有all_weight 下对应的DR
def get_preference_adaptation(instance_list, algorithm_list):
    all_weights = pd.read_csv('../instance/all_weights.csv').values  # DataFrame转换成numpy.array
    APA = DataFrame(np.zeros((len(instance_list), len(algorithm_list))), columns=algorithm_list)

    for ins_index, instance_name in enumerate(instance_list):
        ins_dir = '../ExperimentResult/' + instance_name

        for algorithm_name in algorithm_list:
            all_run_AE = [] # 保存算法每次运行的AE
            alg_path = os.path.join(ins_dir, algorithm_name)
            each_alg_all_running_dirs = get_current_directory_filename(alg_path)

            for J in each_alg_all_running_dirs:
                env = DelayEnergy(instance_name)
                model = torch.load(alg_path + '/' + J + '/model.pt')

                temp_AE = []
                for eps_idx in range(len(all_weights)):
                    one_episode_rewards = []
                    current_state = env.reset()
                    current_weight = all_weights[eps_idx]
                    temp_weight = current_weight
                    current_weight = torch.from_numpy(current_weight).type(FloatTensor)

                    for time_step in range(1, env.N + 1):
                        current_state = torch.from_numpy(current_state).type(FloatTensor)

                        if algorithm_name in ['MORLEN', 'MORLSC']:
                            _, q = model(Variable(current_state.unsqueeze(0)),
                                      Variable(current_weight.unsqueeze(0)))
                        else:
                            q = model(Variable(current_state.unsqueeze(0)),
                                      Variable(current_weight.unsqueeze(0)))
                        q = q.view(-1, 2)
                        q_weight = torch.mv(q, current_weight)
                        action = q_weight.max(0)[1].cpu().numpy()
                        action = int(action)

                        next_state, reward, done = env.step(time_step, action)
                        if done:
                            break
                        current_state = next_state
                        one_episode_rewards.append(np.array(reward))

                    disc_actual = np.sum(np.array([(1 ** i) * r for i, r in enumerate(one_episode_rewards)]), axis=0)
                    disc_actual = np.dot(disc_actual, temp_weight)
                    temp_AE.append(abs(disc_actual))
                all_run_AE.append(np.average(temp_AE))

            APA[algorithm_name][ins_index] = np.average(all_run_AE)

    print('************* PA *************')
    print(APA, '\n\n')
    APA.to_csv('../ExperimentResult/preference adaptation.csv')


def get_AAD_AEC_ACOI(instance_list, algorithm_list):
    all_weights = pd.read_csv('../instance/all_weights.csv').values  # DataFrame转换成numpy.array

    AAD = pd.DataFrame(np.zeros((len(instance_list), len(algorithm_list))), columns=algorithm_list)  # 每一行是一个instance，每一列是一个算法
    AEC = pd.DataFrame(np.zeros((len(instance_list), len(algorithm_list))), columns=algorithm_list)  # 每一行是一个instance，每一列是一个算法
    ACOI = pd.DataFrame(np.zeros((len(instance_list), len(algorithm_list))), columns=algorithm_list)  # 每一行是一个instance，每一列是一个算法

    for ins_index, instance_name in enumerate(instance_list):
        ins_path = '../ExperimentResult/' + instance_name + '/'
        for I, algorithm_name in enumerate(algorithm_list):
            alg = os.path.join(ins_path, algorithm_name)
            PF_know = pd.read_csv(alg + '/algorithmPF.csv').values

            data = []
            for j, weight in enumerate(all_weights):
                index = np.argmin(np.dot(PF_know, weight))
                weight_sum = np.dot(PF_know, weight)[index]
                temp = list(PF_know[index])
                temp.append(weight_sum)
                data.append(temp)
            data = np.array(data)
            AAD.loc[ins_index, algorithm_name] = np.round(np.average(data[:, 0]), 4)
            AEC.loc[ins_index, algorithm_name] = np.round(np.average(data[:, 1]), 4)
            ACOI.loc[ins_index, algorithm_name] = np.round(np.average(data[:, 2]), 4)

    print('---- AAD ----\n: ', AAD,'\n')
    print('---- AEC ----\n: ', AEC,'\n')
    print('---- ACOI ---\n: ', ACOI,'\n')

    AAD.to_csv('../ExperimentResult/AAD.csv', index=False)
    AEC.to_csv('../ExperimentResult/AEC.csv', index=False)
    ACOI.to_csv('../ExperimentResult/ACOI.csv', index=False)


def Friedman_test(algorithm_list, metrics_list):
    for metrics in metrics_list:
        # 第7列保存每个算法在所有instance上的average rank
        # 第8列保存每个算法的position
        result = DataFrame(np.zeros((11, 5)), columns=algorithm_list)
        metrics_data = pd.read_csv('../ExperimentResult/' + metrics + '.csv')
        tmp = copy.deepcopy(algorithm_list)
        tmp.insert(0, '0')
        metrics_data.columns = tmp
        metrics_data.drop(['0'], axis=1, inplace=True) #删除第一列
        for i in range(metrics_data.shape[0]):
            alg_value = metrics_data.loc[i, :].sort_values(ascending=True)

            alg_list = alg_value.index
            for j in range(0, len(alg_list)):
                if j == 0:
                    result.loc[i, alg_list[j]] = j+1
                else:
                    if alg_value[j] == alg_value[j-1]:
                        result.loc[i, alg_list[j]] = result.loc[i, alg_list[j-1]]
                    else:
                        result.loc[i, alg_list[j]] = j + 1

        for alg_name in result.columns:
            result.loc[9, alg_name] = result.loc[:8, alg_name].mean()

        position_value = result.loc[9, :].sort_values()
        position_list = position_value.index
        for j in range(0, len(position_value)):
            if j == 0:
                result.loc[10, position_list[j]] = j+1
            else:
                if position_value[j] == position_value[j-1]:
                    result.loc[10, position_list[j]] = result.loc[10, position_list[j-1]]
                else:
                    result.loc[10, position_list[j]] = j + 1

        print(result, '\n\n')
        result.to_csv('../ExperimentResult/' + metrics + '_Friedman.csv', index=False)




if __name__ == '__main__':
    # instance_list = ['[10,3]', '[10,6]', '[10,9]',
    #                  '[20,3]', '[20,6]', '[20,9]',
    #                  '[30,3]', '[30,6]', '[30,9]']
    # instance_list =  ['[20,3]', '[20,6]', '[20,9]']
    # instance_list =  ['[30,3]', '[30,6]', '[30,9]']
    instance_list =  [ '[30,6]']


    algorithm_list = ['MORLDW', 'MORLTS','MORLNS', 'MORLEN', 'MORLSC']
    # algorithm_list = ['MORLEN', 'MORLSC']


    get_each_algorithm_Pareto_front(instance_list, algorithm_list)
    plot_all_algorithm_Pareto_front_and_referPF(instance_list, algorithm_list)
    # get_IGD_GD_normalize(instance_list, algorithm_list)

    # get_IGD_GD_no_normalize(instance_list, algorithm_list)
    #
    # get_HV_normalize(instance_list, algorithm_list)
    #
    # get_sparsity_normalize(instance_list, algorithm_list)

    #
    # get_preference_adaptation(instance_list, algorithm_list)


    # get_AAD_AEC_ACOI(instance_list, algorithm_list)

    # Friedman_test(algorithm_list, ['AAD',
    #                                'AEC',
    #                                'ACOI'])

