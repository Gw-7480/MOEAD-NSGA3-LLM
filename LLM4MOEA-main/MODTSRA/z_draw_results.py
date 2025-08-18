import os

import matplotlib.pyplot as plt

from Utils import *
import pandas as pd
from operator import itemgetter
from pandas import DataFrame
from hypervolume import InnerHyperVolume


args = get_argument_parser()

all_weights = pd.read_csv(project_path + '/instance/all_weights.csv').values  # DataFrame转换成numpy.array
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






def get_each_algorithm_Pareto_front(instance_list, algorithm_list):
    for instance_name in instance_list:
        ins_dir = project_path + '/ExperimentResult/' + instance_name

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


def get_all_algorithm_referPF(instance_list, algorithm_list):
    for instance_name in instance_list:
        ins_dir = project_path + '/ExperimentResult/' + instance_name

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



def plot_Pareto_front(instance_list, algorithm_list):
    print('----------> Plot Pareto front <----------')
    size = 12
    font = {'size': size}
    Marker = ['<', '^', 'o', '>', 'd', '>', '*', 'd']
    Color = ['Green', 'Blue', 'orange', 'Violet', 'Black', 'red', 'sienna', 'tan', 'k']
    RGB_1 = np.array([1, 86, 153])
    RGB_2 = np.array([88, 170, 90])
    RGB_3 = np.array([95, 198, 201])
    RGB_4 = np.array([79, 89, 109])
    RGB_5 = np.array([247, 144, 61])

    color1 = tuple(RGB_1 / 255)
    color2 = tuple(RGB_2 / 255)
    color3 = tuple(RGB_3 / 255)
    color4 = tuple(RGB_4 / 255)
    color5 = tuple(RGB_5 / 255)

    Color = [color1, color2, color3, color4, color5]

    for instance_name in instance_list:
        ins_dir = project_path + '/ExperimentResult/' + instance_name

        all_alg_PF = dict.fromkeys(algorithm_list, None)
        population = []
        for algorithm_name in algorithm_list:
            alg = os.path.join(ins_dir, algorithm_name)
            all_alg_PF[algorithm_name] = pd.read_csv(alg+'/algorithmPF.csv').values

        plt.figure()
        FG = []
        algorithm_list_fig = ['MOPSO-QBS', 'INSGA2-PGO', 'MOEA/D-TCH', 'MOEA/D-COP', 'MOEA/D-SAC']
        for i, algorithm_name in enumerate(algorithm_list):
            fg, = plt.plot(all_alg_PF[algorithm_name][:, 0], all_alg_PF[algorithm_name][:, 1],
                           marker=Marker[i], markersize=3, color=Color[i],
                           linestyle='', linewidth=2, label=algorithm_list_fig[i])
            FG.append(fg)

        plt.legend(handles=FG, prop=font, loc='upper right')
        plt.tick_params(labelsize=size)
        plt.ylabel('Average energy consumption', fontsize=size)
        plt.xlabel('Average execution delay', fontsize=size)
        path = 'F:/论文/2024/' + instance_name + '.pdf'
        plt.savefig(path, figsize=(2, 1), bbox_inches='tight')
        plt.show()


def plot_Pareto_front_effectiveness(instance_list, algorithm_list):
    print('----------> Plot Pareto front <----------')
    size = 14
    font = {'size': size}
    Marker = ['<', '^', 'd', '>', 'o', '>', '*', 'd']
    Color = ['Green', 'Blue', 'orange', 'Violet', 'Black', 'red', 'sienna', 'tan', 'k']

    RGB_1 = np.array([77, 133, 189])
    RGB_2 = np.array([88, 170, 90])
    RGB_3 = np.array([247, 144, 61])

    color1 = tuple(RGB_1/255)
    color2 = tuple(RGB_2/255)
    color3 = tuple(RGB_3/255)
    Color = [color1, color2, color3]

    for instance_name in instance_list:
        ins_dir = project_path + '/ExperimentResult/' + instance_name

        all_alg_PF = dict.fromkeys(algorithm_list, None)
        population = []
        for algorithm_name in algorithm_list:
            alg = os.path.join(ins_dir, algorithm_name)
            all_alg_PF[algorithm_name] = pd.read_csv(alg+'/algorithmPF.csv').values

        plt.figure()
        FG = []
        algorithm_list_fig = [ 'MOEA/D-ORI', 'MOEA/D-PNS', 'MOEA/D-SAC']
        for i, algorithm_name in enumerate(algorithm_list):
            fg, = plt.plot(all_alg_PF[algorithm_name][:, 0], all_alg_PF[algorithm_name][:, 1],
                           marker=Marker[i], markersize=3, color=Color[i],
                           linestyle='', linewidth=2, label=algorithm_list_fig[i])
            FG.append(fg)


        plt.legend(handles=FG, prop=font, loc='upper right')
        plt.tick_params(labelsize=size)
        plt.ylabel('Average energy consumption', fontsize=size)
        plt.xlabel('Average execution delay', fontsize=size)
        path = 'F:/论文/2024/IEEE/MOEAD-SAC/figs/effectiveness/' + instance_name + '.pdf'
        plt.savefig(path, figsize=(2, 1), bbox_inches='tight')
        plt.show()



def get_IGD_GD_normalize(instance_list, algorithm_list):
    IGD = pd.DataFrame(np.zeros((len(instance_list), len(algorithm_list))),
                       columns=algorithm_list)  # 每一行是一个instance，每一列是一个算法
    GD = pd.DataFrame(np.zeros((len(instance_list), len(algorithm_list))),
                      columns=algorithm_list)  # 每一行是一个instance，每一列是一个算法
    # ***************************求得最小最大值***************************
    for ins_index, instance_name in enumerate(instance_list):
        ins_dir = project_path+'/ExperimentResult/' + instance_name
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
    IGD.to_csv(project_path+'/ExperimentResult/IGD.csv')

    print('************* GD *************')
    print(GD, '\n\n')
    GD.to_csv(project_path+ '/ExperimentResult/GD.csv')

def get_IGD_GD_no_normalize(instance_list, algorithm_list):
    IGD = pd.DataFrame(np.zeros((len(instance_list), len(algorithm_list))),
                       columns=algorithm_list)  # 每一行是一个instance，每一列是一个算法
    GD = pd.DataFrame(np.zeros((len(instance_list), len(algorithm_list))),
                      columns=algorithm_list)  # 每一行是一个instance，每一列是一个算法

    for ins_index, instance_name in enumerate(instance_list):
        ins_dir = project_path+'/ExperimentResult/' + instance_name
        PF_ref = pd.read_csv(ins_dir+'/referPF.csv').values

        for algorithm_name in algorithm_list:
            alg = os.path.join(ins_dir, algorithm_name)
            PF_know = pd.read_csv(alg + '/algorithmPF.csv').values
            IGD[algorithm_name][ins_index] = getIGDValue(PF_ref, PF_know)
            GD[algorithm_name][ins_index] = getGDValue(PF_ref, PF_know)
    print('************* IGD *************')
    print(IGD, '\n\n')
    # IGD.to_csv('../ExperimentResult/Inverted Generational Distance.csv')

    print('************* GD *************')
    print(GD, '\n\n')
    # GD.to_csv('../ExperimentResult/Generational Distance.csv')


def get_HV_normalize(instance_list, algorithm_list):
    HV = pd.DataFrame(np.zeros((len(instance_list), len(algorithm_list))),
                       columns=algorithm_list)  # 每一行是一个instance，每一列是一个算法

    # ***************************求得最小最大值***************************
    for ins_index, instance_name in enumerate(instance_list):
        ins_dir = project_path+'/ExperimentResult/' + instance_name
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
    HV.to_csv(project_path + '/ExperimentResult/HV.csv')


def compute_hypervolume(ep_objs_batch):
    n = len(ep_objs_batch[0])
    HV = InnerHyperVolume(np.ones(n))
    return HV.compute(ep_objs_batch)


def get_sparsity_normalize(instance_list, algorithm_list):
    sparsity = pd.DataFrame(np.zeros((len(instance_list), len(algorithm_list))),
                       columns=algorithm_list)  # 每一行是一个instance，每一列是一个算法

    # ***************************求得最小最大值***************************
    for ins_index, instance_name in enumerate(instance_list):
        ins_dir = project_path+'/ExperimentResult/' + instance_name
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
    sparsity.to_csv(project_path+'/ExperimentResult/sparsity.csv')



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


def get_AED_AEC_ACOI(instance_list, algorithm_list):
    all_weights = pd.read_csv(project_path+'/instance/all_weights.csv').values  # DataFrame转换成numpy.array

    AED = pd.DataFrame(np.zeros((len(instance_list), len(algorithm_list))), columns=algorithm_list)  # 每一行是一个instance，每一列是一个算法
    AEC = pd.DataFrame(np.zeros((len(instance_list), len(algorithm_list))), columns=algorithm_list)  # 每一行是一个instance，每一列是一个算法
    ACOI = pd.DataFrame(np.zeros((len(instance_list), len(algorithm_list))), columns=algorithm_list)  # 每一行是一个instance，每一列是一个算法

    for ins_index, instance_name in enumerate(instance_list):
        ins_path = project_path+'/ExperimentResult/' + instance_name + '/'
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
            AED.loc[ins_index, algorithm_name] = np.round(np.average(data[:, 0]), 4)
            AEC.loc[ins_index, algorithm_name] = np.round(np.average(data[:, 1]), 4)
            ACOI.loc[ins_index, algorithm_name] = np.round(np.average(data[:, 2]), 4)

    print('---- AED ----\n: ', AED,'\n')
    print('---- AEC ----\n: ', AEC,'\n')
    print('---- ACOI ---\n: ', ACOI,'\n')

    AED.to_csv(project_path+'/ExperimentResult/AED.csv', index=False)
    AEC.to_csv(project_path+'/ExperimentResult/AEC.csv', index=False)
    ACOI.to_csv(project_path+'/ExperimentResult/ACOI.csv', index=False)


def Friedman_test(algorithm_list, metrics_list):
    for metrics in metrics_list:
        # 第7列保存每个算法在所有instance上的average rank
        # 第8列保存每个算法的position
        result = DataFrame(np.zeros((6, 5)), columns=algorithm_list)
        metrics_data = pd.read_csv(project_path+'/ExperimentResult/' + metrics + '.csv')
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
        result.to_csv(project_path+'/ExperimentResult/' + metrics + '_Friedman.csv', index=False)



if __name__ == '__main__':
    # instance_list =  ['[20,3]', '[20,6]','[40,3]','[40,6]','[60,3]','[60,6]']
    instance_list =  ['[20,6]']


    # algorithm_list = ['MOEAD_1','MOEAD_2', 'MOEAD_3']
    algorithm_list = ['MOPSO', 'NSGA2','MOEAD_TCH', 'MOEAD_DVFS', 'MOEAD_3']


    # get_each_algorithm_Pareto_front(instance_list, algorithm_list)
    # get_all_algorithm_referPF(instance_list, algorithm_list)
    # plot_Pareto_front_effectiveness(instance_list, algorithm_list)
    plot_Pareto_front(instance_list, algorithm_list)

    #
    # get_IGD_GD_no_normalize(instance_list, algorithm_list)

    # get_AED_AEC_ACOI(instance_list, algorithm_list)

    # Friedman_test(algorithm_list, ['AED',
    #                                'AEC',
    #                                'ACOI'])
#
