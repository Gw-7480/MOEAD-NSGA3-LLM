import numpy as np
import itertools
import os
import copy
import time
import pandas as pd
import plotly.graph_objs as go
from MODTSRA.NSGA import NSGA2

def generate_reference_points(num_objectives, divisions):
    """生成均匀分布的参考点"""
    def recursive_gen(current, left, depth):
        if depth == num_objectives - 1:
            points.append(current + [left])
        else:
            for i in range(left + 1):
                recursive_gen(current + [i], left - i, depth + 1)
    points = []
    recursive_gen([], divisions, 0)
    ref_points = np.array(points) / divisions
    return ref_points

class NSGA3(NSGA2):
    def __init__(self, instance_name, args, num_objectives=3, divisions=12):
        super().__init__(instance_name, args)
        self.num_objectives = num_objectives
        self.divisions = divisions
        self.reference_points = generate_reference_points(num_objectives, divisions)

    def environmental_selection(self, population, N):
        self.fast_non_dominated_sort(population)
        new_population = []
        i = 1
        while len(new_population) + len(self.F_rank[i]) <= N:
            new_population.extend(self.F_rank[i])
            i += 1
        last_layer = self.F_rank[i]
        K = N - len(new_population)
        selected = self.reference_point_selection(last_layer, K)
        new_population.extend(selected)
        return new_population

    def reference_point_selection(self, last_layer, K):
        objs = np.array([ind.fitness for ind in last_layer])
        min_vals = objs.min(axis=0)
        max_vals = objs.max(axis=0)
        norm_objs = (objs - min_vals) / (max_vals - min_vals + 1e-12)
        associations = []
        for i, obj in enumerate(norm_objs):
            dists = np.linalg.norm(self.reference_points - obj, axis=1)
            associations.append((i, np.argmin(dists), np.min(dists)))
        ref_count = {i: 0 for i in range(len(self.reference_points))}
        for _, ref_idx, _ in associations:
            ref_count[ref_idx] += 1
        selected = []
        used = set()
        while len(selected) < K:
            min_ref = min(ref_count, key=lambda x: ref_count[x])
            candidates = [(i, dist) for i, ref_idx, dist in associations if ref_idx == min_ref and i not in used]
            if not candidates:
                ref_count[min_ref] = float('inf')
                continue
            idx, _ = min(candidates, key=lambda x: x[1])
            selected.append(last_layer[idx])
            used.add(idx)
            ref_count[min_ref] += 1
        return selected

    def run(self):
        self.initializePopulation()
        self.fast_non_dominated_sort(self.P_population)
        self.initializeEP(self.F_rank[1])
        for i in range(1, len(self.F_rank)):
            self.crowding_distance_assignment(self.F_rank[i])
        self.Q_population = self.make_new_population(self.P_population)
        self.update_EP_FromSet(self.EP, self.F_rank[1])
        t = 1
        while t <= self.maxGen:
            self.R_population = self.combine_Pt_and_Qt(self.P_population, self.Q_population)
            self.P_population = self.environmental_selection(self.R_population, self.popSize)
            self.Q_population = self.make_new_population(self.P_population)
            self.update_EP_FromSet(self.EP, self.F_rank[1])
            t += 1
        for ep in self.EP:
            ep.temp_fitness = ep.fitness[0]
        test_fast = sorted(self.EP, key=lambda Individual: Individual.temp_fitness)
        EP_list = [copy.deepcopy(ind.fitness) for ind in test_fast]
        return EP_list

# ================= 主流程入口 =================
if __name__ == '__main__':
    from Utils import get_argument_parser, print_info, project_path

    args = get_argument_parser()
    print_info('************************  NSGA3 ************************')
    print_info(
        '\n---------------------- Test instance:  [' + str(args.Nij) + ',' + str(args.K) + ']------------------------')

    instance_name = '[' + str(args.Nij) + ',' + str(args.K) + ']'
    args.save_dir = os.path.join(args.save_dir, instance_name, 'NSGA3')
    os.makedirs(args.save_dir, exist_ok=True)

    extra = "Running time: {} | popSize: {} | maxGen: {}".format(args.runTime, args.popSize, args.maxGen)
    print_info(extra, '\n')

    alg_dir = os.path.join(project_path, 'ExperimentResult', instance_name, 'NSGA3')
    PF_ref = None  # 可选：你可以像NSGA2一样加载参考帕累托前沿

    for I in range(1, args.runTime + 1):
        run_dir = os.path.join(alg_dir, str(I))
        if not os.path.isdir(run_dir):
            os.makedirs(run_dir)
        startTime = time.time()
        nsgaiii = NSGA3(instance_name, args)
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