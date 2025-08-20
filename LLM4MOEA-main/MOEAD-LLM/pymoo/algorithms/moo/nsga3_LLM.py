import numpy as np
import warnings
import os
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.selection import Selection
from pymoo.operators.crossover.gpt import GPT_interface
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection, compare
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.docs import parse_doc_string


def comp_by_cv_then_random(pop, P, **kwargs):
    """
    Tournament selection comparing by constraint violation first, then randomly.
    This function is copied from nsga3.py since it's not available elsewhere.
    """
    if P.shape[1] != 2:
        raise Exception("Only implemented for binary tournament!")

    tournament_type = kwargs.get('tournament_type', 'comp_by_cv_then_random')
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]
        a_cv, b_cv = pop[a].CV[0], pop[b].CV[0]

        # if one is infeasible and the other not
        if a_cv > 0 and b_cv == 0:
            S[i] = b
        elif a_cv == 0 and b_cv > 0:
            S[i] = a
        elif a_cv > 0 and b_cv > 0:
            S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True)
        else:
            S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True)

    return S[:, None].astype(int, copy=False)


class NSGA3_LLM(NSGA3):

    def __init__(self,
                 ref_dirs,
                 pop_size=None,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=comp_by_cv_then_random),
                 crossover=GPT_interface(),  # 使用LLM交叉算子
                 mutation=PM(eta=20),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 output=MultiObjectiveOutput(),
                 debug_mode=False,  # LLM调试模式
                 model_LLM=None,  # LLM模型名
                 endpoint=None,  # LLM API端点
                 key=None,  # LLM API密钥
                 out_file=None,  # LLM日志文件
                 **kwargs):
        """
        NSGA-III with Large Language Model integration

        Parameters
        ----------
        ref_dirs : numpy.ndarray
            Reference directions for the algorithm
        pop_size : int, optional
            Population size. If None, equals number of reference directions
        sampling : Sampling, optional
            Sampling strategy for initialization
        selection : Selection, optional
            Selection operator for parent selection
        crossover : Crossover, optional
            Crossover operator, defaults to GPT_interface for LLM integration
        mutation : Mutation, optional
            Mutation operator
        eliminate_duplicates : bool, optional
            Whether to eliminate duplicates
        n_offsprings : int, optional
            Number of offsprings per generation
        output : Output, optional
            Output configuration
        debug_mode : bool, optional
            Enable debugging mode for LLM operations
        model_LLM : str, optional
            LLM model name (e.g., "deepseek-chat", "gpt-4")
        endpoint : str, optional
            LLM API endpoint
        key : str, optional
            LLM API key
        out_file : str, optional
            Output file path for LLM interaction logs
        """

        # Store LLM-specific parameters
        self.debug_mode = debug_mode
        self.model_LLM = model_LLM
        self.endpoint = endpoint
        self.key = key
        self.out_file = out_file

        # Initialize parent NSGA3 with LLM crossover operator
        super().__init__(
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=eliminate_duplicates,
            n_offsprings=n_offsprings,
            output=output,
            **kwargs
        )

    def _infill(self):
        """
        Override the infill method to pass LLM parameters to the crossover operator
        """
        # 添加调试信息头
        print("\n=== DEBUG INFO START ===")
        print(f"Current population size: {len(self.pop)}")
        print(f"n_offsprings: {self.n_offsprings}")

        # 获取基础参数
        n_individuals = self.n_offsprings if self.n_offsprings is not None else self.pop_size
        n_parents = self.mating.crossover.n_parents

        # 打印选择参数
        print(f"\n[Selection Phase]")
        print(f"n_individuals: {n_individuals}, n_parents: {n_parents}")

        # 手动进行selection获取父代索引
        parents = self.mating.selection.do(self.problem, self.pop, n_individuals, n_parents, to_pop=False)
        print(f"Parents matrix shape: {parents.shape}")
        print("Sample parent indices:", parents[:2])

        # 限制处理的个体数量，避免索引越界
        max_individuals = min(10, len(parents))
        print(f"\n[LLM Processing Limitation]")
        print(f"Max individuals to process: {max_individuals}")

        # 准备LLM需要的Y参数
        Y = []
        print("\n[Y Parameter Construction]")
        for i in range(max_individuals):
            parent_group = []
            for j in range(n_parents):
                parent_idx = int(parents[i, j])
                obj_values = self.pop[parent_idx].F
                parent_group.append([np.sum(obj_values)])
            Y.append(parent_group)
            if i < 2:
                print(f"Y[{i}]: {parent_group}")

                # 准备parents_obj参数
        from pymoo.core.population import Population
        selected_individuals = []
        print("\n[Parent Objects Selection]")
        for i in range(max_individuals):
            for j in range(n_parents):
                parent_idx = int(parents[i, j])
                selected_individuals.append(self.pop[parent_idx])
                if i < 2 and j < n_parents:
                    print(f"Parent[{i},{j}] X: {self.pop[parent_idx].X[:5]}... F: {self.pop[parent_idx].F}")

        parents_obj = [Population.create(*selected_individuals)]
        print(f"\n[Parent Objects Population]")
        print(f"Created population size: {len(parents_obj[0])}")

        # 准备crossover的pop参数
        parents_pop = []
        print("\n[Parent Groups Construction]")
        for i in range(max_individuals):
            parent_group = []
            for j in range(n_parents):
                parent_idx = int(parents[i, j])
                parent_group.append(self.pop[parent_idx])
            parents_pop.append(parent_group)
            if i < 2:
                print(f"Parent group {i}: Indices {parents[i]}")

                # 交叉操作前打印关键参数
        print("\n[Before Crossover]")
        print(f"parents_pop length: {len(parents_pop)}")
        print(f"Y length: {len(Y)}")
        print(f"parents_obj type: {type(parents_obj)}")
        print(f"parents_obj[0] size: {len(parents_obj[0]) if parents_obj else 0}")

        # ===== 关键修改：临时调整 n_offsprings =====
        original_n_offsprings = self.n_offsprings
        original_crossover_n_offsprings = self.mating.crossover.n_offsprings

        # 临时设置为实际处理的个体数量
        self.n_offsprings = max_individuals
        self.mating.crossover.n_offsprings = 2  # GPT交叉算子每次产生2个子代

        print(f"\n[Temporary Settings]")
        print(f"Temporary n_offsprings: {self.n_offsprings}")
        print(f"Temporary crossover n_offsprings: {self.mating.crossover.n_offsprings}")

        # 直接调用crossover
        print("\n[Crossover Call]")
        offspring_list = []

        # 逐个处理每个父代组合
        for i in range(max_individuals):
            try:
                # 为每个父代组合单独调用交叉算子
                single_offspring = self.mating.crossover.do(
                    problem=self.problem,
                    pop=[parents_pop[i]],  # 单个父代组合
                    Y=[Y[i]],  # 对应的Y值
                    debug_mode=self.debug_mode,
                    model_LLM=self.model_LLM,
                    endpoint=self.endpoint,
                    key=self.key,
                    out_filename=self.out_file,
                    parents_obj=parents_obj
                )

                # 从返回的子代中选择一个
                if isinstance(single_offspring, Population) and len(single_offspring) > 0:
                    offspring_list.append(single_offspring[0])
                elif isinstance(single_offspring, np.ndarray) and single_offspring.shape[0] > 0:
                    # 如果返回的是数组，转换为Population
                    from pymoo.core.population import Individual
                    ind = Individual(X=single_offspring[0])
                    offspring_list.append(ind)

            except Exception as e:
                print(f"Error in crossover for individual {i}: {e}")
                # 使用父代作为备选
                offspring_list.append(parents_pop[i][0])

                # 恢复原始设置
        self.n_offsprings = original_n_offsprings
        self.mating.crossover.n_offsprings = original_crossover_n_offsprings

        # 创建最终的子代种群
        if offspring_list:
            offspring = Population.create(*offspring_list)
        else:
            # 如果没有生成任何子代，使用父代
            offspring = Population.create(*[parents_pop[i][0] for i in range(max_individuals)])

            # 交叉后立即打印输出
        print("\n[After Crossover]")
        print(f"Offspring type: {type(offspring)}")
        if isinstance(offspring, Population):
            print(f"Offspring size: {len(offspring)}")
            if len(offspring) > 0:
                print(f"Sample offspring X shape: {offspring[0].X.shape}")

                # 应用mutation
        print("\n[Mutation Phase]")
        offspring = self.mating.mutation.do(self.problem, offspring)

        # 最终输出检查
        print("\n[Final Offspring Check]")
        if isinstance(offspring, Population):
            print(f"Final offspring count: {len(offspring)}")
            if len(offspring) > 0:
                print(f"First offspring X: {offspring[0].X[:5]}...")

        print("=== DEBUG INFO END ===\n")

        return offspring
        # 交叉后立即打印输出
        print("\n[After Crossover]")
        print(f"Offspring type: {type(offspring)}")
        if isinstance(offspring, Population):
            print(f"Offspring size: {len(offspring)}")
            if len(offspring) > 0:
                print(f"Sample offspring X shape: {offspring[0].X.shape}")
        elif isinstance(offspring, np.ndarray):
            print(f"Offspring array shape: {offspring.shape}")
        else:
            print("Unknown offspring type")

        # 应用mutation
        print("\n[Mutation Phase]")
        offspring = self.mating.mutation.do(self.problem, offspring)

        # 最终输出检查
        print("\n[Final Offspring Check]")
        if isinstance(offspring, Population):
            print(f"Final offspring count: {len(offspring)}")
            if len(offspring) > 0:
                print(f"First offspring X: {offspring[0].X[:5]}...")
        else:
            print("Offspring is not Population instance")

        print("=== DEBUG INFO END ===\n")

        # 添加输出跟踪
        if not hasattr(self, 'generation_data'):
            self.generation_data = []
            self.llm_stats = {'calls': 0, 'successes': 0, 'failures': 0}

            # 在交叉操作后记录数据
        generation_info = {
            'generation': getattr(self, 'n_gen', 0),
            'offspring_count': len(offspring) if isinstance(offspring, Population) else 0,
            'llm_calls': max_individuals,
            'successful_crossovers': len(offspring_list) if 'offspring_list' in locals() else 0
        }
        self.generation_data.append(generation_info)

        return offspring

    def save_generation_data(self, output_dir):
        """保存每代的详细数据"""
        import pandas as pd

        # 保存每代统计信息
        df = pd.DataFrame(self.generation_data)
        df.to_csv(os.path.join(output_dir, 'generation_stats.csv'), index=False)

        # 保存参考方向信息
        if hasattr(self, 'ref_dirs'):
            ref_df = pd.DataFrame(self.ref_dirs, columns=[f'obj_{i}' for i in range(self.ref_dirs.shape[1])])
            ref_df.to_csv(os.path.join(output_dir, 'reference_directions.csv'), index=False)

    def _set_optimum(self, **kwargs):
        """
        Set the optimum solution (Pareto front approximation)
        """
        # Use the survival operator's optimum if available (first front + reference direction based)
        if hasattr(self.survival, 'opt') and len(self.survival.opt) > 0:
            self.opt = self.survival.opt
        else:
            # Fallback to first front individuals
            try:
                from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
                fronts = NonDominatedSorting().do(self.pop.get("F"))
                if len(fronts) > 0 and len(fronts[0]) > 0:
                    self.opt = self.pop[fronts[0]]
                else:
                    self.opt = self.pop
            except:
                self.opt = self.pop


# Parse documentation string for the constructor
parse_doc_string(NSGA3_LLM.__init__)