# 导入自定义和pymoo库中的多目标优化算法
from pymoo.algorithms.moo.moead_LLM import MOEAD_LLM  # 集成大模型的MOEAD算法
from pymoo.algorithms.moo.nsga3_LLM import NSGA3_LLM  # 集成大模型的NSGA3算法
from pymoo.algorithms.moo.moead import MOEAD  # 标准MOEAD算法
from pymoo.algorithms.moo.nsga2 import NSGA2  # NSGA-II算法
from pymoo.algorithms.moo.nsga3 import NSGA3  # NSGA-III算法
from pymoo.optimize import minimize  # pymoo的优化主入口
from pymoo.problems import get_problem  # 获取标准测试问题
from problem_interface import RE  # 导入RE系列问题接口
from problem_interface import UF1, UF2, UF3, UF5, UF7, UF8  # 导入UF系列问题
from pymoo.util.ref_dirs import get_reference_directions  # 生成参考方向
from output import output_nsga3_llm, create_combined_analysis  # 结果输出函数
from MODTSRA.fitness import get_argument_parser
from MODTSRA.Utils import *
import os
from pymoo.operators.sampling.rnd import CustomRandomSampling
from pymoo.vendor.cec2018 import false
import numpy as np
import pandas as pd

# 根据问题名称返回对应的问题对象
def get_problem(problemname, dimension=None, args=None):
    if problemname == "UF1":
        problem = UF1(D=dimension)
    elif problemname == "UF2":
        problem = UF2(D=dimension)
    elif problemname == "UF3":
        problem = UF3(D=dimension)
    elif problemname == "UF5":
        problem = UF5(D=dimension)
    elif problemname == "UF7":
        problem = UF7(D=dimension)
    elif problemname == "UF8":
        problem = UF8(D=dimension)
    elif problemname == "RE21":  # 也支持RE系列问题
        problem = RE(0)
    elif problemname == "RE22":
        problem = RE(1)
    elif problemname == "RE23":
        problem = RE(2)
    elif problemname == "RE24":
        problem = RE(3)
    elif problemname == "RE25":
        problem = RE(4)
    elif problemname == "RE31":
        problem = RE(5)
    elif problemname == "RE32":
        problem = RE(6)
    else:
        # 其他问题用pymoo自带的get_problem
        problem = get_problem(problemname, n_var=dimension)
    return problem


# 根据算法名称返回对应的算法对象
def get_algorithm(algorithmname):
    if algorithmname == "MOEAD_LLM":
        algorithm = MOEAD_LLM(
            ref_dirs,
            n_neighbors=neighbor_size,
            prob_neighbor_mating=0.7,
            debug_mode=debug_mode,
            model_LLM=model_LLM,
            endpoint=endpoint,
            key=key,
            out_file=out_filename_gpt,
            sampling=CustomRandomSampling(),  # 使用自定义采样
        )
    elif algorithmname == "NSGA3_LLM":
        algorithm = NSGA3_LLM(
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            debug_mode=debug_mode,
            model_LLM=model_LLM,
            endpoint=endpoint,
            key=key,
            out_file=out_filename_gpt,
            sampling=CustomRandomSampling(),  # 使用自定义采样
        )
    elif algorithmname == "MOEAD":
        algorithm = MOEAD(
            ref_dirs,
            n_neighbors=neighbor_size,
            prob_neighbor_mating=0.7,
        )
    elif algorithmname == "NSGAII":
        algorithm = NSGA2(pop_size=pop_size)
    return algorithm


if __name__ == '__main__':
    args = get_argument_parser()
    K = args.K
    instance_name = '[' + str(args.Nij) + ',' + str(args.K) + ']'
    algorithmname = "NSGA3_LLM"  # NSGA-II, MOEAD, MOEAD_LLM
    record_gpt_solution = False  # record the input and oupt of each run of gpt to learn a general linear operator
    model_LLM = "deepseek-chat"  # LLM模型名
    endpoint = "api.deepseek.com"  # LLM API endpoint
    key = "sk-1f3cc5f02f1548c6b53f3634bd5770ca"  # your key

    debug_mode = False  # 修正: false -> False

    pop_size = 100
    neighbor_size = 10
    n_gen = 20
    n_partition = 10  # for three objective only

    problems = ['RE21']

    n_repeat = 3

    # 创建主输出目录
    main_output_dir = f"results_{algorithmname}_{instance_name}"
    os.makedirs(main_output_dir, exist_ok=True)

    for prob in problems:
        print(f"\n开始处理问题: {prob}")
        problemname = prob

        # 为每个问题创建单独的结果目录
        problem_output_dir = os.path.join(main_output_dir, problemname)
        os.makedirs(problem_output_dir, exist_ok=True)

        for n in range(n_repeat):
            print(f"\n=== 运行 {n + 1}/{n_repeat} ===")

            dimension = 4

            # 修改输出路径结构，使其更清晰
            outputfile = os.path.join(problem_output_dir, f"run_{n}")
            os.makedirs(outputfile, exist_ok=True)

            print(f"当前运行输出目录: {outputfile}")

            if record_gpt_solution:
                out_filename_gpt = os.path.join(outputfile, f"{problemname}_d{dimension}_gpt_sample.dat")
                with open(out_filename_gpt, "w") as file:
                    pass  # 创建空文件
            else:
                out_filename_gpt = None

            # 统一为三目标权重向量
            ref_dirs = get_reference_directions("uniform", 3, n_partitions=n_partition)

            # 替换为自定义问题模型
            from MODTSRA.fitness import MyProblem
            from MODTSRA.Utils import N_n, N_service

            Nij = args.Nij
            len_order = N_n * Nij
            len_location = N_n * Nij
            len_A = N_n
            n_var = len_order + len_location + len_A

            # 关键：修改变量上下界
            xl = np.concatenate([
                np.zeros(len_order),  # order部分下界：0
                np.ones(len_location),  # location部分下界：1
                np.zeros(len_A)  # A部分下界：0
            ])
            xu = np.concatenate([
                (Nij - 1) * np.ones(len_order),  # order部分上界：Nij-1
                (K + 1) * np.ones(len_location),  # location部分上界：K+1
                np.ones(len_A)  # A部分上界：1
            ])
            problem = MyProblem(
                n_var=n_var,
                n_obj=3,
                xl=xl,
                xu=xu,
                instance_name=instance_name,
                args=args
            )

            algorithm = get_algorithm(algorithmname)

            print("开始优化...")
            res = minimize(problem,
                           algorithm,
                           ('n_gen', n_gen),
                           seed=2023 * n,
                           save_history=True,
                           verbose=True)

            print("优化完成，开始保存结果...")

            # 使用改进的输出函数，传入运行ID
            try:
                output_dir = output_nsga3_llm(res, algorithm, problemname, dimension,
                                              outputfile + "/", run_id=n)
                print(f"运行 {n + 1} 结果保存成功!")
            except Exception as e:
                print(f"保存结果时出错: {e}")
                # 即使出错也要保存基本的PF数据
                try:
                    basic_pf_file = os.path.join(outputfile, f"basic_PF_run{n}.csv")
                    pd.DataFrame(res.F, columns=[f'obj_{i}' for i in range(res.F.shape[1])]).to_csv(
                        basic_pf_file, index=False)
                    print(f"至少保存了基本PF数据到: {basic_pf_file}")
                except:
                    print("连基本数据都无法保存!")

        # 所有运行完成后，创建综合分析
        print(f"\n为问题 {problemname} 创建综合分析...")
        try:
            combined_dir = create_combined_analysis(problem_output_dir, problemname, n_repeat)
            print(f"综合分析完成: {combined_dir}")
        except Exception as e:
            print(f"创建综合分析时出错: {e}")

    print(f"\n所有实验完成！结果保存在: {main_output_dir}")