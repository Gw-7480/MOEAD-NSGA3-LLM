from pymoo.visualization.scatter import Scatter
from pymoo.util import plotting
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
import numpy as np
import pandas as pd
import os
import json


# 输出优化结果到文件
def output(res, problemname, dimension, outputfile):
    # 1. 确保输出目录存在（若不存在则自动创建）
    os.makedirs(outputfile, exist_ok=True)

    # 2. 获取最优解的决策变量X和目标值F
    X, F = res.opt.get("X", "F")

    # 3. 获取算法历史记录（每一代的算法状态）
    hist = res.history

    n_evals = []  # 存储每一代的函数评估次数
    hist_F = []  # 存储每一代的目标空间值

    # 4. 遍历每一代的算法状态
    for algo in hist:
        # 记录当前代的函数评估次数
        n_evals.append(algo.evaluator.n_eval)
        # 获取当前代的最优解（可行解）目标值
        opt = algo.opt
        hist_F.append(opt.get("F"))

    # 5. 计算超体积(Hypervolume, HV)收敛曲线
    approx_ideal = F.min(axis=0)  # 近似理想点
    approx_nadir = F.max(axis=0)  # 近似纳迪尔点
    # 根据目标数设置参考点
    if len(approx_nadir) == 3:
        ref_point = np.array([1.1, 1.1, 1.1])
    else:
        ref_point = np.array([1.1, 1.1])
    # 创建超体积评价指标对象
    metric = Hypervolume(ref_point=ref_point,
                         norm_ref_point=False,
                         zero_to_one=True,
                         ideal=approx_ideal,
                         nadir=approx_nadir)
    # 计算每一代的超体积值
    hv = [metric.do(_F) for _F in hist_F]
    # 保存超体积收敛曲线到文件
    filename = outputfile + problemname + "_d" + str(dimension) + "_hv.dat"
    file_hv = open(filename, "w")
    for i in range(len(n_evals)):
        file_hv.write("{} {:.4f} \n".format(n_evals[i], hv[i]))
    file_hv.close()

    # 6. IGD收敛曲线（已注释，若需要可启用）
    # pf = problem.pareto_front(use_cache=False)
    # metric = IGD(pf, zero_to_one=True)
    # igd = [metric.do(_F) for _F in hist_F]
    # filename = outputfile + problemname + "_d" + str(dimension) + "_igd.dat"
    # file_igd = open(filename, "w")
    # for i in range(len(n_evals)):
    #     file_igd.write("{} {:.4f} \n".format(n_evals[i], igd[i]))
    # file_igd.close()

    # 7. 保存最终的Pareto前沿（PF）数据
    filename = outputfile + problemname + "_d" + str(dimension) + "_PF_opt.dat"
    file_pf = open(filename, "w")
    pf_list = res.F
    for i in range(len(pf_list)):
        for j in range(len(pf_list[i])):
            file_pf.write("{:.4f} ".format(pf_list[i][j]))
        file_pf.write("\n")
    file_pf.close()

    # 8. 保存最终的Pareto解集（PS）数据
    filename = outputfile + problemname + "_d" + str(dimension) + "_PS_opt.dat"
    file_ps = open(filename, "w")
    ps_list = res.X
    for i in range(len(ps_list)):
        for j in range(len(ps_list[i])):
            file_ps.write("{:.4f} ".format(ps_list[i][j]))
        file_ps.write("\n")
    file_ps.close()

    # 9. 绘制并保存PF散点图
    Scatter().add(res.F).save(outputfile + problemname + "_d" + str(dimension))


def output_nsga3_llm(res, algorithm, problemname, dimension, outputfile, run_id=0):
    """NSGA3_LLM 特定的输出函数 - 修改版"""

    print(f"开始保存NSGA3_LLM结果到: {outputfile}")

    # 调用标准输出
    output(res, problemname, dimension, outputfile)

    # 创建 NSGA3_LLM 特定输出目录
    nsga3_dir = os.path.join(outputfile, "NSGA3_LLM_analysis")
    os.makedirs(nsga3_dir, exist_ok=True)
    print(f"创建NSGA3_LLM分析目录: {nsga3_dir}")

    # 1. 保存参考方向数据
    if hasattr(algorithm, 'ref_dirs') and algorithm.ref_dirs is not None:
        ref_dirs_file = os.path.join(nsga3_dir, f"reference_directions_run{run_id}.csv")
        pd.DataFrame(algorithm.ref_dirs,
                     columns=[f'obj_{i}' for i in range(algorithm.ref_dirs.shape[1])]).to_csv(
            ref_dirs_file, index=False)
        print(f"保存参考方向到: {ref_dirs_file}")

    # 2. 保存每代详细统计数据
    if hasattr(algorithm, 'generation_data') and len(algorithm.generation_data) > 0:
        gen_stats_file = os.path.join(nsga3_dir, f"generation_statistics_run{run_id}.csv")
        pd.DataFrame(algorithm.generation_data).to_csv(gen_stats_file, index=False)
        print(f"保存每代统计到: {gen_stats_file}")

    # 3. 保存Pareto前沿的详细信息（CSV格式，便于分析）
    pf_detailed_file = os.path.join(nsga3_dir, f"pareto_front_detailed_run{run_id}.csv")
    pf_df = pd.DataFrame(res.F, columns=[f'obj_{i}' for i in range(res.F.shape[1])])

    # 如果有决策变量信息，也一起保存
    if hasattr(res, 'X') and res.X is not None:
        x_df = pd.DataFrame(res.X, columns=[f'var_{i}' for i in range(res.X.shape[1])])
        pf_detailed_df = pd.concat([x_df, pf_df], axis=1)
    else:
        pf_detailed_df = pf_df

    pf_detailed_df.to_csv(pf_detailed_file, index=False)
    print(f"保存详细Pareto前沿到: {pf_detailed_file}")

    # 4. 保存算法执行摘要
    summary_info = {
        'algorithm': 'NSGA3_LLM',
        'problem': problemname,
        'dimension': dimension,
        'run_id': run_id,
        'final_population_size': len(res.F),
        'n_objectives': res.F.shape[1],
        'n_variables': res.X.shape[1] if hasattr(res, 'X') and res.X is not None else 'Unknown',
        'hypervolume_final': None,  # 可以计算最终的HV值
        'ref_dirs_count': len(algorithm.ref_dirs) if hasattr(algorithm, 'ref_dirs') else 0,
        'llm_enabled': hasattr(algorithm, 'model_LLM') and algorithm.model_LLM is not None,
        'model_name': getattr(algorithm, 'model_LLM', 'None')
    }

    # 计算最终超体积值
    try:
        if res.F.shape[1] == 3:
            ref_point = np.array([1.1, 1.1, 1.1])
        else:
            ref_point = np.array([1.1, 1.1])
        metric = Hypervolume(ref_point=ref_point)
        summary_info['hypervolume_final'] = float(metric.do(res.F))
    except Exception as e:
        print(f"计算超体积时出错: {e}")
        summary_info['hypervolume_final'] = None

    summary_file = os.path.join(nsga3_dir, f"algorithm_summary_run{run_id}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_info, f, indent=2, ensure_ascii=False)
    print(f"保存算法摘要到: {summary_file}")

    # 5. 如果有LLM交叉操作的日志，也保存相关统计
    if hasattr(algorithm, 'crossover') and hasattr(algorithm.crossover, 'llm_call_count'):
        llm_stats = {
            'total_llm_calls': getattr(algorithm.crossover, 'llm_call_count', 0),
            'successful_calls': getattr(algorithm.crossover, 'successful_calls', 0),
            'failed_calls': getattr(algorithm.crossover, 'failed_calls', 0),
            'average_response_time': getattr(algorithm.crossover, 'avg_response_time', 0)
        }
        llm_stats_file = os.path.join(nsga3_dir, f"llm_crossover_stats_run{run_id}.json")
        with open(llm_stats_file, 'w', encoding='utf-8') as f:
            json.dump(llm_stats, f, indent=2)
        print(f"保存LLM统计到: {llm_stats_file}")

    print(f"NSGA3_LLM结果保存完成!")
    return nsga3_dir


def create_combined_analysis(output_base_dir, problem_name, n_runs):
    """创建多次运行的综合分析报告"""

    combined_dir = os.path.join(output_base_dir, "combined_analysis")
    os.makedirs(combined_dir, exist_ok=True)

    all_pf_data = []
    all_summaries = []

    # 收集所有运行的数据
    for i in range(n_runs):
        run_dir = os.path.join(f"{problem_name}/results{i}", "NSGA3_LLM_analysis")

        # 收集PF数据
        pf_file = os.path.join(run_dir, f"pareto_front_detailed_run{i}.csv")
        if os.path.exists(pf_file):
            pf_data = pd.read_csv(pf_file)
            pf_data['run_id'] = i
            all_pf_data.append(pf_data)

        # 收集摘要数据
        summary_file = os.path.join(run_dir, f"algorithm_summary_run{i}.json")
        if os.path.exists(summary_file):
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
                all_summaries.append(summary)

    # 保存合并的PF数据
    if all_pf_data:
        combined_pf = pd.concat(all_pf_data, ignore_index=True)
        combined_pf.to_csv(os.path.join(combined_dir, "all_runs_pareto_fronts.csv"), index=False)

    # 保存合并的统计摘要
    if all_summaries:
        combined_summary_df = pd.DataFrame(all_summaries)
        combined_summary_df.to_csv(os.path.join(combined_dir, "all_runs_summary.csv"), index=False)

        # 计算统计指标
        if 'hypervolume_final' in combined_summary_df.columns:
            hv_stats = {
                'mean_hv': float(combined_summary_df['hypervolume_final'].mean()),
                'std_hv': float(combined_summary_df['hypervolume_final'].std()),
                'min_hv': float(combined_summary_df['hypervolume_final'].min()),
                'max_hv': float(combined_summary_df['hypervolume_final'].max())
            }

            with open(os.path.join(combined_dir, "hypervolume_statistics.json"), 'w') as f:
                json.dump(hv_stats, f, indent=2)

    print(f"综合分析报告保存到: {combined_dir}")
    return combined_dir