from pymoo.visualization.scatter import Scatter
from pymoo.util import plotting
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
import numpy as np
import os


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