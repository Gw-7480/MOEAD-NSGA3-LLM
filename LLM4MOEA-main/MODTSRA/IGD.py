# IGD.py 修正版：解决数据泄漏问题

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def normalize_pf(pf_list, min_vals=None, max_vals=None):
    arr = np.array(pf_list)
    if min_vals is None:
        min_vals = np.min(arr, axis=0)
    if max_vals is None:
        max_vals = np.max(arr, axis=0)
    ranges = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
    norm_arr = (arr - min_vals) / ranges
    return norm_arr.tolist(), min_vals, max_vals


def get_IGD_value(PF_ref, PF_know):
    def get_distance(point1, point2):
        return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

    def d_v_PFSet(v, PFSet):
        return min(get_distance(v, pf) for pf in PFSet)

    distances = [d_v_PFSet(v, PF_know) for v in PF_ref]
    return np.mean(distances)


def read_dat_pf(file_path):
    pf = []
    if not os.path.exists(file_path):
        print(f"[Warning] dat file not found: {file_path}")
        return pf
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                pf.append([float(x) for x in line.strip().split()])
    return pf


def read_pf_csv(csv_path):
    if not os.path.exists(csv_path):
        print(f"[Warning] csv file not found: {csv_path}")
        return []
    df = pd.read_csv(csv_path)
    cols = [c.lower() for c in df.columns]
    col_map = dict(zip(cols, df.columns))
    try:
        pf = df[[col_map['time'], col_map['energy'], col_map['cost']]].values.tolist()
    except Exception as e:
        print(f"[Error] {csv_path} columns error: {e}")
        pf = []
    return pf


def get_ref_pf_excluding_algorithm(project_path, instance_name, dat_path, exclude_algo):
    """为每个算法构建排除自身的参考前沿，避免数据泄漏"""
    moead_pf_path = os.path.join(
        project_path, 'ExperimentResult', instance_name, 'MOEAD_1_multisat1', 'PF.csv'
    )
    nsgaii_pf_path = os.path.join(
        project_path, 'ExperimentResult', instance_name, 'NSGA2_multisat', '1', 'PF.csv'
    )
    nsgaiii_pf_path = os.path.join(
        project_path, 'ExperimentResult', instance_name, 'NSGA3', '1', 'PF.csv'
    )

    moead_pf = read_pf_csv(moead_pf_path) if exclude_algo != 'MOEAD' else []
    nsgaii_pf = read_pf_csv(nsgaii_pf_path) if exclude_algo != 'NSGA2' else []
    nsgaiii_pf = read_pf_csv(nsgaiii_pf_path) if exclude_algo != 'NSGA3' else []
    dat_pf = read_dat_pf(dat_path) if exclude_algo != 'DAT' else []

    PF_ref = moead_pf + nsgaii_pf + nsgaiii_pf + dat_pf
    # 去重
    PF_ref = [list(x) for x in set(tuple(row) for row in PF_ref)]

    included_sources = []
    if exclude_algo != 'MOEAD': included_sources.append(f"MOEAD({len(moead_pf)})")
    if exclude_algo != 'NSGA2': included_sources.append(f"NSGA2({len(nsgaii_pf)})")
    if exclude_algo != 'NSGA3': included_sources.append(f"NSGA3({len(nsgaiii_pf)})")
    if exclude_algo != 'DAT': included_sources.append(f"DAT({len(dat_pf)})")

    print(
        f"[Info] Ref PF for {exclude_algo} (excluding self): {' + '.join(included_sources)} = {len(PF_ref)} unique points")
    return PF_ref if PF_ref else None


def get_ref_pf_excluding_nsga3_and_current(project_path, instance_name, dat_path, current_algo):
    """构建排除NSGA3和当前算法的参考前沿"""
    moead_pf_path = os.path.join(
        project_path, 'ExperimentResult', instance_name, 'MOEAD_1_multisat1', 'PF.csv'
    )
    nsgaii_pf_path = os.path.join(
        project_path, 'ExperimentResult', instance_name, 'NSGA2_multisat', '1', 'PF.csv'
    )

    # 读取数据，但排除NSGA3和当前算法
    moead_pf = read_pf_csv(moead_pf_path) if current_algo != 'MOEAD' else []
    nsgaii_pf = read_pf_csv(nsgaii_pf_path) if current_algo != 'NSGA2' else []
    dat_pf = read_dat_pf(dat_path) if current_algo != 'DAT' else []

    PF_ref = moead_pf + nsgaii_pf + dat_pf
    # 去重
    PF_ref = [list(x) for x in set(tuple(row) for row in PF_ref)]

    included_sources = []
    if current_algo != 'MOEAD': included_sources.append(f"MOEAD({len(moead_pf)})")
    if current_algo != 'NSGA2': included_sources.append(f"NSGA2({len(nsgaii_pf)})")
    included_sources.append(f"DAT({len(dat_pf)})" if current_algo != 'DAT' else "")

    # 清理空字符串
    included_sources = [s for s in included_sources if s]

    print(
        f"[Info] Ref PF for {current_algo} (excluding NSGA3 and self): {' + '.join(included_sources)} = {len(PF_ref)} unique points")
    return PF_ref if PF_ref else None


def get_ref_pf_excluding_nsga3(project_path, instance_name, dat_path):
    """构建排除NSGA3的参考前沿"""
    moead_pf_path = os.path.join(
        project_path, 'ExperimentResult', instance_name, 'MOEAD_1_multisat1', 'PF.csv'
    )
    nsgaii_pf_path = os.path.join(
        project_path, 'ExperimentResult', instance_name, 'NSGA2_multisat', '1', 'PF.csv'
    )

    moead_pf = read_pf_csv(moead_pf_path)
    nsgaii_pf = read_pf_csv(nsgaii_pf_path)
    dat_pf = read_dat_pf(dat_path)

    PF_ref = moead_pf + nsgaii_pf + dat_pf
    # 去重
    PF_ref = [list(x) for x in set(tuple(row) for row in PF_ref)]

    included_sources = []
    included_sources.append(f"MOEAD({len(moead_pf)})")
    included_sources.append(f"NSGA2({len(nsgaii_pf)})")
    included_sources.append(f"DAT({len(dat_pf)})")

    print(f"[Info] Ref PF (excluding NSGA3): {' + '.join(included_sources)} = {len(PF_ref)} unique points")
    return PF_ref if PF_ref else None


def print_corrected_igd(project_path, instance_name, dat_path):
    """计算修正后的IGD值，避免数据泄漏"""
    # 使用正确的路径
    pf_nsgaii_path = os.path.join(project_path, 'ExperimentResult', instance_name, 'NSGA2_multisat', '1', 'PF.csv')
    pf_moead_path = os.path.join(project_path, 'ExperimentResult', instance_name, 'MOEAD_1_multisat1', 'PF.csv')
    nsgaiii_pf_path = os.path.join(project_path, 'ExperimentResult', instance_name, 'NSGA3', '1', 'PF.csv')

    pf_nsgaii = read_pf_csv(pf_nsgaii_path)
    pf_moead = read_pf_csv(pf_moead_path)
    pf_nsgaiii = read_pf_csv(nsgaiii_pf_path)
    pf_dat = read_dat_pf(dat_path)

    # 为每个算法计算排除自身的IGD值
    algorithms = [
        ('NSGA2', pf_nsgaii),
        ('MOEAD', pf_moead),
        ('NSGA3', pf_nsgaiii),
        ('DAT', pf_dat)
    ]

    print("\n=== 修正后的IGD比较（排除数据泄漏）===")
    for algo_name, pf_algo in algorithms:
        if pf_algo:
            # 构建排除当前算法的参考前沿
            ref_pf = get_ref_pf_excluding_algorithm(project_path, instance_name, dat_path, algo_name)
            if ref_pf:
                # 计算IGD
                all_points = np.array(ref_pf + pf_algo)
                min_vals = np.min(all_points, axis=0)
                max_vals = np.max(all_points, axis=0)
                norm_ref, _, _ = normalize_pf(ref_pf, min_vals, max_vals)
                norm_algo, _, _ = normalize_pf(pf_algo, min_vals, max_vals)
                igd_corrected = get_IGD_value(norm_ref, norm_algo)
                print(f"IGD value ({algo_name}, 排除自身): {igd_corrected:.6f}")
            else:
                print(f"{algo_name}: 无法构建参考前沿")
        else:
            print(f"{algo_name}: PF数据不存在")


def print_igd_excluding_nsga3(project_path, instance_name, dat_path):
    """计算排除NSGA3后其他算法的IGD值，只使用帕累托前沿"""
    # 使用正确的帕累托前沿路径
    pf_nsgaii_path = os.path.join(project_path, 'ExperimentResult', instance_name, 'NSGA2_multisat', '1', 'PF.csv')
    pf_moead_path = os.path.join(project_path, 'ExperimentResult', instance_name, 'MOEAD_1_multisat1', 'PF.csv')

    # 读取帕累托前沿数据
    pf_nsgaii = read_pf_csv(pf_nsgaii_path)
    pf_moead = read_pf_csv(pf_moead_path)
    pf_dat = read_dat_pf(dat_path)

    print("\n=== 排除NSGA3后，其他算法的IGD值（使用帕累托前沿）===")

    # 计算各算法的IGD值
    algorithms = [
        ('NSGA2', pf_nsgaii),
        ('MOEAD', pf_moead),
        ('DAT', pf_dat)
    ]

    for algo_name, pf_algo in algorithms:
        if pf_algo:
            # 构建排除NSGA3和当前算法的参考前沿
            current_ref_pf = get_ref_pf_excluding_nsga3_and_current(project_path, instance_name, dat_path, algo_name)

            if current_ref_pf:
                # 计算IGD
                all_points = np.array(current_ref_pf + pf_algo)
                min_vals = np.min(all_points, axis=0)
                max_vals = np.max(all_points, axis=0)
                norm_ref, _, _ = normalize_pf(current_ref_pf, min_vals, max_vals)
                norm_algo, _, _ = normalize_pf(pf_algo, min_vals, max_vals)
                igd_value = get_IGD_value(norm_ref, norm_algo)
                print(f"IGD value ({algo_name}): {igd_value:.6f}")
            else:
                print(f"{algo_name}: 无法构建排除NSGA3和自身的参考前沿")
        else:
            print(f"{algo_name}: PF数据不存在")


if __name__ == "__main__":
    # 路径配置
    project_path = 'D:/post programs/LLM4MOEA-main/LLM4MOEA-main/MODTSRA'
    instance_name = "[20,6]"
    dat_path = "D:/post programs/LLM4MOEA-main/LLM4MOEA-main/MOEAD-LLM/RE21/results0/RE21_d4_PF_opt.dat"

    print("=== 验证数据泄漏问题（使用正确路径）===")
    print_corrected_igd(project_path, instance_name, dat_path)

    print("\n" + "=" * 60)
    print_igd_excluding_nsga3(project_path, instance_name, dat_path)

    print("\n=== 说明 ===")
    print("使用帕累托前沿的IGD值计算:")
    print("- NSGA2: NSGA2_multisat/1/PF.csv")
    print("- MOEAD: MOEAD_1_multisat1/PF.csv")
    print("- NSGA3: NSGA3/1/PF.csv (在第二部分计算中被完全排除)")
    print("- DAT: RE21_d4_PF_opt.dat")
    print("第二部分显示排除NSGA3后，其他算法使用帕累托前沿的IGD值，参考前沿不包含NSGA3")