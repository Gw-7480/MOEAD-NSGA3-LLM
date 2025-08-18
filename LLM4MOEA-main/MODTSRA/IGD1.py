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

def merge_and_dedup_pfs(*pf_lists):
    merged = []
    for pf in pf_lists:
        merged.extend(pf)
    # 去重
    merged = [list(x) for x in set(tuple(row) for row in merged)]
    return merged

if __name__ == "__main__":
    # 路径配置
    project_path = 'D:/post programs/LLM4MOEA-main/LLM4MOEA-main/MODTSRA'
    instance_name = "[20,6]"
    dat_path = "D:/post programs/LLM4MOEA-main/LLM4MOEA-main/MOEAD-LLM/RE21/results0/RE21_d4_PF_opt.dat"

    # 四个PF路径
    pf_nsgaii_path = os.path.join(project_path, 'ExperimentResult', instance_name, 'NSGA2_multisat', '1', 'PF.csv')
    pf_moead_path = os.path.join(project_path, 'ExperimentResult', instance_name, 'MOEAD_1_multisat1',  'PF.csv')
    pf_nsgaiii_path = os.path.join(project_path, 'ExperimentResult', instance_name, 'NSGA3', '1', 'PF.csv')
    pf_dat_path = dat_path

    # 读取四个PF
    pf_nsgaii = read_pf_csv(pf_nsgaii_path)
    pf_moead = read_pf_csv(pf_moead_path)
    pf_nsgaiii = read_pf_csv(pf_nsgaiii_path)
    pf_dat = read_dat_pf(pf_dat_path)

    # 合并去重为新参考前沿
    PF_ref = merge_and_dedup_pfs(pf_nsgaii, pf_moead, pf_nsgaiii, pf_dat)
    if not PF_ref:
        print("[Error] Reference PF is empty, check PF.csv/dat files.")
        exit(1)

    # 归一化参数
    all_points = np.array(PF_ref)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    norm_pf_ref, _, _ = normalize_pf(PF_ref, min_vals, max_vals)

    # 计算每个PF的IGD
    results = []
    pf_dict = {
        "NSGA-II": pf_nsgaii,
        "MOEA/D": pf_moead,
        "NSGA-III": pf_nsgaiii,
        "DAT": pf_dat
    }
    for name, pf in pf_dict.items():
        if pf:
            norm_pf, _, _ = normalize_pf(pf, min_vals, max_vals)
            igd_val = get_IGD_value(norm_pf_ref, norm_pf)
            results.append((name, igd_val, len(pf)))
        else:
            results.append((name, None, 0))

    # 输出结果
    print("=== IGD values for each PF (vs merged PF_ref) ===")
    for name, igd_val, pf_size in results:
        if igd_val is not None:
            print(f"{name}: IGD = {igd_val:.6f} (PF size: {pf_size})")
        else:
            print(f"{name}: PF not found or empty.")

    # 可选：保存结果到文件
    # with open(os.path.join(project_path, 'ExperimentResult', instance_name, 'IGD_summary.txt'), 'w') as f:
    #     for name, igd_val, pf_size in results:
    #         if igd_val is not None:
    #             f.write(f"{name}: IGD = {igd_val:.6f} (PF size: {pf_size})\n")
    #         else:
    #             f.write(f"{name}: PF not found or empty.\n")