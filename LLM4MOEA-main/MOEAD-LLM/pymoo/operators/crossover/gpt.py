import numpy as np  # 导入NumPy库，用于数值计算

from pymoo.core.crossover import Crossover  # 从pymoo库导入Crossover基类
from pymoo.util.misc import crossover_mask  # 导入crossover_mask工具函数（本文件未用到）
import json  # 导入json库，用于处理JSON数据
import http.client  # 导入http.client库，用于HTTP请求
import re  # 导入正则表达式库，用于字符串匹配
import time  # 导入time库，用于延时操作
import os  # 导入os库，用于文件路径操作
import random  # 导入随机数库
# 导入N_n和Nij
from MODTSRA.Utils import N_n, get_argument_parser, N_service

class GPT(Crossover):  # 定义GPT类，继承自pymoo的Crossover基类

    def __init__(self, n_new, **kwargs):
        super().__init__(10, 2, **kwargs)  # 初始化父类，假设10为子代数，2为父代数
        self.n_new = n_new  # 记录新生成个体的数量

        # 加载DAG信息用于生成正确的拓扑排序
        args = get_argument_parser()
        self.Nij = args.Nij
        self.K = args.K
        self.instance_name = '[' + str(args.Nij) + ',' + str(args.K) + ']'
        self.dag_dependencies = self._load_dag_dependencies()

    def _load_dag_dependencies(self):
        """加载DAG依赖关系"""
        dag_info = {}
        try:
            # 修正路径计算
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            # 从 MOEAD-LLM/pymoo/operators/crossover/ 到项目根目录 LLM4MOEA-main
            # 需要回退4级：../../../..
            project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..', '..', '..'))
            instance_path = os.path.join(project_root, 'MODTSRA', 'instance', self.instance_name, 'DAG')

            print(f"GPT: Looking for DAG files in: {instance_path}")  # 调试信息

            for uav_id in range(1, N_n + 1):
                dag_file = os.path.join(instance_path, f'{uav_id}.txt')
                if os.path.exists(dag_file):
                    dependencies = {}
                    with open(dag_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if ':' in line:
                                parts = line.split(':')
                                if len(parts) == 3:
                                    predecessor = parts[0] if parts[0] else None
                                    task_id = int(parts[1])
                                    successors = [int(x) for x in parts[2].split(',') if x]

                                    dependencies[task_id] = {
                                        'predecessors': [int(x) for x in predecessor.split(',') if
                                                         x] if predecessor else [],
                                        'successors': successors
                                    }
                    dag_info[uav_id - 1] = dependencies  # 使用0开始的索引
                else:
                    print(f"Warning: DAG file not found: {dag_file}")
        except Exception as e:
            print(f"Error loading DAG dependencies: {e}")

        return dag_info

    def _generate_valid_topological_order(self, uav_index):
        """为指定UAV生成有效的拓扑排序"""
        if uav_index not in self.dag_dependencies:
            # 如果没有DAG信息，返回简单的顺序排列
            return list(range(self.Nij))

        dependencies = self.dag_dependencies[uav_index]

        # 实现拓扑排序算法
        in_degree = {}
        for task_id in range(self.Nij):
            in_degree[task_id] = 0

        # 计算入度
        for task_id, info in dependencies.items():
            for pred in info['predecessors']:
                if pred in in_degree:
                    in_degree[task_id] += 1

        # Kahn算法进行拓扑排序
        queue = []
        result = []

        # 找到所有入度为0的节点
        for task_id in range(self.Nij):
            if in_degree[task_id] == 0:
                queue.append(task_id)

        # 随机化初始队列以产生不同的拓扑排序
        random.shuffle(queue)

        while queue:
            # 随机选择一个入度为0的节点（增加随机性）
            current = queue.pop(random.randint(0, len(queue) - 1))
            result.append(current)

            # 更新后继节点的入度
            if current in dependencies:
                for successor in dependencies[current]['successors']:
                    if successor in in_degree:
                        in_degree[successor] -= 1
                        if in_degree[successor] == 0:
                            queue.append(successor)

        # 如果结果长度不等于任务数，说明有环或其他问题
        if len(result) != self.Nij:
            # 返回简单的顺序排列作为备选
            return list(range(self.Nij))

        return result

    def get_prompt(self, x, y, obj_p):
        args = get_argument_parser()
        Nij = args.Nij
        K = args.K
        n_uav = N_n
        len_order = n_uav * Nij
        len_location = n_uav * Nij
        len_A = n_uav

        pop_content = ""
        for i in range(len(x)):
            # 变量分割
            order = [int(v) for v in x[i][:len_order]]
            location = [int(v) for v in x[i][len_order:len_order + len_location]]
            A = [round(float(v), 6) for v in x[i][len_order + len_location:len_order + len_location + len_A]]
            point_vars = order + location + A
            pop_content += "<start>" + ",".join(str(v) for v in point_vars) + "<end> \n"
            pop_content += "objective 1: " + str(round(obj_p[i][0], 4))
            pop_content += " objective 2: " + str(round(obj_p[i][1], 4))
            if len(obj_p[i]) > 2:
                pop_content += " objective 3: " + str(round(obj_p[i][2], 4))
            pop_content += "\n\n"

        prompt_content = (
            f"Now you will help me minimize {len(obj_p[0])} objectives with {len(x[0])} variables. "
            f"The variables are divided into three parts: "
            f"the first {len_order} are UAV task order variables (for each UAV, a permutation of 0~{Nij - 1}, integers only), "
            f"the next {len_location} are UAV position variables (integers in [1,{K + 1}]), "
            f"and the last {len_A} are allocation ratios A1, A2, A3 (real numbers in [0,1], sum=1). "
            f"I have some points with their objective values. The points start with <start> and end with <end>.\n\n"
            f"{pop_content}"
            f"Give me two new points that are different from all points above, and not dominated by any of the above. "
            f"The first {len_order} values must be a permutation of 0~{Nij - 1} for each UAV. "
            f"The next {len_location} values must be integers in [1,{K + 1}]. "
            f"The last {len_A} values (A1, A2, A3) must be in [0,1] and sum to 1. "
            f"Do not write code. Do not give any explanation. Each output new point must start with <start> and end with <end>."
        )
        return prompt_content  # 返回构造好的prompt字符串

    def parse_llm_points_raw(self, response, n_var, len_order, len_location, len_A, K):
        """
        解析LLM输出的<start>...<end>格式字符串，返回shape为(2, n_var)的numpy数组
        保证order部分符合拓扑排序，location为[1, K+1]区间整数，A为[0,1]且归一化
        """
        import numpy as np
        import re

        points = re.findall(r"<start>(.*?)<end>", response)
        valid_points = []

        for point_str in points:
            try:
                arr = np.fromstring(point_str, sep=",", dtype=float)
                if len(arr) != n_var or np.any(np.isnan(arr)):
                    continue

                # 修复order部分：确保每个UAV的任务顺序符合拓扑排序
                corrected_order = []
                for uav_idx in range(N_n):
                    start_idx = uav_idx * self.Nij
                    end_idx = (uav_idx + 1) * self.Nij

                    # 获取当前UAV的order部分
                    uav_order = arr[start_idx:end_idx].astype(int)

                    # 生成有效的拓扑排序来替换
                    valid_order = self._generate_valid_topological_order(uav_idx)

                    # 如果LLM生成的顺序不合法，直接使用有效的拓扑排序
                    # 否则可以尝试修复LLM的顺序使其符合拓扑约束
                    corrected_order.extend(valid_order)

                # 更新order部分
                arr[:len_order] = np.array(corrected_order)

                # location部分clip到[1, K+1]并转为int
                arr[len_order:len_order + len_location] = np.clip(np.round(arr[len_order:len_order + len_location]), 1,
                                                                  K + 1)
                arr[len_order:len_order + len_location] = arr[len_order:len_order + len_location].astype(int)

                # A部分clip到[0,1]
                arr[len_order + len_location:len_order + len_location + len_A] = np.clip(
                    arr[len_order + len_location:len_order + len_location + len_A], 0, 1)
                # A部分归一化（和为1）
                s = np.sum(arr[len_order + len_location:len_order + len_location + len_A])
                if s > 0:
                    arr[len_order + len_location:len_order + len_location + len_A] /= s
                else:
                    # 如果和为0，设置为均匀分布
                    arr[len_order + len_location:len_order + len_location + len_A] = 1.0 / len_A

                valid_points.append(arr)
            except Exception as e:
                print(f"Error processing point: {e}")
                continue

        # 如果没有足够的有效点，生成备用点
        while len(valid_points) < 2:
            fallback_point = self._generate_fallback_point(n_var, len_order, len_location, len_A, K)
            valid_points.append(fallback_point)

        return np.stack(valid_points[:2], axis=0)

    def _generate_fallback_point(self, n_var, len_order, len_location, len_A, K):
        """生成符合约束的备用解"""
        arr = np.zeros(n_var)

        # 生成符合拓扑排序的order部分
        corrected_order = []
        for uav_idx in range(N_n):
            valid_order = self._generate_valid_topological_order(uav_idx)
            corrected_order.extend(valid_order)
        arr[:len_order] = np.array(corrected_order)

        # 生成随机location
        arr[len_order:len_order + len_location] = np.random.randint(1, K + 2, size=len_location)

        # 生成归一化的A
        A_values = np.random.dirichlet([1] * len_A)
        arr[len_order + len_location:len_order + len_location + len_A] = A_values

        return arr

    def _do(self, _, X, Y, debug_mode, model_LLM, endpoint, key, out_filename, parents_obj, **kwargs):

        # x_scale = 1000.0
        y_p = np.zeros(len(Y))
        x_p = np.zeros((len(X), len(X[0][0])))
        for i in range(len(Y)):
            y_p[i] = round(Y[i][0][0], 4)
            x_p[i] = X[i][0]


        # x_p = x_scale*x_p

        sort_idx = sorted(range(len(Y)), key=lambda k: Y[k], reverse=True)
        x_p = [x_p[idx] for idx in sort_idx]
        y_p = [y_p[idx] for idx in sort_idx]
        obj_p = parents_obj[0][:10].get("F")
        obj_p = [obj_p[idx] for idx in sort_idx]

        prompt_content = self.get_prompt(x_p, y_p, obj_p)

        if debug_mode:
            print(prompt_content)
            print("> enter to continue")
            input()

        payload = json.dumps({
            # "model": "gpt-3.5-turbo",
            # "model": "gpt-4-0613",
            "model": model_LLM,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_content
                }
            ],
            "safe_mode": False
        })
        headers = {
            'Authorization': 'Bearer ' + key,
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json',
            'x-api2d-no-cache': 1
        }

        conn = http.client.HTTPSConnection(endpoint)

        # conn = http.client.HTTPSConnection("oa.api2d.site")
        # conn.request("POST", "/v1/chat/completions", payload, headers)

        retries = 50  # Number of retries
        retry_delay = 2  # Delay between retries (in seconds)
        while retries > 0:
            try:

                conn.request("POST", "/v1/chat/completions", payload, headers)

                res = conn.getresponse()
                data = res.read()

                # response_data = data.decode('utf-8')
                json_data = json.loads(data)
                # pprint.pprint(json_data)
                response = json_data['choices'][0]['message']['content']

                while (len(re.findall(r"<start>(.*?)<end>", response)) < 2):
                    conn = http.client.HTTPSConnection(endpoint)
                    # conn = http.client.HTTPSConnection("oa.api2d.site")
                    conn.request("POST", "/v1/chat/completions", payload, headers)
                    res = conn.getresponse()
                    data = res.read()

                    # response_data = data.decode('utf-8')
                    json_data = json.loads(data)

                    response = json_data['choices'][0]['message']['content']

                args = get_argument_parser()
                n_var = len(x_p[0])
                len_order = N_n * args.Nij
                len_location = N_n * args.Nij
                len_A = N_n

                # 解析并修复LLM输出
                try:
                    parsed_points = self.parse_llm_points_raw(response, n_var, len_order, len_location, len_A, args.K)
                    off1 = parsed_points[0]
                    off2 = parsed_points[1]
                except Exception as e:
                    print(f"Error parsing LLM output: {e}, using fallback generation")
                    off1 = self._generate_fallback_point(n_var, len_order, len_location, len_A, args.K)
                    off2 = self._generate_fallback_point(n_var, len_order, len_location, len_A, args.K)

                if out_filename != None:
                    filename = out_filename
                    file = open(filename, "a")
                    for i in range(len(x_p)):
                        for j in range(len(x_p[i])):
                            file.write("{:.4f} ".format(x_p[i][j]))
                        file.write("{:.4f} ".format(y_p[i]))
                    for i in range(len(off1)):
                        file.write("{:.4f} ".format(off1[i]))
                    for i in range(len(off1)):
                        file.write("{:.4f} ".format(off2[i]))
                    # file.write("{:.4f} {:.4f} {:.4f} {:.4f} \n".format(off1[0],off1[1],off[1][0][0],off[1][0][1]))
                    file.write("\n")
                    file.close

                off1 = off1[np.newaxis, :]
                off2 = off2[np.newaxis, :]
                off = np.concatenate([off1, off2], axis=0)
                off = off[:, np.newaxis, :]  # shape: (2, 1, n_var)
                print(off)
                break

            except:
                print(f"Request {retries} failed !  ")
                retries -= 1
                if retries > 0:
                    print("Retrying in", retry_delay, "seconds...")
                    time.sleep(retry_delay)

        if debug_mode:
            print(response)
            if 'off1' in locals():
                print("Generated point 1:", off1)
                print("Generated point 2:", off2)
            print(off)
            print("> enter to continue")
            input()
        # print(off.shape)

        return off


class GPT_interface(GPT):

    def __init__(self, **kwargs):
        super().__init__(n_new=1, **kwargs)
