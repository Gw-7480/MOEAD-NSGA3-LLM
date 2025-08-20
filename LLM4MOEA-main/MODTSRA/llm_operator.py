import copy
import json
import random
import numpy as np

class LLMSearchOperator:
    """
    黑盒搜索算子（LLM驱动）：
    - 输入：双亲（parents），任务DAG、资源和约束信息（从个体内部拿）
    - 输出：一个或两个子代（修改S和A；B暂保持或轻微继承）
    设计要点：
      1) 构造强约束提示词，让LLM只在‘顺序/位置/资源比例’上提出改进；
      2) 解析 LLM 返回的 JSON 方案；
      3) 进行稳健的 Repair（拓扑合法、位置范围、A 归一化等）；
      4) 若LLM不可用或返回无效，回退到启发式（可运行）。
    """
    def __init__(self, K, Nij, use_llm=True, llm_caller=None, llm_temperature=0.2, p_apply_llm=1.0):
        """
        Args:
            K: 核数量（执行位置范围 1..K+1）
            Nij: 每个 workflow 的子任务数
            use_llm: 是否启用LLM
            llm_caller(prompt:str)->str: 你接入 deepseek 的函数；返回 JSON 字符串
            llm_temperature: 给LLM的温度
            p_apply_llm: 本次交叉是否用LLM（可<1做混合）
        """
        self.K = int(K)
        self.Nij = int(Nij)
        self.use_llm = use_llm
        self.llm_caller = llm_caller
        self.llm_temperature = llm_temperature
        self.p_apply_llm = p_apply_llm

    # ======= 对外主入口 ======= #
    def propose_offspring_pair(self, parent1, parent2):
        """
        输入两个父代 Individual，输出两个子代 Individual（深拷贝后修改）
        """
        c1 = copy.deepcopy(parent1)
        c2 = copy.deepcopy(parent2)

        # 按概率使用LLM；否则用启发式回退
        if self.use_llm and (random.random() <= self.p_apply_llm) and (self.llm_caller is not None):
            try:
                plan = self._ask_llm_for_plan(parent1, parent2)
                if plan:
                    self._apply_plan_to_individual(c1, plan.get("child1"))
                    self._apply_plan_to_individual(c2, plan.get("child2", plan.get("child1")))
                    self._repair_individual(c1)
                    self._repair_individual(c2)
                    return c1, c2
            except Exception:
                # 回退到启发式
                pass

        # 启发式回退（可运行保障）
        self._heuristic_edit(c1, c2)
        self._repair_individual(c1)
        self._repair_individual(c2)
        return c1, c2

    # ======= LLM 相关 ======= #
    def _ask_llm_for_plan(self, p1, p2):
        """
        构造提示词 -> 调用 deepseek -> 解析 JSON
        返回 dict:
        {
          "child1": {"S":[{"order":[...], "location":[...]}, ...], "A":[...]},
          "child2": {"S":[...], "A":[...]}
        }
        """
        prompt = self._build_prompt(p1, p2)
        raw = self.llm_caller(prompt)  # 由你接入 deepseek
        # 允许 LLM 有少量自然语言，提取第一个 {...} JSON
        json_str = self._extract_first_json(raw)
        plan = json.loads(json_str)
        print("=== LLM Prompt ===")
        print(prompt[:500])  # 打印前 500 字
        print("=== LLM Raw Output ===")
        print(raw)
        return plan

    def _build_prompt(self, p1, p2):
        """
        面向 deepseek 的约束明确的指令，强制 JSON 输出。
        只提供 S/A，B 不让 LLM 瞎动（B 延用父代或轻微交换）。
        """
        def encode_ind(ind):
            S = []
            for uav in ind.chromosome['S']:
                S.append({
                    "order": list(map(int, uav.workflow.order)),
                    "location": list(map(int, uav.workflow.location))
                })
            A = list(map(float, ind.chromosome['A']))
            return {"S": S, "A": A}

        p1_json = encode_ind(p1)
        p2_json = encode_ind(p2)

        spec = {
            "constraints": {
                "order_topological": True,
                "location_range": [1, self.K + 1],
                "A_nonnegative": True,
                "A_l1_norm": 1.0
            },
            "objectives": ["minimize Time", "minimize Energy", "minimize Cost"],
            "notes": [
                "Only propose changes to S.order, S.location and A.",
                "Keep task precedence: an item can appear only after all its predecessors.",
                "Return strictly valid JSON with keys: child1, child2; each has S(list) and A(list).",
                "Lengths must match parents (same #UAV, same Nij).",
                "Do not include any text outside JSON."
            ]
        }

        prompt = (
            "You are a black-box variation operator for a multi-objective evolutionary algorithm.\n"
            "Given two parents, propose two improved children by editing scheduling order and execution locations for each UAV (S), "
            "and the resource allocation vector A. Follow constraints exactly.\n\n"
            f"SPEC = {json.dumps(spec, ensure_ascii=False)}\n\n"
            f"PARENT_1 = {json.dumps(p1_json, ensure_ascii=False)}\n\n"
            f"PARENT_2 = {json.dumps(p2_json, ensure_ascii=False)}\n\n"
            'Respond ONLY with JSON like:\n'
            '{\n'
            '  "child1": {"S":[{"order":[...], "location":[...]} , ...], "A":[... ]},\n'
            '  "child2": {"S":[{"order":[...], "location":[...]} , ...], "A":[... ]}\n'
            '}\n'
            f"Do not write code. Do not give any explanation."
        )
        return prompt

    @staticmethod
    def _extract_first_json(text):
        """
        从文本中提取第一个 JSON 对象（鲁棒处理 LLM 前后赘词）。
        """
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or end <= start:
            raise ValueError("LLM未返回有效JSON。")
        return text[start:end+1]

    # ======= 应用与修复 ======= #
    def _apply_plan_to_individual(self, child, plan_child):
        if plan_child is None:
            return
        # A
        if "A" in plan_child and isinstance(plan_child["A"], list):
            a = np.array(plan_child["A"], dtype=float)
            a = np.abs(a)
            s = a.sum()
            child.chromosome['A'] = (a / s).tolist() if s > 0 else child.chromosome['A']

        # S
        if "S" in plan_child and isinstance(plan_child["S"], list):
            # 数量对齐
            U = min(len(child.chromosome['S']), len(plan_child["S"]))
            for i in range(U):
                gene = child.chromosome['S'][i]
                plan_si = plan_child["S"][i]
                if "location" in plan_si:
                    loc = list(map(int, plan_si["location"]))
                    # 长度对齐
                    if len(loc) == len(gene.workflow.location):
                        # 截断到合法范围
                        loc = [min(self.K+1, max(1, x)) for x in loc]
                        gene.workflow.location = loc
                if "order" in plan_si:
                    ord_ = list(map(int, plan_si["order"]))
                    if len(ord_) == len(gene.workflow.order):
                        gene.workflow.order = ord_

    def _repair_individual(self, ind):
        # A 归一化
        a = np.array(ind.chromosome['A'], dtype=float)
        a = np.abs(a)
        ind.chromosome['A'] = (a / (a.sum() + 1e-12)).tolist()

        # S 修复：order 拓扑合法 + location范围
        for uav in ind.chromosome['S']:
            # 位置
            loc = [min(self.K+1, max(1, int(x))) for x in uav.workflow.location]
            uav.workflow.location = loc
            # 拓扑检查/修复
            self._topo_repair(uav)

    def _topo_repair(self, uav):
        """
        若给定 order 违反前驱约束，做一次保守修复：
        - 过滤到合法拓扑序；缺失节点按就绪时刻插入；
        - 若失败则回退到 initializeWorkflowSequence 的结果（由外部保证存在）。
        """
        order = list(map(int, uav.workflow.order))
        n = len(order)
        task_set = uav.workflow.taskSet

        seen = set()
        new_order = []
        remaining = set(range(n))

        # 先保留输入顺序里能合法就位的
        for t in order:
            preds = set(task_set[t].preTaskSet)
            if preds.issubset(seen):
                new_order.append(t)
                seen.add(t)
                remaining.discard(t)

        # 逐步插入剩余任务
        changed = True
        while remaining and changed:
            changed = False
            for t in list(remaining):
                preds = set(task_set[t].preTaskSet)
                if preds.issubset(seen):
                    new_order.append(t)
                    seen.add(t)
                    remaining.discard(t)
                    changed = True

        # 若仍有未放入的（图或输入异常），回退到原顺序的拓扑排序近似：按ID升序补齐
        if remaining:
            new_order.extend(sorted(list(remaining)))

        uav.workflow.order = new_order[:n]

    # ======= 启发式回退（无LLM时也能跑） ======= #
    def _heuristic_edit(self, c1, c2):
        """
        简易‘交叉+微扰’：
          - 对每个 UAV：交换部分 prefix 的 order 与 location；
          - A 做一次 SBX 风格混合 + 归一化。
        """
        # S: 交换前缀
        for i in range(len(c1.chromosome['S'])):
            g1 = c1.chromosome['S'][i]
            g2 = c2.chromosome['S'][i]
            L = len(g1.workflow.order)
            if L <= 2:
                continue
            cpt = random.randint(1, L-1)
            # 位置前缀交换
            for j in range(cpt):
                g1.workflow.location[j], g2.workflow.location[j] = g2.workflow.location[j], g1.workflow.location[j]
            # 顺序前缀交换（去重拼接）
            pref1 = g1.workflow.order[:cpt]
            pref2 = g2.workflow.order[:cpt]
            tail1 = [x for x in g1.workflow.order if x not in pref2]
            tail2 = [x for x in g2.workflow.order if x not in pref1]
            g1.workflow.order = pref2 + tail1
            g2.workflow.order = pref1 + tail2

        # A: SBX-like
        A1 = np.array(c1.chromosome['A'], dtype=float)
        A2 = np.array(c2.chromosome['A'], dtype=float)
        u = np.random.rand(*A1.shape)
        r = np.where(u <= 0.5, (2*u)**(1/2), (1/(2*(1-u)))**(1/2))
        new1 = 0.5*((1+r)*A1 + (1-r)*A2)
        new2 = 0.5*((1-r)*A1 + (1+r)*A2)
        c1.chromosome['A'] = (np.abs(new1)/ (np.sum(np.abs(new1))+1e-12)).tolist()
        c2.chromosome['A'] = (np.abs(new2)/ (np.sum(np.abs(new2))+1e-12)).tolist()