import copy


def minimize(problem, algorithm, termination=None, copy_algorithm=True, copy_termination=True, **kwargs):
    """

    单变量或多变量函数的最小化，目标与约束条件

    此函数作为便捷工具，通过默认配置执行多种算法（这些配置在单一测试场景中已验证有效）。但进化计算的核心思想在于定制元算法。建议通过面向对象接口定制算法以提升收敛性。

    Parameters
    ----------

    problem : :class:~pymoo.core.problem.Problem
　　基于 pymoo 定义的问题对象

    algorithm : :class:~pymoo.core.algorithm.Algorithm
　　用于优化的算法对象

    termination : :class:~pymoo.core.termination.Termination 或 tuple
　　用于停止算法的终止准则

    seed : integer
　　随机数种子

    verbose : bool
　　是否打印输出信息

    display : :class:~pymoo.util.display.Display
　　算法默认包含输出显示对象，可自定义覆盖

    callback : :class:~pymoo.core.callback.Callback
　　每次算法迭代时调用的回调对象

    save_history : bool
　　是否保存优化历史记录

    copy_algorithm : bool
　　优化前是否复制算法对象

    Returns
    -------
    res : :class:`~pymoo.core.result.Result`
        The optimization result represented as an object.

    """

    # 创建算法对象的副本，避免副作用（如多次运行时参数被修改）
    if copy_algorithm:
        algorithm = copy.deepcopy(algorithm)

    # 如果算法对象还没有绑定问题，则进行初始化
    if algorithm.problem is None:
        if termination is not None:
            # 如果需要，也对终止条件对象进行深拷贝，避免副作用
            if copy_termination:
                termination = copy.deepcopy(termination)

            # 将终止条件加入初始化参数
            kwargs["termination"] = termination

        # 用指定问题和参数初始化算法对象   这是初始化了参考方向向量66 领域10 以及分解方法PBI
        algorithm.setup(problem, **kwargs)

    # 实际运行算法，得到结果对象
    res = algorithm.run()

    # 将深拷贝后的算法对象存入结果对象，便于后续分析
    res.algorithm = algorithm

    # 返回优化结果对象
    return res
