from Config import Config
from Utility import (
    BAM_main_algorithm_tabular,
    prepare_config_and_log,
    # generate_random_tabular_data_function,  # 生成随机数据的函数，用于目标模型
)

# 准备配置和日志
prepare_config_and_log()
config = Config.instance  # 配置实例

# 设置BAM算法的参数
num_of_classes = 12
k = 3000
epsilon = 0.05
population_size = 10000
generations = 30
search_spread = 10

# 加载目标模型
victim_model = load_model()  # 这里需要替换为用户实际加载目标模型的代码

# 运行BAM算法
surrogate_model = BAM_main_algorithm_tabular(
    victim_model,
    SurrogateModelClass,
    generate_random_tabular_data_function,
    num_of_classes=num_of_classes,
    k=k,
    epsilon=epsilon,
    population_size=population_size,
    generations=generations,
    search_spread=search_spread,
)

# 测试代理模型的准确度
surrogate_acc = surrogate_model.test_model()  # 需要用户提供实际的测试函数
print(f"代理模型的准确率是 {surrogate_acc}")
