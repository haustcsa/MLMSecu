import json
from collections import defaultdict

def analyze_predictions(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    clean_images = []
    poisoned_images = []
    clean_num=0
    poisoned_num=0
    for item in data:
        if "clean" in item["image"]:
            clean_images.append(item)
            clean_num+=1
        elif "poisoned" in item["image"]:
            poisoned_images.append(item)
            poisoned_num+=1
    clean_dict = defaultdict(list)
    poisoned_dict = defaultdict(list)
    accurate_count = 0
    detect_success_count = 0
    for item in clean_images:
        base_name = item["image"][:-5]  # 去掉最后 5 个字符
        clean_dict[base_name].append(item["prediction"])
    # print(clean_dict)
    # print(len (clean_dict))
    # print(clean_dict.items())
    for item in poisoned_images:
        base_name = item["image"][:-5]  # 去掉最后 5 个字符
        poisoned_dict[base_name].append(item["prediction"])

    for base_name, predictions in clean_dict.items():
        if len(predictions) == 2 and predictions[0] != predictions[1]:
            accurate_count += 1
    
    for base_name, predictions in poisoned_dict.items():
        if len(predictions) == 2 and predictions[0] == predictions[1]:
            detect_success_count += 1
    clean_num=clean_num/2
    poisoned_num=poisoned_num/2
    print("## 成功检测出干净样本的数量为:%s" % accurate_count)
    print("## 成功检测出有毒样本的数量为 :%s" % detect_success_count)
    detect_clean_accuracy = accurate_count / clean_num if clean_num > 0 else 0
    detect_poisoned_success_rate = detect_success_count / poisoned_num if poisoned_num > 0 else 0
    detect_success_rate=detect_clean_accuracy + detect_poisoned_success_rate
    return {
        "干净样本": clean_num,
        "有毒样本": poisoned_num,
        "检测干净成功率": detect_clean_accuracy,
        "检测有毒成功率": detect_poisoned_success_rate,
        "检测准确率":detect_success_rate/2
    }


# json_file = "/root/autodl-fs/root/badnets-pytorch-master/dataset/FTDdatatest/MNIST-badnets/prediction_results.json"
# # 调用函数并打印结果
# results = analyze_predictions(json_file)
# print("Analysis Results:", results)