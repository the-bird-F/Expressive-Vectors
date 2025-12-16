import torch
import os
from collections import OrderedDict
import argparse
import matplotlib.pyplot as plt

def visualize_diff_norms(diff_norms, sort_flag = False, x_flag = True):
    # sort
    if sort_flag:
        diff_items = sorted(diff_norms.items(), key=lambda item: item[1], reverse=True)
    else:
        diff_items = diff_norms.items()
    values = []
    keys = []
    for k,v in diff_items:
        if v < 10:
            values.append(v)
            keys.append(k)

    # Plot
    plt.figure(figsize=(120, 4))
    plt.bar(range(len(values)), values, color="#4682B4", edgecolor="black")
    plt.xlabel("Layer Index (Sorted by L2 Norm)", fontsize=12)
    plt.ylabel("L2 Norm of Difference", fontsize=12)
    plt.title("Magnitude of Parameter Changes Across Layers (80k)", fontsize=14)
    if x_flag:
        plt.xticks(ticks=range(len(keys)), labels=keys, rotation=90, fontsize = 7) 
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("parameter.png", dpi=300) 
    plt.show()

def flatten_state_dict(d, parent_key='', sep='@'):
    """递归展开嵌套 OrderedDict"""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + str(k) if parent_key else str(k)
        if isinstance(v, (OrderedDict, dict)):
            flattened = flatten_state_dict(v, new_key, sep)
            items.extend(flattened.items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_state_dict(flat_dict, sep='@'):
    """将扁平化的 state_dict 恢复为嵌套 dict"""
    unflat = dict()
    for flat_key, value in flat_dict.items():
        parts = flat_key.split(sep)
        d = unflat
        for part in parts[:-1]:
            if part not in d or not isinstance(d[part], dict):
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return unflat


def load_state_dict(path):
    """读取模型"""
    print(f"[info] Loading model from: {path}")
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, (dict, OrderedDict)):
        return flatten_state_dict(checkpoint)
        

def compare_models(model_dir, model1, model2):
    """比较微调前后模型的数值差距"""
    state_dict1 = load_state_dict(os.path.join(model_dir,model1))
    state_dict2 = load_state_dict(os.path.join(model_dir,model2))

    diff_dict = {}
    diff_norms = {}

    for key in state_dict1:
        if key not in state_dict2:
            print(f"[info] model2 中缺少参数: {key}")
            continue

        param1, param2 = state_dict1[key], state_dict2[key]

        if not isinstance(param1, torch.Tensor) or not isinstance(param2, torch.Tensor):
            print(f"[info] 参数 {key} 不是 Tensor 类型，跳过")
            continue

        if param1.shape != param2.shape:
            print(f"[info] 参数维度不一致: {key}: {param1.shape} vs {param2.shape}")
            continue

        if not torch.is_floating_point(param1):
            print(f"[info] 非浮点类型参数: {key}: {param1} (dtype = {param1.dtype})")
            # print(f"[info] {key} 参数量：{param1.numel()} 形式：{param1.size()}")
            continue

        diff = param2 - param1
        diff_dict[key] = diff
        # print(f"[info] {key} 参数量：{diff.numel()} 形式：{diff.size()}")
        diff_norms[key] = diff.pow(2).mean().sqrt().item() # torch.norm(diff).item()
    
    # print(f"[info] 模型参数：{diff_norms}")
    # diff_path = os.path.join(output_dir, 'model_diff.pt')
    # torch.save(diff, diff_path)
    # print(f"[info] 差值已保存: {diff_path}")

    # 统计
    print("\n=== 各层参数差值的 L2 范数 (自上而下排序) ===")
    for key, norm in sorted(diff_norms.items(), key=lambda item: item[1], reverse=True):
        print(f"{key}: ||diff|| = {norm:.6f}")

    return state_dict1, state_dict2, diff_dict, diff_norms

def assert_nested_dict_equal(d1, d2):
    """验证两个模型的结构相同"""
    assert isinstance(d1, dict) and isinstance(d2, dict), "Both should be dicts"
    assert d1.keys() == d2.keys(), f"Keys mismatch: {d1.keys()} != {d2.keys()}"
    
    for k in d1:
        v1, v2 = d1[k], d2[k]
        if isinstance(v1, dict) and isinstance(v2, dict):
            assert_nested_dict_equal(v1, v2)  # 递归比较
        else:
            # assert v1 == v2, f"Values mismatch at key {k}: {v1} != {v2}"
            continue
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="模型所在目录")
    parser.add_argument("--pre_model", type=str, default="pretrained_model_1250000.pt",
                        help="预训练模型名称")
    parser.add_argument("--model1", type=str, default='pretrained_model_1250000.pt',
                        help="模型1文件名")
    parser.add_argument("--model2", type=str, default='model_60000.pt',
                        help="模型2文件名")
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="Alpha系数")
    parser.add_argument("--norm_flag", type=bool, default=False,
                        help="是否开启归一化标志")

    args = parser.parse_args()
    model_dir = args.model_dir
    pre_model = args.pre_model
    model1 = args.model1
    model2 = args.model2
    alpha = args.alpha
    norm_flag = args.norm_flag

    model1_flag = (model1 == pre_model)

    state_dict1, state_dict2, diff_dict, diff_norms = compare_models(model_dir, model1, model2)

    diff_nested = unflatten_state_dict(diff_dict)
    diff_path = os.path.join(model_dir, f"diff_{model2}_{model1}")
    torch.save(diff_nested, diff_path)

    interpolated = {}
    for k, v in state_dict1.items():
        if k in diff_dict:
            if norm_flag:
                interpolated[k] = v + alpha * diff_dict[k] * diff_norms[k]
            else:
                interpolated[k] = v + alpha * diff_dict[k] 
        else:
            interpolated[k] = v
            
    interpolated_nested = unflatten_state_dict(interpolated)
    if model1_flag:
        interpolated_name = f'interpolated_{model2}_a{alpha:.1f}_n{str(norm_flag)}.pt'
    else:
        interpolated_name = f'interpolated_{model2}_a{alpha:.1f}_n{str(norm_flag)}_{model1}.pt'
    interpolated_path = os.path.join(model_dir, interpolated_name)
    torch.save(interpolated_nested, interpolated_path)

    state_dict1, state_dict2, diff_dict, diff_norms = compare_models(model_dir, pre_model, interpolated_name)
    print(f"\n 插值模型已保存: {interpolated_path}")