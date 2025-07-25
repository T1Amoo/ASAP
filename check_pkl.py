import joblib
import numpy as np

file_path = '/home/hiyio/ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl'

# 加载数据
data = joblib.load(file_path)

print("=== 数据集详细信息 ===")
print(f"顶层数据类型: {type(data)}")
print(f"顶层键: {list(data.keys())}")

# 获取主要数据
main_key = list(data.keys())[0]
main_data = data[main_key]

print(f"\n主数据键: {main_key}")
print(f"主数据类型: {type(main_data)}")

if isinstance(main_data, dict):
    print(f"主数据包含的键: {list(main_data.keys())}")
    
    # 查看每个键的详细信息
    for key, value in main_data.items():
        print(f"\n--- {key} ---")
        print(f"  类型: {type(value)}")
        
        if hasattr(value, 'shape'):
            print(f"  形状: {value.shape}")
            print(f"  数据类型: {value.dtype}")
            print(f"  数据范围: {value.min():.3f} ~ {value.max():.3f}")
            
            # 如果是二维数组，显示前几行
            if len(value.shape) == 2:
                print(f"  前3行数据:")
                print(value[:3])
            elif len(value.shape) == 1:
                print(f"  前10个值: {value[:10]}")
                
        elif isinstance(value, (list, tuple)):
            print(f"  长度: {len(value)}")
            if len(value) > 0:
                print(f"  第一个元素类型: {type(value[0])}")
                print(f"  前几个元素: {value[:3]}")
                
        elif isinstance(value, (str, int, float)):
            print(f"  值: {value}")
            
        else:
            print(f"  内容: {str(value)[:100]}...")

# 如果数据看起来像动作捕捉数据，提供一些分析
print("\n=== 动作捕捉数据分析 ===")
if isinstance(main_data, dict):
    # 常见的动作捕捉数据键
    motion_keys = ['poses', 'trans', 'root_trans', 'root_orient', 'pose_body', 'mocap_framerate']
    found_keys = [k for k in main_data.keys() if any(mk in k.lower() for mk in motion_keys)]
    
    if found_keys:
        print(f"找到可能的动作数据键: {found_keys}")
        
        for key in found_keys:
            if hasattr(main_data[key], 'shape'):
                shape = main_data[key].shape
                print(f"  {key}: 形状 {shape}")
                
                # 分析可能的含义
                if len(shape) == 2:
                    frames, dims = shape
                    print(f"    可能是: {frames} 帧, 每帧 {dims} 维数据")
                    if dims == 3:
                        print(f"    可能是位置数据 (x, y, z)")
                    elif dims % 3 == 0:
                        print(f"    可能是 {dims//3} 个关节的位置数据")