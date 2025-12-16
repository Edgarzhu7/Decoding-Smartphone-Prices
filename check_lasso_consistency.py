"""
检查所有模型使用的Lasso参数是否一致
"""
import re
import os

def extract_lasso_params(file_path):
    """从文件中提取LassoCV参数"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    params = {
        'file': os.path.basename(file_path),
        'lasso_calls': [],
        'preprocess': 'preprocess_features' in content,
        'scaler': 'StandardScaler' in content,
        'random_state': None,
        'max_iter': None,
        'cv': None,
    }
    
    # 提取所有LassoCV调用
    lasso_pattern = r'LassoCV\([^)]+\)'
    matches = re.findall(lasso_pattern, content)
    
    for match in matches:
        lasso_params = {}
        # 提取cv参数
        cv_match = re.search(r'cv=([^,)]+)', match)
        if cv_match:
            lasso_params['cv'] = cv_match.group(1)
        
        # 提取random_state
        rs_match = re.search(r'random_state=(\d+)', match)
        if rs_match:
            lasso_params['random_state'] = rs_match.group(1)
        
        # 提取max_iter
        mi_match = re.search(r'max_iter=(\d+)', match)
        if mi_match:
            lasso_params['max_iter'] = mi_match.group(1)
        
        params['lasso_calls'].append(lasso_params)
    
    return params

def check_quarter_models():
    """检查季度模型"""
    quarter_dir = 'Quarter'
    files = [
        'lasso_price_prediction.py',
        'predicted_jevons_index_calculator.py',
        'predicted_jevons_index_with_error.py',
        'lasso_delta_price_change.py',
        'lasso_delta_price_change_with_error.py',
    ]
    
    print("=" * 80)
    print("季度模型 Lasso 参数检查")
    print("=" * 80)
    
    all_params = []
    for file in files:
        file_path = os.path.join(quarter_dir, file)
        if os.path.exists(file_path):
            params = extract_lasso_params(file_path)
            all_params.append(params)
            print(f"\n文件: {params['file']}")
            print(f"  使用 preprocess_features: {params['preprocess']}")
            print(f"  使用 StandardScaler: {params['scaler']}")
            print(f"  LassoCV 调用次数: {len(params['lasso_calls'])}")
            for i, call in enumerate(params['lasso_calls'], 1):
                print(f"    调用 {i}:")
                print(f"      cv: {call.get('cv', 'N/A')}")
                print(f"      random_state: {call.get('random_state', 'N/A')}")
                print(f"      max_iter: {call.get('max_iter', 'N/A')}")
    
    # 检查一致性
    print("\n" + "=" * 80)
    print("一致性检查")
    print("=" * 80)
    
    # 检查所有模型的random_state和max_iter是否一致
    all_rs = set()
    all_mi = set()
    
    for params in all_params:
        for call in params['lasso_calls']:
            if call.get('random_state'):
                all_rs.add(call['random_state'])
            if call.get('max_iter'):
                all_mi.add(call['max_iter'])
    
    print(f"\nRandom State 值: {sorted(all_rs)}")
    if len(all_rs) == 1:
        print("✓ 所有模型使用相同的 random_state")
    else:
        print("✗ 警告: random_state 不一致!")
    
    print(f"\nMax Iter 值: {sorted(all_mi)}")
    if len(all_mi) == 1:
        print("✓ 所有模型使用相同的 max_iter")
    else:
        print("✗ 警告: max_iter 不一致!")

def check_annual_models():
    """检查年度模型"""
    annual_dir = 'annual'
    files = [
        'predicted_annual_jevons_index_calculator.py',
        'predicted_annual_jevons_index_with_error.py',
        'lasso_delta_price_change_annual.py',
        'lasso_delta_price_change_annual_with_error.py',
    ]
    
    print("\n" + "=" * 80)
    print("年度模型 Lasso 参数检查")
    print("=" * 80)
    
    all_params = []
    for file in files:
        file_path = os.path.join(annual_dir, file)
        if os.path.exists(file_path):
            params = extract_lasso_params(file_path)
            all_params.append(params)
            print(f"\n文件: {params['file']}")
            print(f"  使用 preprocess_features: {params['preprocess']}")
            print(f"  使用 StandardScaler: {params['scaler']}")
            print(f"  LassoCV 调用次数: {len(params['lasso_calls'])}")
            for i, call in enumerate(params['lasso_calls'], 1):
                print(f"    调用 {i}:")
                print(f"      cv: {call.get('cv', 'N/A')}")
                print(f"      random_state: {call.get('random_state', 'N/A')}")
                print(f"      max_iter: {call.get('max_iter', 'N/A')}")
    
    # 检查一致性
    print("\n" + "=" * 80)
    print("一致性检查")
    print("=" * 80)
    
    all_rs = set()
    all_mi = set()
    
    for params in all_params:
        for call in params['lasso_calls']:
            if call.get('random_state'):
                all_rs.add(call['random_state'])
            if call.get('max_iter'):
                all_mi.add(call['max_iter'])
    
    print(f"\nRandom State 值: {sorted(all_rs)}")
    if len(all_rs) == 1:
        print("✓ 所有模型使用相同的 random_state")
    else:
        print("✗ 警告: random_state 不一致!")
    
    print(f"\nMax Iter 值: {sorted(all_mi)}")
    if len(all_mi) == 1:
        print("✓ 所有模型使用相同的 max_iter")
    else:
        print("✗ 警告: max_iter 不一致!")

def check_feature_consistency():
    """检查特征预处理一致性"""
    print("\n" + "=" * 80)
    print("特征预处理一致性检查")
    print("=" * 80)
    
    # 检查所有模型是否都使用preprocess_features和get_feature_columns
    quarter_files = [
        'Quarter/lasso_price_prediction.py',
        'Quarter/predicted_jevons_index_calculator.py',
        'Quarter/predicted_jevons_index_with_error.py',
        'Quarter/lasso_delta_price_change.py',
        'Quarter/lasso_delta_price_change_with_error.py',
    ]
    
    annual_files = [
        'annual/predicted_annual_jevons_index_calculator.py',
        'annual/predicted_annual_jevons_index_with_error.py',
        'annual/lasso_delta_price_change_annual.py',
        'annual/lasso_delta_price_change_annual_with_error.py',
    ]
    
    all_files = quarter_files + annual_files
    
    print("\n检查所有模型是否使用 preprocess_features 和 get_feature_columns:")
    for file_path in all_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            
            uses_preprocess = 'preprocess_features' in content
            uses_get_features = 'get_feature_columns' in content
            
            status = "✓" if (uses_preprocess and uses_get_features) else "✗"
            print(f"  {status} {os.path.basename(file_path)}")
            if not uses_preprocess:
                print(f"    警告: 未使用 preprocess_features")
            if not uses_get_features:
                print(f"    警告: 未使用 get_feature_columns")

if __name__ == '__main__':
    check_quarter_models()
    check_annual_models()
    check_feature_consistency()
    
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("""
所有模型应该使用：
1. random_state=42
2. max_iter=2000
3. cv=min(5, len(data)//2) 或类似的自适应CV
4. preprocess_features() 进行特征预处理
5. get_feature_columns() 获取特征列
6. StandardScaler() 进行特征标准化

如果以上检查都通过，说明所有模型使用一致的Lasso设置。
    """)

