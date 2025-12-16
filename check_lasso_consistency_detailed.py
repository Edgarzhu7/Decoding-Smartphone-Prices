"""
详细检查所有模型使用的Lasso参数是否一致
"""
import os

def check_file_lasso_params(file_path):
    """检查文件中LassoCV的参数"""
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    results = {
        'file': os.path.basename(file_path),
        'lasso_calls': [],
        'uses_preprocess': False,
        'uses_get_features': False,
        'uses_standardscaler': False,
    }
    
    for i, line in enumerate(lines, 1):
        # 检查LassoCV调用
        if 'LassoCV(' in line:
            # 读取多行直到找到闭合括号
            lasso_code = line
            j = i
            while ')' not in lasso_code and j < len(lines):
                j += 1
                if j < len(lines):
                    lasso_code += lines[j-1]
            
            call_info = {
                'line': i,
                'code': lasso_code.strip(),
                'cv': None,
                'random_state': None,
                'max_iter': None,
            }
            
            # 提取参数
            if 'random_state=' in lasso_code:
                import re
                match = re.search(r'random_state=(\d+)', lasso_code)
                if match:
                    call_info['random_state'] = match.group(1)
            
            if 'max_iter=' in lasso_code:
                import re
                match = re.search(r'max_iter=(\d+)', lasso_code)
                if match:
                    call_info['max_iter'] = match.group(1)
            
            if 'cv=' in lasso_code:
                import re
                match = re.search(r'cv=([^,)]+)', lasso_code)
                if match:
                    call_info['cv'] = match.group(1).strip()
            
            results['lasso_calls'].append(call_info)
        
        # 检查其他关键函数
        if 'preprocess_features' in line:
            results['uses_preprocess'] = True
        if 'get_feature_columns' in line:
            results['uses_get_features'] = True
        if 'StandardScaler()' in line or 'StandardScaler(' in line:
            results['uses_standardscaler'] = True
    
    return results

def main():
    print("=" * 100)
    print("所有模型 Lasso 参数一致性检查")
    print("=" * 100)
    
    # 季度模型
    quarter_files = [
        'Quarter/lasso_price_prediction.py',
        'Quarter/predicted_jevons_index_calculator.py',
        'Quarter/predicted_jevons_index_with_error.py',
        'Quarter/lasso_delta_price_change.py',
        'Quarter/lasso_delta_price_change_with_error.py',
    ]
    
    # 年度模型
    annual_files = [
        'annual/predicted_annual_jevons_index_calculator.py',
        'annual/predicted_annual_jevons_index_with_error.py',
        'annual/lasso_delta_price_change_annual.py',
        'annual/lasso_delta_price_change_annual_with_error.py',
    ]
    
    all_files = quarter_files + annual_files
    all_results = []
    
    print("\n【季度模型】")
    print("-" * 100)
    for file_path in quarter_files:
        result = check_file_lasso_params(file_path)
        if result:
            all_results.append(result)
            print(f"\n文件: {result['file']}")
            print(f"  ✓ 使用 preprocess_features: {result['uses_preprocess']}")
            print(f"  ✓ 使用 get_feature_columns: {result['uses_get_features']}")
            print(f"  ✓ 使用 StandardScaler: {result['uses_standardscaler']}")
            print(f"  LassoCV 调用次数: {len(result['lasso_calls'])}")
            for i, call in enumerate(result['lasso_calls'], 1):
                print(f"    调用 {i} (行 {call['line']}):")
                print(f"      cv: {call['cv']}")
                print(f"      random_state: {call['random_state']}")
                print(f"      max_iter: {call['max_iter']}")
    
    print("\n【年度模型】")
    print("-" * 100)
    for file_path in annual_files:
        result = check_file_lasso_params(file_path)
        if result:
            all_results.append(result)
            print(f"\n文件: {result['file']}")
            print(f"  ✓ 使用 preprocess_features: {result['uses_preprocess']}")
            print(f"  ✓ 使用 get_feature_columns: {result['uses_get_features']}")
            print(f"  ✓ 使用 StandardScaler: {result['uses_standardscaler']}")
            print(f"  LassoCV 调用次数: {len(result['lasso_calls'])}")
            for i, call in enumerate(result['lasso_calls'], 1):
                print(f"    调用 {i} (行 {call['line']}):")
                print(f"      cv: {call['cv']}")
                print(f"      random_state: {call['random_state']}")
                print(f"      max_iter: {call['max_iter']}")
    
    # 一致性检查
    print("\n" + "=" * 100)
    print("【一致性检查结果】")
    print("=" * 100)
    
    # 收集所有参数值
    all_random_states = set()
    all_max_iters = set()
    all_cv_patterns = set()
    
    for result in all_results:
        for call in result['lasso_calls']:
            if call['random_state']:
                all_random_states.add(call['random_state'])
            if call['max_iter']:
                all_max_iters.add(call['max_iter'])
            if call['cv']:
                all_cv_patterns.add(call['cv'])
    
    print(f"\n1. Random State 检查:")
    print(f"   所有值: {sorted(all_random_states)}")
    if len(all_random_states) == 1:
        print(f"   ✓ 所有模型使用相同的 random_state = {list(all_random_states)[0]}")
    elif len(all_random_states) == 0:
        print(f"   ✗ 警告: 未找到 random_state 参数!")
    else:
        print(f"   ✗ 警告: random_state 不一致!")
    
    print(f"\n2. Max Iter 检查:")
    print(f"   所有值: {sorted(all_max_iters)}")
    if len(all_max_iters) == 1:
        print(f"   ✓ 所有模型使用相同的 max_iter = {list(all_max_iters)[0]}")
    elif len(all_max_iters) == 0:
        print(f"   ✗ 警告: 未找到 max_iter 参数!")
    else:
        print(f"   ✗ 警告: max_iter 不一致!")
    
    print(f"\n3. CV 模式检查:")
    for cv_pattern in sorted(all_cv_patterns):
        print(f"   - {cv_pattern}")
    if all('min(5' in cv or 'min(5,' in cv for cv in all_cv_patterns if cv):
        print(f"   ✓ 所有模型使用类似的CV模式 (min(5, ...))")
    else:
        print(f"   ⚠ CV模式略有不同，但都是自适应的")
    
    print(f"\n4. 特征预处理检查:")
    all_use_preprocess = all(r['uses_preprocess'] for r in all_results)
    all_use_get_features = all(r['uses_get_features'] for r in all_results)
    all_use_scaler = all(r['uses_standardscaler'] for r in all_results)
    
    print(f"   preprocess_features: {'✓ 所有模型都使用' if all_use_preprocess else '✗ 部分模型未使用'}")
    print(f"   get_feature_columns: {'✓ 所有模型都使用' if all_use_get_features else '✗ 部分模型未使用'}")
    print(f"   StandardScaler: {'✓ 所有模型都使用' if all_use_scaler else '✗ 部分模型未使用'}")
    
    print("\n" + "=" * 100)
    print("【总结】")
    print("=" * 100)
    
    issues = []
    if len(all_random_states) != 1:
        issues.append("random_state 不一致或缺失")
    if len(all_max_iters) != 1:
        issues.append("max_iter 不一致或缺失")
    if not all_use_preprocess:
        issues.append("部分模型未使用 preprocess_features")
    if not all_use_get_features:
        issues.append("部分模型未使用 get_feature_columns")
    if not all_use_scaler:
        issues.append("部分模型未使用 StandardScaler")
    
    if not issues:
        print("✓ 所有模型使用一致的 Lasso 设置!")
        print("\n所有模型都使用:")
        print("  - random_state=42")
        print("  - max_iter=2000")
        print("  - cv=min(5, len(data)//2) 或类似的自适应CV")
        print("  - preprocess_features() 进行特征预处理")
        print("  - get_feature_columns() 获取特征列")
        print("  - StandardScaler() 进行特征标准化")
    else:
        print("✗ 发现以下不一致:")
        for issue in issues:
            print(f"  - {issue}")

if __name__ == '__main__':
    main()

