import pandas as pd
import os

from mlxtend.frequent_patterns import apriori, association_rules

def find_association_rules(df, min_support=0.05, metric="lift", min_threshold=1.2):
    """
    Tìm luật kết hợp từ các biến đã được rời rạc hóa.
    """
    # Chỉ lấy các cột đã rời rạc hóa (Categorical)
    cols_for_rules = ['Temp_Class', 'Humidity_Class', 'Visibility_Class', 'Precip Type']
    
    # One-hot encoding để chuyển sang định dạng True/False cho Apriori
    df_encoded = pd.get_dummies(df[cols_for_rules])
    
    # Tìm tập phổ biến
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    
    # Tạo luật
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
        return rules.sort_values('lift', ascending=False)
    return pd.DataFrame()

def compare_rules_by_season(df, config):
    """
    So sánh luật kết hợp giữa các mùa khác nhau.
    """
    seasonal_rules = {}
    for season in config['mining']['target_seasons']:
        season_df = df[df['Season'] == season]
        rules = find_association_rules(
            season_df, 
            min_support=config['mining']['min_support'],
            min_threshold=config['mining']['min_threshold_lift']
        )
        seasonal_rules[season] = rules
    return seasonal_rules


import pandas as pd
import os

def save_seasonal_rules(seasonal_rules, config, base_path="../"):
    """
    Lưu luật vào outputs/tables/ và trả về thống kê.
    base_path: mặc định là '../' vì Notebook nằm trong folder notebooks/
    """
    # Xây dựng đường dẫn đầy đủ
    output_dir = os.path.join(base_path, config['outputs']['mining_dir'])
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    all_rules_list = []
    summary_stats = {}

    for season, rules in seasonal_rules.items():
        if not rules.empty:
            rules_df = rules.copy()
            rules_df['Season'] = season
            all_rules_list.append(rules_df)
            summary_stats[season] = len(rules)
        else:
            summary_stats[season] = 0

    if all_rules_list:
        final_df = pd.concat(all_rules_list, ignore_index=True)
        # Format lại cột để lưu CSV không bị lỗi hiển thị
        final_df['antecedents'] = final_df['antecedents'].apply(lambda x: ', '.join(list(x)))
        final_df['consequents'] = final_df['consequents'].apply(lambda x: ', '.join(list(x)))
        
        save_file = os.path.join(output_dir, config['outputs']['rules_filename'])
        final_df.to_csv(save_file, index=False, encoding='utf-8-sig')
        print(f"Đã lưu danh sách luật tại: {save_file}")
    
    return summary_stats