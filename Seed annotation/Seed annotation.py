import pandas as pd
import numpy as np
import warnings
import spectral_entropy
import tools

warnings.filterwarnings('ignore')

# ==================== 参数设置 ====================
ppm_tolerance = 5e-6  # 5 ppm
rt_tolerance = 0.2  # 0.2分钟
min_intensity = 0.01
ms2_ppm = 10
ms2_da = 0.01
spectral_similarity_threshold = 0.7

# ==================== 数据读取 ====================
print("正在读取数据...")
# 读取CSV文件（包含name, m/z, rt, MS2）
csv_data = pd.read_csv(r"C:\Users\ZYJ\Desktop\机器学习文章\机器学习\分子网络\library.csv")

# 读取Excel文件（包含m/z, rt, MS2）
excel_data = pd.read_excel(r"Z:\张宇杰 赵春霞阅\machine learning for risk substances\其他数据\样本筛查数据\筛查数据\95%正离子\screen_result-95%-pos.xlsx")

print(f"CSV文件包含 {len(csv_data)} 个条目")
print(f"Excel文件包含 {len(excel_data)} 个条目")

print("数据将按原始顺序处理（不排序）")


# ==================== 函数定义 ====================

def parse_ms2_standard(ms2_string):
    """解析MS2数据：53.03934 0.107;77.03918 0.016;..."""
    if pd.isna(ms2_string) or ms2_string == '':
        return np.array([])

    peaks = []
    for peak in str(ms2_string).split(';'):
        if peak.strip():
            parts = peak.strip().split()
            if len(parts) == 2:
                mz = float(parts[0])
                intensity = float(parts[1])
                peaks.append([mz, intensity])

    return np.array(peaks, dtype=np.float32) if peaks else np.array([])


def clean_ms2_spectrum(ms2_array, precursor_mz):
    """清洗MS2谱图"""
    if ms2_array.size == 0:
        return np.array([])

    try:
        cleaned_array = tools.clean_spectrum(
            ms2_array,
            max_mz=precursor_mz,
            noise_removal=0.01,
            ms2_ppm=ms2_ppm
        )
        return cleaned_array
    except:
        return ms2_array


def calculate_spectral_entropy_similarity(spec_query, spec_reference):
    """计算谱熵相似度"""
    if spec_query.size == 0 or spec_reference.size == 0:
        return 0

    try:
        similarity = spectral_entropy.calculate_entropy_similarity(
            spec_query,
            spec_reference,
            ms2_da=ms2_da
        )
        return similarity
    except:
        return 0


# ==================== 处理数据 ====================
print("\n正在处理数据...")

# 处理CSV数据
csv_processed = []
for idx, row in csv_data.iterrows():
    ms2_array = parse_ms2_standard(row['MS2'])
    cleaned_array = clean_ms2_spectrum(ms2_array, row['m/z'])
    csv_processed.append({
        'name': row['Name'],
        'mz': row['m/z'],
        'rt': row['rt'],
        'ms2_array': cleaned_array
    })
csv_processed_df = pd.DataFrame(csv_processed)
print(f"CSV数据处理完成，共 {len(csv_processed_df)} 个条目")

# 处理Excel数据
excel_processed = []
for idx, row in excel_data.iterrows():
    ms2_array = parse_ms2_standard(row['MSMS'])
    cleaned_array = clean_ms2_spectrum(ms2_array, row['PEPMASS'])
    excel_processed.append({
        'index': idx,
        'mz': row['PEPMASS'],
        'rt': row['RT'],
        'ms2_array': cleaned_array
    })
excel_processed_df = pd.DataFrame(excel_processed)
print(f"Excel数据处理完成，共 {len(excel_processed_df)} 个条目")

# ==================== 匹配过程 ====================
print("\n开始匹配...")

# 初始化结果列
excel_data['Matched_Name'] = ''
excel_data['CL'] = 0
excel_data['Similarity_Score'] = 0.0

matched_count = 0
cl1_count = 0
cl2_count = 0
low_similarity_count = 0  # 统计m/z和RT符合但相似度<0.7的情况

# 为每个Excel条目寻找匹配
for excel_idx, excel_row in excel_processed_df.iterrows():
    excel_mz = excel_row['mz']
    excel_rt = excel_row['rt']
    excel_ms2 = excel_row['ms2_array']
    original_idx = excel_row['index']

    best_score = 0
    best_name = ''
    best_cl = 0
    best_rt_note = ''

    # 遍历所有CSV条目寻找匹配
    for _, csv_row in csv_processed_df.iterrows():
        # 检查m/z是否在容忍范围内
        mz_diff = abs(excel_mz - csv_row['mz'])
        if mz_diff > csv_row['mz'] * ppm_tolerance:
            continue

        # 检查CSV的RT是否为空
        csv_rt_is_empty = pd.isna(csv_row['rt']) or csv_row['rt'] == ''

        # 检查RT是否匹配（如果CSV的RT不为空）
        rt_matched = False
        if csv_rt_is_empty:
            rt_matched = True  # RT为空视为可匹配
        else:
            if abs(excel_rt - csv_row['rt']) <= rt_tolerance:
                rt_matched = True

        if not rt_matched:
            continue

        # 计算相似度
        score = calculate_spectral_entropy_similarity(excel_ms2, csv_row['ms2_array'])

        # 如果相似度大于当前最佳，更新最佳匹配
        if score > best_score:
            best_score = score
            best_name = csv_row['name']
            if score >= spectral_similarity_threshold:
                best_cl = 2 if csv_rt_is_empty else 1
            else:
                best_cl = 0  # 相似度低于阈值，CL保持0

    # 记录结果 - 只要有匹配（m/z和RT符合）就记录name和score
    if best_name:  # 找到匹配
        excel_data.at[original_idx, 'Matched_Name'] = best_name
        excel_data.at[original_idx, 'Similarity_Score'] = round(best_score, 4)
        excel_data.at[original_idx, 'CL'] = best_cl

        if best_cl > 0:
            matched_count += 1
            if best_cl == 1:
                cl1_count += 1
            else:
                cl2_count += 1
        else:
            low_similarity_count += 1  # 相似度低于阈值但有匹配

    # 显示进度
    if (excel_idx + 1) % 100 == 0:
        print(f"已处理 {excel_idx + 1}/{len(excel_processed_df)} 个条目")


# ==================== 保存结果 ====================
output_file = '种子注释95%正离子.xlsx'
excel_data.to_excel(output_file, index=False)
print(f"\n结果已保存至: {output_file}")

