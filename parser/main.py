import os
import csv
import json
from utils.ppt_extract import extract_images_from_ppt
from utils.hash_utils import load_template_hashes, classify_image

# 路径配置
UPLOAD_DIR = 'data'          # 待处理PPT目录
IMAGE_DIR = 'extracted_images'  # 提取的图片目录
TEMPLATE_DIR = 'hash_templates' # 模板目录
OUTPUT_CSV = 'outputs/result.csv' # 结果输出
DEBUG_DIST_DIR = 'debug/distances' # 距离数据目录
CROP_SIZE = (120, 120)      # 裁剪尺寸（左上角区域）
TARGET_RESIZE = (800, 600)  # 测试图片先缩放到该尺寸（统一Logo位置）


def main():
    os.makedirs('debug', exist_ok=True)
    os.makedirs(DEBUG_DIST_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    # 加载模板（每个类别最多5个，不缩放/裁剪）
    print("===== 加载Logo模板 =====")
    templates = load_template_hashes(TEMPLATE_DIR, max_templates_per_category=5)
    if not templates:
        print("警告：未加载到有效模板！结果均为 unknown")
        return

    # 类别参数（根据实际情况调整）
    category_thresholds = {'wall': 32, 'dragon': 32, 'ball': 30}
    category_min_matches = {'wall': 1, 'dragon': 1, 'ball': 1}
    secondary_check_ratios = {'wall': 0.9, 'dragon': 0.75, 'ball': 0.8}

    results = []

    # 处理PPT文件
    print("\n===== 开始处理PPT =====")
    if not os.path.exists(UPLOAD_DIR):
        print(f"错误：{UPLOAD_DIR} 不存在")
        return

    ppt_files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith('.pptx')]
    if not ppt_files:
        print(f"提示：{UPLOAD_DIR} 中无PPT文件（.pptx）")
        return

    for ppt_file in ppt_files:
        ppt_path = os.path.join(UPLOAD_DIR, ppt_file)
        print(f"\n----- 处理PPT：{ppt_file} -----")

        # 提取图片
        extract_count = extract_images_from_ppt(ppt_path, IMAGE_DIR)
        if extract_count == 0:
            print("  未提取到图片，跳过")
            continue

        # 识别图片
        print(f"  开始识别 {extract_count} 张图片...")
        ppt_prefix = os.path.splitext(ppt_file)[0]

        for image_name in os.listdir(IMAGE_DIR):
            if not image_name.startswith(ppt_prefix):
                continue

            image_path = os.path.join(IMAGE_DIR, image_name)
            # 调用分类函数（传递缩放尺寸）
            label, best_dist, top5 = classify_image(
                image_path,
                templates,
                category_thresholds=category_thresholds,
                category_min_matches=category_min_matches,
                secondary_check_ratios=secondary_check_ratios,
                crop_size=CROP_SIZE,
                target_resize=TARGET_RESIZE  # 关键：先缩放再裁剪
            )

            # 保存距离数据（调试）
            debug_data = {
                'image': image_name,
                'best_label': label,
                'best_distance': float(best_dist),
                'ranking': [(cat, float(dist)) for cat, dist in top5]
            }
            debug_file = os.path.join(DEBUG_DIST_DIR, f"{os.path.splitext(image_name)[0]}.json")
            with open(debug_file, 'w', encoding='utf-8') as f:
                json.dump(debug_data, f, indent=2, ensure_ascii=False)

            # 打印结果
            print(f"图片：{image_name}")
            print(f"  最佳匹配：{label}（距离 {best_dist}）")
            print("  距离排行：")
            for tname, dist in top5[:5]:
                print(f"    {tname}: {dist}")

            results.append([ppt_file, image_name, label, best_dist])

    # 保存结果到CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['PPT文件名', '图片名称', '分类结果', '最佳匹配距离'])
        writer.writerows(results)

    print(f"\n===== 处理完成 =====")
    print(f"结果已保存至：{OUTPUT_CSV}")
    print(f"预处理图像保存至：debug/processed/")
    print(f"距离数据保存至：{DEBUG_DIST_DIR}/")
    print(f"使用的缩放尺寸：{TARGET_RESIZE[0]}x{TARGET_RESIZE[1]}（缩放后裁剪 {CROP_SIZE[0]}x{CROP_SIZE[1]}）")


if __name__ == '__main__':
    main()