from PIL import Image
import imagehash
import os


def crop_top_left(img, crop_size=(100, 100)):
    """裁剪图片左上角区域"""
    width, height = img.size
    crop_width = min(crop_size[0], width)
    crop_height = min(crop_size[1], height)
    return img.crop((0, 0, crop_width, crop_height))


def resize_keep_aspect_ratio(img, target_size, bg_color=(255, 255, 255)):
    """保持宽高比缩放，左/上对齐，空白填充背景色（兼容旧版本Pillow）"""
    original_width, original_height = img.size
    target_width, target_height = target_size

    # 计算缩放比例（取最小比例，避免超出目标尺寸）
    scale_ratio = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale_ratio)
    new_height = int(original_height * scale_ratio)

    # 兼容旧版本：使用Image.LANCZOS替代ImageResampling.LANCZOS
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    # 创建目标尺寸的画布，填充背景色
    canvas = Image.new("RGB", target_size, bg_color)
    # 左对齐、上对齐（确保Logo在裁剪区域内）
    canvas.paste(resized_img, (0, 0))
    return canvas


def compute_hashes_from_image(img, preprocess=False, crop=False, crop_size=(100, 100)):
    """从内存Image对象计算哈希（仅裁剪，无预处理）"""
    if crop:
        img = crop_top_left(img, crop_size)
    # 仅转灰度+resize（无滤波/阈值，保留原始信息）
    processed_img = img.convert('L').resize((256, 256))
    return {
        'phash': imagehash.phash(processed_img),
        'dhash': imagehash.dhash(processed_img)
    }


def compute_hashes(image_path, preprocess=False, crop=False, crop_size=(100, 100)):
    """从文件计算哈希"""
    with Image.open(image_path) as img:
        return compute_hashes_from_image(img, preprocess, crop, crop_size)


def load_template_hashes(template_dir, max_templates_per_category=5):
    """加载模板（不缩放、不裁剪、不预处理）"""
    templates = {}
    if not os.path.exists(template_dir):
        print(f"警告：模板目录不存在 - {template_dir}")
        return templates

    for category in os.listdir(template_dir):
        category_path = os.path.join(template_dir, category)
        if not os.path.isdir(category_path):
            continue

        templates[category] = {'phash': [], 'dhash': []}
        loaded_count = 0

        for fname in os.listdir(category_path):
            if loaded_count >= max_templates_per_category:
                break
            fpath = os.path.join(category_path, fname)
            if not (fpath.lower().endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(fpath)):
                continue

            try:
                with Image.open(fpath) as img:
                    # 模板直接用原始尺寸，不裁剪、不预处理
                    hashes = compute_hashes_from_image(img, preprocess=False, crop=False)
                    templates[category]['phash'].append(hashes['phash'])
                    templates[category]['dhash'].append(hashes['dhash'])
                    loaded_count += 1
                    print(f"加载模板：{category}/{fname}（{loaded_count}/{max_templates_per_category}）")
            except Exception as e:
                print(f"加载模板失败 {fpath}：{str(e)}")

        if not templates[category]['phash']:
            del templates[category]
            print(f"警告：类别 {category} 无有效模板，已忽略")
        else:
            print(f"类别 {category} 共加载 {loaded_count} 个模板")

    print(f"共加载 {len(templates)} 个有效类别")
    return templates


def classify_image(image_path, templates,
                   category_thresholds={'wall': 32, 'dragon': 32, 'ball': 30},
                   category_min_matches={'wall': 1, 'dragon': 1, 'ball': 1},
                   secondary_check_ratios={'wall': 0.9, 'dragon': 0.75, 'ball': 0.8},
                   crop_size=(120, 120),
                   target_resize=(800, 600)):  # 先缩放到该尺寸
    """测试图片处理：先缩放→裁剪→哈希计算（已移除二次校验）"""
    try:
        with Image.open(image_path) as img:
            debug_dir = os.path.join('debug', 'processed')
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, os.path.basename(image_path))

            # 步骤1：统一缩放到目标尺寸（左/上对齐，填充白色）
            img = resize_keep_aspect_ratio(img, target_resize)

            # 步骤2：裁剪左上角固定区域
            img_cropped = crop_top_left(img, crop_size)

            # 步骤3：转灰度+resize到哈希尺寸（256×256）
            processed_img = img_cropped.convert('L').resize((256, 256))
            processed_img.save(debug_path)  # 保存调试图

            # 计算哈希（无额外预处理）
            hashes = compute_hashes_from_image(processed_img, preprocess=False, crop=False)
            phash_val = hashes['phash']
            dhash_val = hashes['dhash']

    except Exception as e:
        print(f"处理图像失败 {image_path}：{str(e)}")
        return "error", float('inf'), []

    category_analysis = []
    for category in templates:
        p_templates = templates[category]['phash']
        d_templates = templates[category]['dhash']
        if not p_templates or not d_templates:
            continue

        # 计算哈希距离（PHash + DHash 平均）
        p_distances = [phash_val - pt for pt in p_templates]
        d_distances = [dhash_val - dt for dt in d_templates]
        combined_distances = [(p + d) / 2 for p, d in zip(p_distances, d_distances)]

        min_dist = min(combined_distances)
        avg_dist = sum(combined_distances) / len(combined_distances)
        category_threshold = category_thresholds.get(category, 30)
        match_count = sum(1 for d in combined_distances if d <= category_threshold)

        category_analysis.append({
            'category': category,
            'min_dist': min_dist,
            'avg_dist': avg_dist,
            'match_count': match_count,
            'threshold': category_threshold,
            'distances': combined_distances
        })

        print(f"\n类别 {category} 分析:")
        print(f"  阈值: {category_threshold}")
        print(f"  最小距离: {min_dist}")
        print(f"  平均距离: {avg_dist:.2f}")
        print(f"  符合阈值的模板数量: {match_count}/{len(combined_distances)}")

    # 【核心修改】移除二次校验，直接使用所有类别进行排序
    valid_categories = category_analysis  # 不再过滤，全部参与排序

    # 按距离排序（最小距离优先，平均距离次之）
    valid_categories.sort(key=lambda x: (x['min_dist'], x['avg_dist']))
    category_ranking = [(ca['category'], ca['min_dist']) for ca in valid_categories]

    if valid_categories:
        best = valid_categories[0]
        # 仅用最小距离与阈值比较（不再参考二次校验）
        return (best['category'], best['min_dist'], category_ranking) if best['min_dist'] <= best['threshold'] else (
            "unknown", best['min_dist'], category_ranking)
    else:
        return "unknown", float('inf'), []
