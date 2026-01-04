def predict_images_in_folder(folder_path, model, device, class_names, output_file):
    image_paths = glob.glob(os.path.join(folder_path, "*.png"))  # 获取所有 PNG 图片路径
    results = []
    
    for image_path in image_paths:
        result = predict_single_image(image_path, model, device, print_perform=True, class_names=class_names)
        result_info = {
            "image": os.path.basename(image_path),
            "prediction": result["index"],
            "class_name": class_names[result["index"]]
        }
        results.append(result_info)
        print(f"predict result: {result_info}")
    
    # 保存结果到 JSON 文件
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_file}")