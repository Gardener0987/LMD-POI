import torch
import os
from tqdm import tqdm
import pandas as pd
import json
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from datetime import datetime
from llm_prompt import LargeModelAugmentor
import csv

def read_and_group_trajectories(file_path):

    df = pd.read_csv(file_path)
    print(f"成功读取数据，共 {len(df)} 条记录。")

    unique_pois = df["POI_id"].unique()
    poi_id_map = {poi: idx + 1 for idx, poi in enumerate(unique_pois)}
    id_to_poi_map = {v: k for k, v in poi_id_map.items()}  # 反向映射
    print(f"共发现 {len(poi_id_map)} 个唯一POI，已建立映射表。")

    def extract_hms(utc_time_str):
        try:
            dt = datetime.strptime(utc_time_str, "%a %b %d %H:%M:%S %z %Y")
            return dt.strftime("%H:%M:%S")
        except Exception:
            return "00:00:00"

    df["time_only"] = df["UTC_time"]
    user_trajectories = {}

    for _, row in df.iterrows():
        user_id = row["user_id"]
        traj_id = row["trajectory_id"]
        poi_mapped = poi_id_map[row["POI_id"]]
        poi_cat = row["POI_catname"]
        time_str = row["time_only"]

        if user_id not in user_trajectories:
            user_trajectories[user_id] = {}
        if traj_id not in user_trajectories[user_id]:
            user_trajectories[user_id][traj_id] = []

        user_trajectories[user_id][traj_id].append((poi_mapped, poi_cat, time_str))


    return user_trajectories, poi_id_map, id_to_poi_map

def traverse_all_trajectories(user_trajectories):

    trajectories_list = []

    for user_id, trajs in user_trajectories.items():
        for traj_id, points in trajs.items():
            trajectory_info = {
                'user_id': user_id,
                'trajectory_id': traj_id,
                'trajectory': points
            }
            trajectories_list.append(trajectory_info)

    return trajectories_list

def load_all_similar_users(json_path, topk=None):

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            similarity_data = json.load(f)
    except Exception as e:
        print(f" 无法读取文件：{e}")
        return {}

    user_similar_map = {}

    for user_id_str, similar_dict in similarity_data.items():
        try:
            user_id = int(user_id_str)
            # 获取相似用户及其分数
            similar_items = list(similar_dict.items())

            # 若指定 topk，按相似度降序取前k个
            if topk is not None:
                similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)[:topk]

            # 转换为整数 ID 列表
            similar_user_ids = [int(uid) for uid, _ in similar_items]
            user_similar_map[user_id] = similar_user_ids

        except ValueError:
            # 某些ID无法转换为整数，跳过
            continue

    return user_similar_map


def merge_augmented_trajectories_(user_trajectories, augmentation_json_path, id_to_poi_map, poi_info_path, output_csv_path="merged_augmented_trajectories.csv"):

    import json
    import csv
    import pandas as pd
    from datetime import datetime


    try:
        with open(augmentation_json_path, "r", encoding="utf-8") as f:
            augmented_data = json.load(f)
        print(f" 成功加载增强数据，共 {len(augmented_data)} 个用户的增强轨迹。")
    except Exception as e:
        print(f" 无法读取增强数据文件：{e}")
        return

 
    try:
        poi_df = pd.read_csv(poi_info_path)
        poi_coord_map = dict(zip(poi_df["POI_id"], zip(poi_df["longitude"], poi_df["latitude"])))
        print(f" 已加载 {len(poi_coord_map)} 个POI的经纬度信息。")
    except Exception as e:
        print(f" 加载POI经纬度失败：{e}")
        return

    merged_rows = []
    augmented_user_count = 0
    original_user_count = 0

    def parse_time(t):
        try:
            return datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None


    for user_id, traj_dict in user_trajectories.items():
        user_id_str = str(user_id)
        

        if user_id_str in augmented_data:
            augmented_user_count += 1
            user_aug_data = augmented_data[user_id_str]
            
            for traj_id, original_traj in traj_dict.items():
                traj_id_str = str(traj_id)
                
                if traj_id_str in user_aug_data:
                    try:
                        aug_pois = eval(user_aug_data[traj_id_str])
                    except Exception as e:
                        print(f" 跳过解析失败的增强轨迹 user:{user_id} traj:{traj_id}")
                        aug_pois = []
                    
          
                    combined_traj = original_traj + aug_pois
                    
              
                    try:
                        combined_traj_sorted = sorted(combined_traj, key=lambda x: parse_time(x[2]))
                    except Exception as e:
                        print(f" 排序失败：{user_id}:{traj_id} - {e}")
                        combined_traj_sorted = original_traj  # 排序失败时使用原始轨迹
                else:
          
                    combined_traj_sorted = original_traj
        else:
         
            original_user_count += 1
            for traj_id, original_traj in traj_dict.items():
                combined_traj_sorted = original_traj
        
    
        for traj_id, trajectory in traj_dict.items():
       
            user_id_str = str(user_id)
            traj_id_str = str(traj_id)
            
            if user_id_str in augmented_data and traj_id_str in augmented_data[user_id_str]:
                try:
                    aug_pois = eval(augmented_data[user_id_str][traj_id_str])
                    combined_traj = trajectory + aug_pois
                    try:
                        combined_traj_sorted = sorted(combined_traj, key=lambda x: parse_time(x[2]))
                    except Exception:
                        combined_traj_sorted = trajectory
                except Exception:
                    combined_traj_sorted = trajectory
            else:
                combined_traj_sorted = trajectory
            
     

            for poi_id_mapped, poi_cat, time_str in combined_traj_sorted:
              
                poi_id = id_to_poi_map.get(poi_id_mapped, f"unknown({poi_id_mapped})")
                lon, lat = poi_coord_map.get(poi_id, ("", ""))
                merged_rows.append([user_id, poi_id, poi_cat, time_str, traj_id, lon, lat])


    try:
        with open(output_csv_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "POI_id", "POI_catname", "UTC_time", "trajectory_id", "longitude", "latitude"])
            writer.writerows(merged_rows)
        
        print(f" 合并成功，已写入文件：{output_csv_path}")
        print(f" 统计信息:")
        print(f"   - 总用户数: {len(user_trajectories)}")
        print(f"   - 应用增强的用户数: {augmented_user_count}")
        print(f"   - 保持原始的用户数: {original_user_count}")
        print(f"   - 总轨迹点数: {len(merged_rows)}")
        
    except Exception as e:
        print(f" 写入CSV失败：{e}")


def clean_trajectories_with_denoise_1(user_trajectories, denoise_json_path, id_to_poi_map, poi_info_path, output_csv_path="cleaned_trajectories.csv"):

    import json
    import csv
    import pandas as pd
    from datetime import datetime


    try:
        with open(denoise_json_path, "r", encoding="utf-8") as f:
            denoise_data = json.load(f)
        print(f" 成功加载去噪数据，共 {len(denoise_data)} 个用户的去噪结果。")
    except Exception as e:
        print(f" 无法读取去噪数据文件：{e}")
        return


    try:
        poi_df = pd.read_csv(poi_info_path)
        poi_coord_map = dict(zip(poi_df["POI_id"], zip(poi_df["longitude"], poi_df["latitude"])))
        print(f" 已加载 {len(poi_coord_map)} 个POI的经纬度信息。")
    except Exception as e:
        print(f" 加载POI经纬度失败：{e}")
        return

    cleaned_rows = []
    removed_count = 0
    total_count = 0
    cleaned_user_count = 0
    original_user_count = 0
    skipped_trajectories_count = 0 


    for user_id, traj_dict in user_trajectories.items():
        user_id_str = str(user_id)
        
        if user_id_str in denoise_data:
            cleaned_user_count += 1
            user_denoise_data = denoise_data[user_id_str]
            
            for traj_id, original_traj in traj_dict.items():
                traj_id_str = str(traj_id)
                total_count += len(original_traj)
                
                if traj_id_str in user_denoise_data:
                    try:
                        noise_points = eval(user_denoise_data[traj_id_str]) if user_denoise_data[traj_id_str] != "[]" else []
                    except Exception as e:
                        print(f" 解析噪声点失败 user:{user_id} traj:{traj_id}: {e}")
                        noise_points = []
                    
                    noise_count = 0
                    for poi_point in original_traj:
                        poi_id_mapped, poi_cat, time_str = poi_point
                        
                        for noise_poi_id, noise_poi_cat, noise_time in noise_points:
                            if (poi_id_mapped == noise_poi_id and 
                                poi_cat == noise_poi_cat and 
                                time_str == noise_time):
                                noise_count += 1
                                break
                    
                    remaining_count = len(original_traj) - noise_count
                    
                    if remaining_count <= 1:
                        skipped_trajectories_count += 1
                        for poi_point in original_traj:
                            poi_id_mapped, poi_cat, time_str = poi_point
                            poi_id = id_to_poi_map.get(poi_id_mapped, f"unknown({poi_id_mapped})")
                            lon, lat = poi_coord_map.get(poi_id, ("", ""))
                            cleaned_rows.append([user_id, poi_id, poi_cat, time_str, traj_id, lon, lat])
                    else:

                        for poi_point in original_traj:
                            poi_id_mapped, poi_cat, time_str = poi_point
                            poi_id = id_to_poi_map.get(poi_id_mapped, f"unknown({poi_id_mapped})")
                            

                            is_noise = False
                            for noise_poi_id, noise_poi_cat, noise_time in noise_points:
                                if (poi_id_mapped == noise_poi_id and 
                                    poi_cat == noise_poi_cat and 
                                    time_str == noise_time):
                                    is_noise = True
                                    removed_count += 1
                                    break
                            

                            if not is_noise:
                                lon, lat = poi_coord_map.get(poi_id, ("", ""))
                                cleaned_rows.append([user_id, poi_id, poi_cat, time_str, traj_id, lon, lat])
                else:

                    for poi_point in original_traj:
                        poi_id_mapped, poi_cat, time_str = poi_point
                        poi_id = id_to_poi_map.get(poi_id_mapped, f"unknown({poi_id_mapped})")
                        lon, lat = poi_coord_map.get(poi_id, ("", ""))
                        cleaned_rows.append([user_id, poi_id, poi_cat, time_str, traj_id, lon, lat])
        else:

            original_user_count += 1
            for traj_id, original_traj in traj_dict.items():
                total_count += len(original_traj)
                for poi_point in original_traj:
                    poi_id_mapped, poi_cat, time_str = poi_point
                    poi_id = id_to_poi_map.get(poi_id_mapped, f"unknown({poi_id_mapped})")
                    lon, lat = poi_coord_map.get(poi_id, ("", ""))
                    cleaned_rows.append([user_id, poi_id, poi_cat, time_str, traj_id, lon, lat])


    try:
        with open(output_csv_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "POI_id", "POI_catname", "UTC_time", "trajectory_id", "longitude", "latitude"])
            writer.writerows(cleaned_rows)
        
        print(f" 清洗成功，已写入文件：{output_csv_path}")
        print(f" 统计信息:")
        print(f"   - 总用户数: {len(user_trajectories)}")
        print(f"   - 应用去噪的用户数: {cleaned_user_count}")
        print(f"   - 保持原始的用户数: {original_user_count}")
        print(f"   - 原始轨迹点数: {total_count}")
        print(f"   - 移除噪声点数: {removed_count}")
        print(f"   - 清洗后轨迹点数: {len(cleaned_rows)}")
        print(f"   - 因轨迹过短跳过的轨迹数: {skipped_trajectories_count}")
        print(f"   - 清洗比例: {removed_count/total_count*100:.2f}%" if total_count > 0 else "   - 清洗比例: 0%")
        
    except Exception as e:
        print(f" 写入CSV失败：{e}")

    return cleaned_rows

def main():

    start_datetime = datetime.now()
    start_time = time.perf_counter()
    print(" start time：", start_datetime.strftime("%Y-%m-%d %H:%M:%S"))

    file_path = "code/data/NYC_init.csv"  
    # file_path = "code/data/NYC_denoise.csv"  

    user_trajectories, poi_id_map, id_to_poi_map = read_and_group_trajectories(file_path)
    all_trajectories = traverse_all_trajectories(user_trajectories)
    poi_info_path = "code/data/poi_info.csv"
    all_similar_users = load_all_similar_users("code/user_sim.json",topk=4)
    augmentor = LargeModelAugmentor(
        model_path ="llm/DeepSeek-14B",
        device="cuda:4"
        )
    # step1:denoising
    noise_results = augmentor.batch_denoise_trajectories(
        user_trajectories,
        save_path="denoise.json"
    )

    # denoising file
    clean_trajectories_with_denoise_1(
        user_trajectories=user_trajectories,
        denoise_json_path="denoise.json",
        id_to_poi_map=id_to_poi_map,
        poi_info_path=poi_info_path,
        output_csv_path="code/NYC_denoise.csv"
    )
    # step2:augmentation
    # aug_results = augmentor.augment_all_users(
    #     user_trajectories,
    #     all_trajectories,
    #     all_similar_users,
    #     topk=2,
    #     save_path="augment.json"
    # )
    # augmentation file
    # merge_augmented_trajectories_(
    #     user_trajectories=user_trajectories,
    #     augmentation_json_path="augment.json",
    #     id_to_poi_map=id_to_poi_map,
    #     poi_info_path=poi_info_path,
    #     output_csv_path="NYC_train.csv"
    # )

    end_datetime = datetime.now()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print("\n end time：", end_datetime.strftime("%Y-%m-%d %H:%M:%S"))
    print(" from {} to {}".format(
        start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        end_datetime.strftime("%Y-%m-%d %H:%M:%S")
    ))

if __name__ == "__main__":
    main()
