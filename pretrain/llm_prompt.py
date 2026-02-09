import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime
import re
import ast
import time
class LargeModelAugmentor:
    def __init__(self,model_path, device="cuda:1", max_new_tokens=4096):
        print(f"Loading model from {model_path} on {device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            # device_map="auto",
            device_map=device
        )
        self.max_new_tokens = max_new_tokens
        print("Model loaded successfully.")
    

    def _build_prompt(self, current_traj, similar_trajs, all_trajs):

        all_poi_ids = set()
        for traj in all_trajs:
            for poi_id, _, _ in traj:
                all_poi_ids.add(poi_id)


        for user_trajs in similar_trajs.values():
            for traj in user_trajs.values():
                for poi_id, _, _ in traj:
                    all_poi_ids.add(poi_id)


        prompt = f"""
        1.Analyze the user’s historical trajectory {all_trajs} to summarize long-term behavioral patterns and historical preference characteristics.
        2.Analyze similar users’ trajectory {similar_trajs} to characterize the behavioral preferences of the similar users group.
        3.Based on the integrated evidence from the user’s historical preferences and the behavioral tendencies of similar users, 
        evaluate the plausibility of the current user trajectory {current_traj} to determine whether trajectory enhancement and completion are required. 
        4.If potential missing check-ins are identified within the trajectory, generate and output two augmented POI instances that are plausible in terms of both POI category and check-in time; otherwise, return an empty list.
        
        Output: [(POI_id1,POI_catname1,Time1),(POI_id2,POI_catname2,Time2)] or []
        """

        return prompt.strip()
            
    def infer(self, current_traj, similar_trajs, all_trajs):

        prompt = self._build_prompt(current_traj, similar_trajs, all_trajs)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.6,
                top_p=0.95,
                do_sample=True
            )[0][len(inputs.input_ids[0]):].tolist()

        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        print(f"Model output: {output_text}")

        if "[]" in output_text:
            result = "[]"
        else:
            pattern = r"\[\s*(?:\(\s*\d+\s*,\s*'[^']+'\s*,\s*'[^']+'\s*\)\s*,?\s*)+\]"
            matches = re.findall(pattern, output_text)
            if matches:
                result = matches[-1].strip()
            else:
                result = "[]"
        print(f"Model output clean:--> {result}")
        return result

    def augment_user(self, user_id, user_trajectories, all_trajectories, all_similar_users, topk=2):
 
        if user_id not in user_trajectories:
            print(f"用户 {user_id} 不存在。")
            return {}

        enhanced_results = {}
        similar_user_ids = all_similar_users.get(user_id, [])[:topk]

        similar_trajs = {}
        for sim_uid in similar_user_ids:
            user_trajs = user_trajectories.get(sim_uid, {})
            limited_trajs = dict(list(user_trajs.items())[-5:])
            similar_trajs[sim_uid] = limited_trajs

        user_all_trajs = user_trajectories[user_id]

        for traj_id, traj_points in user_trajectories[user_id].items():
            start_datetime = datetime.now()
            start_time = time.perf_counter()
            print("程序运行时间：", start_datetime.strftime("%Y-%m-%d %H:%M:%S"))
            print(f"正在处理用户 {user_id} 的轨迹 {traj_id} ...")
            result = self.infer(traj_points, similar_trajs, list(user_trajectories[user_id].values())[-10:])
            enhanced_results[traj_id] = result

        return enhanced_results
    def augment_all_users(self, user_trajectories, all_trajectories, all_similar_users, topk=2, save_path=None):

        all_results = {}

        total_users = len(user_trajectories)
        print(f"开始批量处理，共 {total_users} 位用户。")

        for i, user_id in enumerate(user_trajectories.keys(), 1):
            print(f"\n [{i}/{total_users}] 当前处理用户 {user_id} ...")

            try:
                results = self.augment_user(user_id, user_trajectories, all_trajectories, all_similar_users, topk=topk)
                all_results[user_id] = results

                if save_path and i % 1 == 0:
                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(all_results, f, ensure_ascii=False, indent=2)
                    print(f"已保存前 {i} 位用户结果至 {save_path}")

            except Exception as e:
                print(f"用户 {user_id} 处理出错：{e}")

        print("\n所有用户轨迹增强完成。")

        # 最终保存结果
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f" 所有结果已保存至 {save_path}")

        return all_results
    
    def _build_prompt_noise(self, current_traj,all_traj):
  
            prompt = f"""
            Please analyze whether the current trajectory {current_traj} is consistent with the user’s within-day activity patterns.
            If the trajectory is deemed unreasonable, identify and output a POI noise point that exhibits a clear inconsistency or 
            significant deviation between its POI category and check-in time;otherwise, return an empty list.
            
            Output: [(POI_id,POI_catname,Time)] or []
            """
            
            return prompt.strip()
    
    def infer_noise(self, current_traj,all_traj):
        prompt = self._build_prompt_noise(current_traj,all_traj)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.6,
                top_p=0.9,
                do_sample=True 
            )[0][len(inputs.input_ids[0]):].tolist()

        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        print(f"Model output: {output_text}")
    
        if "[]" in output_text:
            result = "[]"
        else:
            pattern = r"\[\s*(?:\(\s*\d+\s*,\s*'[^']+'\s*,\s*'[^']+'\s*\)\s*,?\s*)+\]"
            matches = re.findall(pattern, output_text)
            if matches:
                result = matches[-1].strip()
            else:
                result = "[]"
        print(f" Model output clean:--> {result}")
        return result
    def denoise_user(self, user_id, user_trajectories):

        if user_id not in user_trajectories:
            print(f" 用户 {user_id} 不存在。")
            return {}

        noise_results = {}

        for traj_id, traj_points in user_trajectories[user_id].items():
            print(f" 正在处理用户 {user_id} 的轨迹 {traj_id} ...")
            result = self.infer_noise(traj_points,list(user_trajectories[user_id].values())[:20])
            noise_results[traj_id] = result
        return noise_results
    
    def denoise_all_users(self, user_trajectories, save_path=None):

        all_results = {}

        total_users = len(user_trajectories)
        print(f" 开始批量处理，共 {total_users} 位用户。")

        for i, user_id in enumerate(user_trajectories.keys(), 1):
            print(f"\n [{i}/{total_users}] 当前处理用户 {user_id} ...")
            try:
                results = self.denoise_user(user_id, user_trajectories)
                all_results[user_id] = results

                if save_path and i % 1 == 0:
                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(all_results, f, ensure_ascii=False, indent=2)
                    print(f" 已保存前 {i} 位用户结果至 {save_path}")

            except Exception as e:
                print(f" 用户 {user_id} 处理出错：{e}")

        print("\n 所有用户轨迹增强完成。")

        # 最终保存结果
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f" 所有结果已保存至 {save_path}")

        return all_results

    
    def batch_denoise_trajectories(self, user_trajectories, save_path=None):

        print(" 开始批量轨迹去噪处理...")
        
        all_results = {}
        total_users = len(user_trajectories)

        for i, user_id in enumerate(user_trajectories.keys(), 1):
            print(f"\n [{i}/{total_users}] 当前处理用户 {user_id} 的轨迹去噪...")
            
            try:
                user_denoise_results = {}
                for traj_id, traj_points in user_trajectories[user_id].items():
                    print(f"  处理轨迹 {traj_id} 去噪...")
                    result = self.infer_noise(traj_points, list(user_trajectories[user_id].values())[:20])
                    user_denoise_results[traj_id] = result
                
                all_results[user_id] = user_denoise_results

                if save_path and i % 1 == 0:
                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(all_results, f, ensure_ascii=False, indent=2)
                    print(f"已保存前 {i} 位用户去噪结果至 {save_path}")

            except Exception as e:
                print(f" 用户 {user_id} 轨迹去噪处理出错：{e}")

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f" 所有轨迹去噪结果已保存至 {save_path}")

        return all_results

