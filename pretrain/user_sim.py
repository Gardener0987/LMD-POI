import numpy as np
import h5py
import json
from sklearn.metrics.pairwise import cosine_similarity

def load_user_trajectory_embeddings_from_h5(file_path="traj_emb.h5"):

    user_trajectory_dict = {}
    with h5py.File(file_path, 'r') as f:
        for user_key in f.keys():
            user_id = user_key.replace('user_', '')
            trajectories = []
            for traj_key in f[user_key].keys():
                trajectories.append(np.array(f[user_key][traj_key]))
            user_trajectory_dict[user_id] = trajectories
    print(f"已加载 {len(user_trajectory_dict)} 个用户的轨迹嵌入")
    return user_trajectory_dict


def compute_user_embeddings(user_trajectory_dict):

    user_embeddings = {}
    for user_id, trajectories in user_trajectory_dict.items():
        traj_means = [np.mean(traj, axis=0) for traj in trajectories if traj.size > 0]
        if len(traj_means) > 0:
            user_embeddings[user_id] = np.mean(traj_means, axis=0)
    print(f"已生成 {len(user_embeddings)} 个用户嵌入向量")
    return user_embeddings


def compute_and_save_user_similarity(user_embeddings, output_path="user_similarity.json", top_k=10):

    user_ids = list(user_embeddings.keys())
    emb_matrix = np.stack([user_embeddings[u] for u in user_ids])
    sim_matrix = cosine_similarity(emb_matrix)
    
    user_similarity = {}
    for i, uid in enumerate(user_ids):
        sim_scores = sim_matrix[i]
        sorted_indices = np.argsort(sim_scores)[::-1][1:top_k+1]  # 排除自己
        topk_users = {user_ids[j]: float(sim_scores[j]) for j in sorted_indices}
        user_similarity[uid] = topk_users

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(user_similarity, f, ensure_ascii=False, indent=2)
    print(f"用户相似度结果已保存到 {output_path} （每个用户 Top-{top_k}）")
    return user_similarity



if __name__ == "__main__":
    user_traj_dict = load_user_trajectory_embeddings_from_h5("traj_emb_czx_TKY.h5")
    
    user_emb = compute_user_embeddings(user_traj_dict)
    
    compute_and_save_user_similarity(user_emb, output_path="user_sim.json", top_k=5)
