from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import os

MODEL_NAME = "Qwen/Qwen3-8B"

print("="*60)
print(f"加载模型: {MODEL_NAME}")
print("="*60)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# get the embedding layer
embedding_layer = model.get_input_embeddings()
print(f"Embedding shape: {embedding_layer.weight.shape}")  # [token_vocab_size, hidden_size]

# def get_word_embedding(word):
#     """获取单词的 embedding"""
#     inputs = tokenizer(word, return_tensors="pt")
#     input_ids = inputs["input_ids"]
#     num_tokens = input_ids.shape[1]
#     if num_tokens > 1:
#         print(f"Warning: '{word}' tokenized to {num_tokens} tokens, using mean pooling")
#     # 处理多 token 的情况，取平均
#     embeddings = embedding_layer(input_ids)
#     return embeddings.mean(dim=1).squeeze(0)

# def cosine_similarity(a, b):
#     """计算余弦相似度"""
#     a_norm = a / a.norm()
#     b_norm = b / b.norm()
#     return (a_norm * b_norm).sum().item()

# # 词类比: king - man + woman ≈ queen
# king = get_word_embedding("king")
# man = get_word_embedding("man")
# woman = get_word_embedding("woman")
# queen = get_word_embedding("queen")

# # 计算: king - man + woman
# result = king - man + woman

# # 计算与 queen 的相似度
# similarity = cosine_similarity(result, queen)
# print(f"\n词类比测试: king - man + woman ≈ queen")
# print(f"与 queen 的余弦相似度: {similarity:.4f}")

# # 找出最相似的词
# vocab = tokenizer.get_vocab()
# with torch.no_grad():
#     # 手动计算余弦相似度，避免 F.cosine_similarity 的形状问题
#     result_norm = result / result.norm()
#     weights_norm = embedding_layer.weight / embedding_layer.weight.norm(dim=-1, keepdim=True)
#     similarities = (result_norm.unsqueeze(0) * weights_norm).sum(dim=-1)
    
#     top_k = torch.topk(similarities, k=10)
#     print(f"\n最相似的 10 个词:")
#     for idx, sim in zip(top_k.indices, top_k.values):
#         word = tokenizer.decode([idx.item()])
#         print(f"  {word}: {sim.item():.4f}")

# # ==================== 降维方法对比测试 ====================
# print("\n" + "="*60)
# print("降维方法对比测试")
# print("="*60)

original_embedding = embedding_layer.weight  # [151936, 4096]
orig_dim = original_embedding.shape[1]  # 4096

# # 转换为 float32 用于矩阵分解
embed_float = original_embedding.float()

# def compute_reconstruction_loss(orig, recon):
#     """计算重构损失"""
#     diff = orig - recon
#     return (torch.norm(diff, p='fro') / torch.norm(orig, p='fro')).item()

# # 1. 随机正交投影 (RIP)
# print("\n1. 随机正交投影 (RIP):")
# for target_dim in [512, 1024]:
#     random_matrix = torch.randn(orig_dim, target_dim, dtype=torch.float32)
#     Q, _ = torch.linalg.qr(random_matrix)
#     R_projection = Q.to(dtype=original_embedding.dtype)
    
#     reduced = original_embedding @ R_projection
#     reconstructed = reduced @ R_projection.T
#     loss = compute_reconstruction_loss(original_embedding.float(), reconstructed.float())
#     print(f"  维度 {target_dim:4d}: 重构损失={loss:.4f}, 保留率={(1-loss)*100:.2f}%")

# # 2. PCA (主成分分析)
# print("\n2. PCA (主成分分析):")
# # PCA 需要中心化
# embedding_centered = embed_float - embed_float.mean(dim=1, keepdim=True)
# cov = embedding_centered.T @ embedding_centered / (embedding_centered.shape[0] - 1)

# for target_dim in [512, 1024]:
#     # 使用 SVD 找主成分
#     U, S, Vt = torch.linalg.svd(cov, full_matrices=False)
#     principal_components = Vt[:target_dim].T  # [4096, target_dim]
    
#     reduced = embed_float @ principal_components  # [151936, target_dim]
#     reconstructed = reduced @ principal_components.T + embed_float.mean(dim=1, keepdim=True)
#     loss = compute_reconstruction_loss(embed_float, reconstructed)
#     print(f"  维度 {target_dim:4d}: 重构损失={loss:.4f}, 保留率={(1-loss)*100:.2f}%")

# # 3. SVD (截断奇异值分解)
# print("\n3. SVD (截断奇异值分解):")
# for target_dim in [512, 1024]:
#     U, S, Vt = torch.linalg.svd(embed_float, full_matrices=False)
#     reduced = U[:, :target_dim] * S[:target_dim]  # [151936, target_dim]
#     reconstructed = reduced @ Vt[:target_dim, :]  # [151936, 4096]
#     loss = compute_reconstruction_loss(embed_float, reconstructed)
#     print(f"  维度 {target_dim:4d}: 重构损失={loss:.4f}, 保留率={(1-loss)*100:.2f}%")

# # 4. 只保留 top-k 奇异值 (类似谱投影)
# print("\n4. 谱投影 (保留 top-k 奇异值):")
# for target_dim in [512, 1024]:
#     U, S, Vt = torch.linalg.svd(embed_float, full_matrices=False)
#     # 只用 top-k 奇异向量重构
#     reduced = U[:, :target_dim] * S[:target_dim]  # [151936, target_dim]
#     reconstructed = reduced @ Vt[:target_dim, :]  # [151936, 4096]
#     loss = compute_reconstruction_loss(embed_float, reconstructed)
#     variance_explained = (S[:target_dim]**2).sum() / (S**2).sum()
#     print(f"  维度 {target_dim:4d}: 重构损失={loss:.4f}, 方差解释={variance_explained:.4f}")

# # ==================== BERT 对比实验 ====================
# print("\n\n" + "="*60)
# print("加载 BERT 进行对比: bert-base-uncased")
# print("="*60)

# bert_model = AutoModel.from_pretrained("bert-base-uncased")
# bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# bert_embedding = bert_model.get_input_embeddings()
# bert_embed_float = bert_embedding.weight.float()

# print(f"BERT Embedding shape: {bert_embed_float.shape}")  # [30522, 768]

# # 只测 512 维度作为对比
# target_dim = 512

# print(f"\nBERT 降维到 {target_dim} 维的对比:")

# # RIP
# random_matrix = torch.randn(768, target_dim, dtype=torch.float32)
# Q, _ = torch.linalg.qr(random_matrix)
# R_projection = Q.to(dtype=bert_embedding.weight.dtype)
# reduced = bert_embedding.weight @ R_projection
# reconstructed = reduced @ R_projection.T
# rip_loss = compute_reconstruction_loss(bert_embed_float, reconstructed.float())
# print(f"  RIP: 重构损失={rip_loss:.4f}, 保留率={(1-rip_loss)*100:.2f}%")

# # PCA
# embedding_centered = bert_embed_float - bert_embed_float.mean(dim=1, keepdim=True)
# cov = embedding_centered.T @ embedding_centered / (embedding_centered.shape[0] - 1)
# U, S, Vt = torch.linalg.svd(cov, full_matrices=False)
# principal_components = Vt[:target_dim].T
# reduced = bert_embed_float @ principal_components
# reconstructed = reduced @ principal_components.T + bert_embed_float.mean(dim=1, keepdim=True)
# pca_loss = compute_reconstruction_loss(bert_embed_float, reconstructed)
# variance_explained = (S[:target_dim]**2).sum() / (S**2).sum()
# print(f"  PCA: 重构损失={pca_loss:.4f}, 保留率={(1-pca_loss)*100:.2f}%, 方差解释={variance_explained:.4f}")

# # SVD
# U, S, Vt = torch.linalg.svd(bert_embed_float, full_matrices=False)
# reduced = U[:, :target_dim] * S[:target_dim]
# reconstructed = reduced @ Vt[:target_dim, :]
# svd_loss = compute_reconstruction_loss(bert_embed_float, reconstructed)
# print(f"  SVD: 重构损失={svd_loss:.4f}, 保留率={(1-svd_loss)*100:.2f}%")

# print("\n对比总结: BERT (768->512) vs Qwen3-8B (4096->512)")
# print(f"  BERT:  RIP={rip_loss:.4f}, PCA={pca_loss:.4f}, SVD={svd_loss:.4f}")
# print(f"  Qwen3:  RIP=0.9277, PCA=0.8164, SVD=0.8164")

# ==================== 保存最佳投影 (SVD) ====================
print("\n" + "="*60)
print("保存 SVD 投影到 512 维")
print("="*60)

target_dim = 512
save_dir = "/data1/neu_lab2/denseslm4/src/denseslm4/embedding/projected_embedding"

# SVD 投影
U, S, Vt = torch.linalg.svd(embed_float, full_matrices=False)
projection_matrix = Vt[:target_dim, :].float()  # [512, 4096] - 这是将 4096 维映射到 512 维的矩阵
reduced_embedding = (embed_float @ projection_matrix.T)  # [151936, 512]

# Qwen3 实际 vocab_size 是 151936 (embedding 矩阵行数)，不是 tokenizer 的 151669
Qwen3_VOCAB_SIZE = 151936

# 保存投影矩阵和降维后的 embedding
os.makedirs(save_dir, exist_ok=True)
torch.save({
    "projection_matrix": projection_matrix,  # [512, 4096]
    "reduced_embedding": reduced_embedding[:Qwen3_VOCAB_SIZE],    # [151936, 512]
    "original_mean": embed_float.mean(dim=0), # [4096] PCA 中心化用的均值
    "vocab_size": Qwen3_VOCAB_SIZE,
    "original_dim": orig_dim,
    "target_dim": target_dim,
    "method": "SVD",
}, os.path.join(save_dir, "svd_projection_512.pt"))

print(f"投影矩阵 shape: {projection_matrix.shape}")  # [512, 4096]
print(f"降维后 embedding shape: {reduced_embedding[:Qwen3_VOCAB_SIZE].shape}")  # [151936, 512]
print(f"Qwen3 vocab_size: {Qwen3_VOCAB_SIZE}")
print(f"已保存到: {save_dir}/svd_projection_512.pt")