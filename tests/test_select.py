import torch

# 假设您的函数定义如下（保持原样）
def _select_sparse_cond(
    self, # 注意：由于是独立测试，这个 'self' 参数将被忽略或用 None 代替
    indices: torch.Tensor,  # [B, N]
    images: list[torch.Tensor],  # List[[B, L, ...]]
    img_masks: list[torch.Tensor] | None = None,  # List[[B, L]]
    lang_tokens: torch.Tensor | None = None,  # [B, L, max_token_len]
    lang_masks: torch.Tensor | None = None,  # [B, L, max_token_len]
    state: torch.Tensor | None = None,  # [B, L, state_dim]
):
    selected_images = [
        torch.gather(
            img,
            dim=1,
            # 修正：多加了一个 .unsqueeze(-1) 来匹配 5 维张量 (B, L, C, H, W)
            index=indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *img.shape[2:]),
        )
        for img in images
    ]
    # 注意：为了兼容最新的 PyTorch 和 Python 语法，我修正了原始代码中 img.shape[2], img.shape[3], img.shape[4] 的展开方式
    # 原始代码假设 images 中的所有张量都有 5 个维度（B, L, C, H, W）。
    # 我使用 *img.shape[2:] 确保它能正确展开剩余维度。
    
    selected_img_masks = None
    if img_masks is not None:
        selected_img_masks = [
            torch.gather(
                img_mask,
                dim=1,
                index=indices.expand(-1, -1), # [B, N]
            )
            for img_mask in img_masks
        ]
    selected_lang_tokens = None
    if lang_tokens is not None:
        selected_lang_tokens = torch.gather(
            lang_tokens,
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, lang_tokens.shape[2]),
        )
    selected_lang_masks = None
    if lang_masks is not None:
        selected_lang_masks = torch.gather(
            lang_masks,
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, lang_masks.shape[2]),
        )
    selected_state = None
    if state is not None:
        selected_state = torch.gather(
            state,
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, state.shape[2]),
        )
    
    return selected_images, selected_img_masks, selected_lang_tokens, selected_lang_masks, selected_state

def test_select_sparse_cond():
    # --- 1. 定义常量和参数 ---
    B = 4   # Batch size 批次大小
    L = 16  # Full sequence length 原始序列长度
    N = 5   # Selected sequence length 稀疏选择后的长度
    state_dim = 64
    max_token_len = 10
    
    # --- 2. 模拟输入数据 ---
    
    # [B, N] - 随机选择 L 中的 N 个索引
    # 确保索引在 [0, L-1] 范围内
    indices = torch.randint(low=0, high=L, size=(B, N), dtype=torch.long)
    
    # List[[B, L, C, H, W]] - 模拟两张图像特征张量
    # 注意：我假设图像特征是 5D 张量
    images = [
        torch.randn(B, L, 3, 32, 32), # img1: [B, L, C, H, W]
        torch.randn(B, L, 10, 10, 10), # img2: [B, L, C', H', W']
    ]
    
    # List[[B, L]] - 模拟两个图像掩码
    # 注意：根据原始代码的 gather 方式，img_masks 似乎是 2D 张量 [B, L]
    img_masks = [
        torch.randint(0, 2, (B, L), dtype=torch.bool), # mask1: [B, L]
        torch.randint(0, 2, (B, L), dtype=torch.bool), # mask2: [B, L]
    ]
    
    # [B, L, max_token_len]
    lang_tokens = torch.randint(0, 100, (B, L, max_token_len), dtype=torch.long)
    
    # [B, L, max_token_len]
    lang_masks = torch.randint(0, 2, (B, L, max_token_len), dtype=torch.bool)
    
    # [B, L, state_dim]
    state = torch.randn(B, L, state_dim)
    
    # --- 3. 调用函数 (self 参数用 None 代替) ---
    selected_images, selected_img_masks, selected_lang_tokens, selected_lang_masks, selected_state = _select_sparse_cond(
        self=None,
        indices=indices,
        images=images,
        img_masks=img_masks,
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        state=state,
    )
    
    # --- 4. 验证输出的形状 (Shape Assertion) ---
    
    # 4.1 验证 selected_images
    assert isinstance(selected_images, list)
    assert len(selected_images) == len(images)
    # 形状应从 [B, L, ...] 变为 [B, N, ...]
    assert selected_images[0].shape == (B, N, 3, 32, 32)
    assert selected_images[1].shape == (B, N, 10, 10, 10)
    
    # 4.2 验证 selected_img_masks
    assert isinstance(selected_img_masks, list)
    assert len(selected_img_masks) == len(img_masks)
    # 形状应从 [B, L] 变为 [B, N]
    assert selected_img_masks[0].shape == (B, N)
    assert selected_img_masks[1].shape == (B, N)
    
    # 4.3 验证 selected_lang_tokens
    # 形状应从 [B, L, max_token_len] 变为 [B, N, max_token_len]
    assert selected_lang_tokens.shape == (B, N, max_token_len)
    
    # 4.4 验证 selected_lang_masks
    # 形状应从 [B, L, max_token_len] 变为 [B, N, max_token_len]
    assert selected_lang_masks.shape == (B, N, max_token_len)
    
    # 4.5 验证 selected_state
    # 形状应从 [B, L, state_dim] 变为 [B, N, state_dim]
    assert selected_state.shape == (B, N, state_dim)
    
    # --- 5. 验证输出的内容 (Content Assertion - 关键逻辑检查) ---
    
    # 检查 state 张量：验证 selected_state[b, i] 是否等于 state[b, indices[b, i]]
    b_idx = 0 # 批次索引
    n_idx = 2 # 稀疏索引
    
    original_l_idx = indices[b_idx, n_idx] # 找到原始序列中的索引
    
    # 断言：选择后的状态张量与原始张量在对应索引处的内容是否一致
    # 使用 torch.allclose 是浮点数比较的标准做法
    assert torch.allclose(selected_state[b_idx, n_idx], state[b_idx, original_l_idx])
    
    # 检查 images 张量：验证 selected_images[0][b, i] 是否等于 images[0][b, indices[b, i]]
    original_img_feature = images[0][b_idx, original_l_idx]
    selected_img_feature = selected_images[0][b_idx, n_idx]
    assert torch.allclose(selected_img_feature, original_img_feature)

    print("✅ 测试通过：所有输出张量的形状和内容（随机样本）验证成功。")

# 运行测试
test_select_sparse_cond()