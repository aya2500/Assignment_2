# ===========================
# Assignment 2: Debugging Vision Transformer (ViT)
# ===========================

import torch
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image

# ---------------------------
# 1. Load and preprocess image
# ---------------------------

image_path = "sample.jpg"
weights = ViT_B_16_Weights.IMAGENET1K_V1

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

image = Image.open(image_path).convert("RGB")   # 🔴 BREAKPOINT #1 → Step Into
input_tensor = transform(image).unsqueeze(0)    # 🔴 BREAKPOINT #2 → Step Over
# Snapshot #1
print("Snapshot #1: Raw input image tensor")
print("Shape:", input_tensor.shape)
print("Values slice:", input_tensor[0, :, :5, :5])

# ---------------------------
# 2. Load ViT Model
# ---------------------------

model = vit_b_16(weights=weights)               # 🔴 BREAKPOINT #3 → Step Into
model.eval()

# ---------------------------
# 3. Patch Embedding Step
# ---------------------------

patch_embed = model.conv_proj

with torch.no_grad():
    patches = patch_embed(input_tensor)         # 🔴 BREAKPOINT #4 → Step Into
    # Snapshot #2
    print("\nSnapshot #2: Image divided into patches")
    print("Shape:", patches.shape)

    B, embed_dim, Hf, Wf = patches.shape
    patches_flat = patches.flatten(2).transpose(1, 2)  # 🔴 BREAKPOINT #5 → Step Over
    # Snapshot #3
    print("\nSnapshot #3: Flattened patches")
    print("Shape:", patches_flat.shape)
    print("Values slice:", patches_flat[0, :5, :5])

    # Snapshot #4 (same shape as flattened patches)
    print("\nSnapshot #4: Patch embeddings after linear projection")
    print("Shape:", patches_flat.shape)

# ---------------------------
# 4. Add class token and positional embedding
# ---------------------------

cls_token = model.class_token.expand(B, -1, -1)   # 🔴 BREAKPOINT #6 → Step Over
# Snapshot #5
print("\nSnapshot #5: Class token before concat")
print("Shape:", cls_token.shape)
print("Values slice:", cls_token[0, :, :5])

embeddings = torch.cat((cls_token, patches_flat), dim=1)  # 🔴 BREAKPOINT #7 → Step Over
# Snapshot #6
print("\nSnapshot #6: Embeddings after adding class token")
print("Shape:", embeddings.shape)

embeddings = embeddings + model.encoder.pos_embedding[:, :embeddings.size(1), :]  # 🔴 BREAKPOINT #8 → Step Over
# Snapshot #7
print("\nSnapshot #7: Embeddings after adding positional encoding")
print("Shape:", embeddings.shape)

# ---------------------------
# 5. Trace one Encoder block
# ---------------------------

encoder_block = model.encoder.layers[0]
x = embeddings
# Snapshot #8
print("\nSnapshot #8: Encoder block input")
print("Shape:", x.shape)
print("Values slice:", x[0, :3, :5])

# ---- Multi-Head Attention ----
# Handle both old (attn) and new (self_attention) naming
attn = getattr(encoder_block, "self_attention", None)
if attn is None:
    attn = getattr(encoder_block, "attn", None)
if attn is None:
    raise AttributeError("Could not find attention module in encoder_block")

# ---- Q/K/V Projection ----
if hasattr(attn, "qkv"):
    # Old torchvision implementation (has qkv linear)
    qkv = attn.qkv(x)                          # (B, S, 3*D)
    q, k, v = qkv.chunk(3, dim=-1)
else:
    # New torchvision implementation (nn.MultiheadAttention)
    # expects (S, B, D), so we swap batch/sequence
    x_t = x.transpose(0, 1)                    # (S, B, D)
    W, b = attn.in_proj_weight, attn.in_proj_bias
    qkv = torch.nn.functional.linear(x_t, W, b)  # (S, B, 3*D)
    q, k, v = qkv.chunk(3, dim=-1)
    # swap back to (B, S, D)
    q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

# Snapshot #9
print("\nSnapshot #9: Queries Q")
print("Shape:", q.shape)
print("Values slice:", q[0, :3, :5])

# Snapshot #10
print("\nSnapshot #10: Keys K")
print("Shape:", k.shape)
print("Values slice:", k[0, :3, :5])

# Snapshot #11
print("\nSnapshot #11: Values V")
print("Shape:", v.shape)
print("Values slice:", v[0, :3, :5])

# ---- Q/K/V reshaping ----
num_heads = attn.num_heads
head_dim = embed_dim // num_heads

def reshape_heads(t):
    return t.reshape(B, -1, num_heads, head_dim).permute(0, 2, 1, 3)

q_h, k_h, v_h = map(reshape_heads, (q, k, v))


# Attention scores
# scale factor (works for both old and new)
scale = getattr(attn, "scale", 1.0 / (head_dim ** 0.5))

attn_scores = (q_h @ k_h.transpose(-2, -1)) * scale   # 🔴 BREAKPOINT #10 → Step Over
# Snapshot #12
print("\nSnapshot #12: Attention scores before softmax")
print("Shape:", attn_scores.shape)
print("Values slice:", attn_scores[0, 0, :3, :3])

attn_probs = attn_scores.softmax(dim=-1)                   # 🔴 BREAKPOINT #11 → Step Over
# Snapshot #13
print("\nSnapshot #13: Attention scores after softmax")
print("Shape:", attn_probs.shape)
print("Values slice:", attn_probs[0, 0, :3, :3])

attn_out = attn_probs @ v_h                                # 🔴 BREAKPOINT #12 → Step Over
attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, -1, embed_dim)
# Use out_proj if available, else proj (for old torchvision ViT)
if hasattr(attn, "out_proj"):
    attn_out = attn.out_proj(attn_out)   # ✅ new MultiheadAttention          # 🔴 BREAKPOINT #13 → Step Into
elif hasattr(attn, "proj"):
    attn_out = attn.proj(attn_out)       # ✅ old VisionTransformer Attention   # 🔴 BREAKPOINT #13 → Step Into
else:
    raise AttributeError("No projection layer found in attention module")
# Snapshot #14
print("\nSnapshot #14: Multi-head attention output")
print("Shape:", attn_out.shape)

# Residual + Norm
x = encoder_block.ln_1(x + attn_out)                        # 🔴 BREAKPOINT #14 → Step Into
# Snapshot #15
print("\nSnapshot #15: Residual + normalization (post-attention)")
print("Shape:", x.shape)

# ---- Feed-Forward ----
mlp = encoder_block.mlp
ff_input = x

# Snapshot #16
print("\nSnapshot #16: Feed-forward input")
print("Shape:", ff_input.shape)

# Try old style (fc1 / fc2)
fc1 = getattr(mlp, "fc1", None)
fc2 = getattr(mlp, "fc2", None)

if fc1 is not None and fc2 is not None:
    # ---- Old torchvision MLPBlock ----
    ff_hidden = fc1(ff_input)          # 🔴 BREAKPOINT #15 → Step Into
    print("\nSnapshot #17: Feed-forward hidden (fc1)")
    print("Shape:", ff_hidden.shape)

    ff_hidden = mlp.act(ff_hidden)
    ff_hidden = mlp.dropout(ff_hidden)

    ff_out = fc2(ff_hidden)            # 🔴 BREAKPOINT #16 → Step Into
    ff_out = mlp.dropout(ff_out)
    print("\nSnapshot #18: Feed-forward output (fc2)")
    print("Shape:", ff_out.shape)

else:
    # ---- Newer versions → mlp is Sequential ----
    ff_hidden = mlp[0](ff_input)       # first Linear
    print("\nSnapshot #17: Feed-forward hidden (Sequential[0])")
    print("Shape:", ff_hidden.shape)

    ff_hidden = mlp[1](ff_hidden)      # activation (GELU)
    ff_hidden = mlp[2](ff_hidden)      # dropout

    ff_out = mlp[3](ff_hidden)         # second Linear
    ff_out = mlp[4](ff_out)            # dropout
    print("\nSnapshot #18: Feed-forward output (Sequential[3])")
    print("Shape:", ff_out.shape)


x = encoder_block.ln_2(ff_input + ff_out)                   # 🔴 BREAKPOINT #17 → Step Into
# Snapshot #19
print("\nSnapshot #19: Residual + norm post-MLP")
print("Shape:", x.shape)

# Snapshot #20
print("\nSnapshot #20: Encoder block final output")
print("Shape:", x.shape)

# ---------------------------
# 6. Deeper Encoder Blocks
# ---------------------------

x_block2 = model.encoder.layers[1](x)                      # 🔴 BREAKPOINT #18 → Step Into
# Snapshot #21
print("\nSnapshot #21: Encoder block 2 output")
print("Shape:", x_block2.shape)

x_last = x_block2
for layer in model.encoder.layers[2:]:
    x_last = layer(x_last)
# Snapshot #22
print("\nSnapshot #22: Encoder last block output")
print("Shape:", x_last.shape)

# ---------------------------
# 7. Final Output
# ---------------------------

seq_out = x_last                                           # 🔴 BREAKPOINT #19 → Step Over
# Snapshot #23
print("\nSnapshot #23: Final sequence output (including class token)")
print("Shape:", seq_out.shape)

cls_final = seq_out[:, 0]                                  # 🔴 BREAKPOINT #20 → Step Over
# Snapshot #24
print("\nSnapshot #24: Class token extracted (final representation)")
print("Shape:", cls_final.shape)
print("Values slice:", cls_final[0, :5])

logits = model.heads(cls_final)                            # 🔴 BREAKPOINT #21 → Step Into
# Snapshot #25
print("\nSnapshot #25: Classification head logits")
print("Shape:", logits.shape)
print("Values slice:", logits[0, :5])

probs = torch.nn.functional.softmax(logits, dim=-1)        # 🔴 BREAKPOINT #22 → Step Over
# Snapshot #26
print("\nSnapshot #26: Softmax probabilities (slice)")
print("Shape:", probs.shape)
print("Values slice:", probs[0, :5])
