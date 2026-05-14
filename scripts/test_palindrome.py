"""Test script: train on simple repeating sequence and verify regeneration."""

import torch
from torch.nn.functional import cross_entropy
from denseslm4.modeling_denseslm4 import DenseSLM4ForCausalLM
from denseslm4.configuration_denseslm4 import DenseSLM4Config

# Create tiny model
config = DenseSLM4Config(
    vocab_size=20,  # 0-9 digits + special tokens
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64,
    max_position_embeddings=64,
    dropout=0.0,
)
model = DenseSLM4ForCausalLM(config).cuda()
print(f"Model params: {sum(p.numel() for p in model.parameters())}")

# Create training sequence: simple repeating pattern "1234567890..."
pattern = "1234567890" * 20  # 200 chars
train_seq = [int(c) + 3 for c in pattern]  # offset by 3 to use tokens 3-12

# Create dataset
seq_len = 50
inputs = []
labels = []
for i in range(0, len(train_seq) - seq_len, 5):  # overlapping windows
    inp = train_seq[i:i+seq_len]
    lab = train_seq[i+1:i+seq_len+1]
    inputs.append(inp)
    labels.append(lab)

inputs = torch.tensor(inputs, dtype=torch.long).cuda()
labels = torch.tensor(labels, dtype=torch.long).cuda()

print(f"Training samples: {len(inputs)}, seq_len: {seq_len}")

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
model.train()

for epoch in range(50):
    total_loss = 0
    for inp, lab in zip(inputs, labels):
        optimizer.zero_grad()
        logits = model(inp.unsqueeze(0)).logits  # (1, seq_len, vocab)
        loss = cross_entropy(
            logits[0, :-1, :].reshape(-1, 20),
            lab[1:],
            ignore_index=-100,
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: avg loss = {total_loss/len(inputs):.4f}")

# Test generation
model.eval()
test_input = torch.tensor([[int(c) + 3 for c in "1234567890"]], dtype=torch.long).cuda()
print(f"\nInput: 1234567890")

with torch.no_grad():
    # Teacher forcing test
    logits = model(test_input).logits
    preds = logits[0, :-1, :].argmax(dim=-1)
    print(f"Expected: 2345678901")
    print(f"Predicted: {''.join(str(p.item()-3) for p in preds[:9])}")
    
    # Free generation (first 20 tokens)
    gen = test_input.clone()
    for _ in range(20):
        logits = model(gen[:, -seq_len:]).logits
        next_tok = logits[0, -1, :].argmax().item()
        gen = torch.cat([gen, torch.tensor([[next_tok]]).cuda()], dim=1)
    
    print(f"\nFree gen (first 30 tokens):")
    print(f"Expected pattern: 12345678901234567890...")
    print(f"Generated:      {''.join(str(t.item()-3) for t in gen[0][:30])}")
