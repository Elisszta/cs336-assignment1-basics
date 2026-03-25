## Byte-Pair Tokenizer

### Problem (unicode1): Understanding Unicode
- a) It will return a blank space.
- b) It will return '\x00'.
- c) It will return as a blank space while in the sentence.

### Problem (unicode2): Unicode Encodings
- a) It contains lot of '00' and BOM head that're useless or even harmful to the tokenizer training for they will introduce noises.
- b) For example, "Hello, \xf0\xf0" can't be correctly decoded. The reason is the for-cycle took apart every single bytes, and by decoding it, it's decoding single byte while the right case is to decode by reading the marks to determine how many bytes should be considered as whole to decode(1, 2, 3 or 4 bytes).
- c) For example, \xf0\xf0, the mark of the first byte is \b1111, which indicates it a 4-byte utf-8, but the second byte starts with \b11110, which isn't a correct start mark of a continuing byte(\b10).

### Problem (train_bpe): BPE Tokenizer Training
- a) All pytest Passed

### Problem (train_bpe_tinystories): BPE Training on TinyStories
- a) It took 117s/10G RAM; the longest token is " accomplishment".
- b) The pre tokenizer part remains constant and can be accelerated by multi-processes while the bpe cycle time increase with the vocab size. 

### Problem (train_bpe_expts_owt): BPE Training on OpenWebText
- a) WAITING FOR IMPLEMENTING

### Problem (tokenizer): Implementing the tokenizer
- a) 24 Passed, 1 XPassed

### Problem (tokenizer_experiments): Experiments with tokenizers
- a) WAITING FOR IMPLEMENTING

## Transformer Language Model Architecture

### Problem (linear): Implementing the linear module
- a) Passed

### Problem (embedding): Implement the embedding module
- a) Passed

### Problem (rmsnorm): Root Mean Square Layer Normalization
- a) Passed

### Problem (rope): Implement RoPE
- a) Passed

### Problem (softmax): Implement softmax
- a) Passed

### Problem (scaled_dot_product_attention): Implement scaled dot-product attention
- a) Passed

### Problem (multihead_self_attention): Implement causal multi-head self-attention
- a) Passed

### Problem (transformer_block): Implement the Transformer block
- a) Passed

### Problem (transformer_lm): Implementing the Transformer LM
- a) Passed

### Problem (transformer_accounting): Transformer LM resource accounting
- a) The whole trainable parameters are d + 2vd + n*(2d + 4d^2 + 3d * d_ff) = about 2.14B, which need about 8.1GB (v)RAM
- b) The whole FLOP is n*(8sd^2 + 4s^2d + 6sdd_ff + 6sd) + 3sd + 2sdv = about 4.5 TFLOPs
- c) The transformer's FFN part
- d) The transformer block's FFN part, the bigger model size is, the greater FFN will consume
- e) About 150 TFLOPs, the attention part consumes most

## Training a Transformer LM



