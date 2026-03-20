## Byte-Pair Tokenizer

### Problem (unicode1): Understanding Unicode
- a) It will return a blank space.
- b) It will return '\x00'.
- c) It will return as a blank space while in the sentence.

### Problem (unicode2): Unicode Encodings
- a) It contains lot of '00' and BOM head that're useless or even harmful to the tokenizer training for they will introduce noises.
- b) For example, "Hello, ÄãºÃ" can't be correctly decoded. The reason is the for-cycle took apart every single bytes, and by decoding it, it's decoding single byte while the right case is to decode by reading the marks to determine how many bytes should be considered as whole to decode(1, 2, 3 or 4 bytes).
- c) For example, \xf0\xf0, the mark of the first byte is \b1111, which indicates it a 4-byte utf-8, but the second byte starts with \b11110, which isn't a correct start mark of a continuing byte(\b10).

### Problem (train_bpe): BPE Tokenizer Training
- a) All pytest Passed

### Problem (train_bpe_tinystories): BPE Training on TinyStories
- a) It took 132s/10G RAM; the longest token is " accomplishment".
- b) The pre tokenizer part remains constant and can be accelerated by multi-processes while the bpe cycle time increase with the vocab size. 

### Problem (train_bpe_expts_owt): BPE Training on OpenWebText
- a) WAITING FOR IMPLEMENTING

### Problem (tokenizer): Implementing the tokenizer
- a) 24 Passed, 1 XPassed

### Problem (tokenizer_experiments): Experiments with tokenizers
- a)

