import regex as re

def main():
    test_str = "Never gonna give you up, I'll consider this as a test    !!!"
    print(regex_tokenizer(test_str))

def regex_tokenizer(string: str) -> list:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return(re.findall(PAT, string))

if __name__ == "__main__":
    main()
