import hashlib

blockhash = int("0000000000000000000009bc181675a2371b1c417a6ff87b4e088a90b4a1ac89", 16)
hash_str = f"{blockhash:0256b}"
zeros = 0
for c in hash_str:
    if c == "0":
        zeros += 1
    else:
        break

print(f"{hash_str} zeros {zeros}")
