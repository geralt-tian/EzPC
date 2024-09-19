from Crypto.Cipher import DES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

# DES 的块大小是 8 字节（64 比特）
BLOCK_SIZE = 8

# 密钥必须是 8 字节
key = b'8bytekey'

# 生成随机的初始化向量 (IV)
iv = get_random_bytes(BLOCK_SIZE)

# 创建 DES 加密器，使用 CBC 模式
cipher = DES.new(key, DES.MODE_CBC, iv)

# 明文长度超过 64 比特
plaintext = b"11111111111111111111111111111"

# 明文需要先进行填充
padded_text = pad(plaintext, BLOCK_SIZE)

# 加密
ciphertext = cipher.encrypt(padded_text)
print(f"Ciphertext (hex): {ciphertext.hex()}")

# 解密时也需要相同的 IV
decipher = DES.new(key, DES.MODE_CBC, iv)

# 解密
decrypted_padded_text = decipher.decrypt(ciphertext)

# 解密后去掉填充
decrypted_text = unpad(decrypted_padded_text, BLOCK_SIZE)
print(f"Decrypted text: {decrypted_text.decode('utf-8')}")