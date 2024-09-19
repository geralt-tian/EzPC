from Crypto.Cipher import DES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

# DES 的块大小是 8 字节
BLOCK_SIZE = 8

# 密钥必须是 8 字节（64 位）
key = b'8bytekey'

# 创建 DES 加密器
des = DES.new(key, DES.MODE_ECB)

# 要加密的明文
plaintext = b"11111111111111111111111111111"

# 明文需要先进行填充，使其长度是块大小（8 字节）的倍数
padded_text = pad(plaintext, BLOCK_SIZE)

# 加密
ciphertext = des.encrypt(padded_text)
print(f"Ciphertext: {ciphertext.hex()}")

# 解密
decrypted_padded_text = des.decrypt(ciphertext)

# 解密后去掉填充
decrypted_text = unpad(decrypted_padded_text, BLOCK_SIZE)
print(f"Decrypted text: {decrypted_text.decode('utf-8')}")