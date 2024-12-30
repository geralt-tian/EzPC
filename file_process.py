import re

# 输入文件路径
input_file = "elu_la10_ld10_s7.csv"  # 替换为您的输入文件路径
output_file = "elu_la10_ld10_s7_test.csv"  # 输出文件路径

# 正则表达式匹配数据块
pattern = r"la=(\d+),ld=(\d+),s=(\d+)\nstd::vector<std::vector<uint64_t>> data = (.*?);"

# 读取文件并解析数据块
data_list = []
with open(input_file, "r") as file:
    content = file.read()
    matches = re.findall(pattern, content, re.DOTALL)
    for match in matches:
        la = int(match[0])
        ld = int(match[1])
        s = int(match[2])
        data = match[3].strip()
        data_list.append((la, ld, s, data))

# 按照 la 和 ld 排序
data_list.sort(key=lambda x: (x[0], x[1]))

# 将排序结果写入输出文件
with open(output_file, "w") as file:
    for la, ld, s, data in data_list:
        file.write(f"la={la},ld={ld},s={s}\n")
        file.write(f"std::vector<std::vector<uint64_t>> data = {data};\n")

print(f"排序完成，结果已保存到 {output_file}")
