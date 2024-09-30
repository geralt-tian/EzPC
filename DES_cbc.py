import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
data = pd.read_csv('scatter_data.csv')

# 分离两组数据
decoded_ring = data[data['dataset'] == 'Decoded Ring']
gelu = data[data['dataset'] == 'GELU']

# 创建绘图
plt.figure(figsize=(10, 6))

# 绘制第一组散点数据（Decoded Ring）
plt.scatter(decoded_ring['x'], decoded_ring['y'], 
            color='red', 
            marker=',', 
            s=1,       # 点的大小
            label='Decoded Ring',
            alpha=0.7)  # 透明度

# 绘制第二组散点数据（GELU）
plt.scatter(gelu['x'], gelu['y'], 
            color='blue', 
            marker=',', 
            s=1,       # 点的大小
            label='GELU',
            alpha=0.7)  # 透明度

# 设置标题和轴标签
plt.title('Simple Scatter Plot')
plt.xlabel('x-axis')
plt.ylabel('y-axis')

# 添加图例
plt.legend()

# 添加网格（可选）
# plt.grid(True)

# 保存为SVG矢量图
plt.savefig('/home/zhaoqian/EzPC/test.svg', format='svg')

# 显示图形
plt.show()