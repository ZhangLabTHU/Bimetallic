import numpy as np
import matplotlib.pyplot as plt

# 示例数据
fig, ax=plt.subplots(figsize=(8,4))
plt.rc('font',family='Arial')
# 设置刻度线朝内
ax.xaxis.set_tick_params(direction='in',labelsize=12, which='major', length=6,)  # x 轴刻度线朝内
ax.yaxis.set_tick_params(direction='in',labelsize=12, which='major', length=6,)  # y 轴刻度线朝内

categories = ['PtCo', 'PtNi', 'PtRh', 'PtPd', 'PtAg', 'PtAu']
values1 =    [0.20,   0.21,    0.12,   0.01,    -0.10,   -0.19]
values2 =    [0.03,  -0.01,    0.04,  -0.04,   0.13,     0.13]
values3 =    [0.00,   0.00,    0.00,   0.00,   0.19,     0.24]
# values_all = [0.23,   0.23,    0.19,   ,  0.04,     0.17]

# 定义每个条形的宽度
bar_width = 0.25

# 计算每个类别的x坐标
x = np.arange(len(categories))
# 创建条形统计图
plt.bar(x-bar_width, values1, bar_width, color='#D85558', label='strain')
plt.bar(x, values2, bar_width, color='#276BB3', label='electronic')
plt.bar(x+bar_width, values3, bar_width, color='#E5B82D', label='ensemble',alpha=0.8)
bar_width = 0.375
# plt.hlines(values_all[0], xmin=x[0]-bar_width, xmax=x[0]+bar_width, color='#FF0000',alpha=0.8)
# plt.hlines(values_all[1], xmin=x[1]-bar_width, xmax=x[1]+bar_width, color='#FF0000',alpha=0.8)
# plt.hlines(values_all[2], xmin=x[2]-bar_width, xmax=x[2]+bar_width, color='#FF0000',alpha=0.8)
# plt.hlines(values_all[3], xmin=x[3]-bar_width, xmax=x[3]+bar_width, color='#FF0000',alpha=0.8)
# plt.hlines(values_all[4], xmin=x[4]-bar_width, xmax=x[4]+bar_width, color='#FF0000',alpha=0.8)
# plt.hlines(values_all[5], xmin=x[5]-bar_width, xmax=x[5]+bar_width, color='#FF0000',alpha=0.8)
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'
plt.tick_params(labelsize=12)

# 添加标题和标签
# plt.title('每个种类的两栏条形统计图')
plt.ylim([-0.22,0.35])
# plt.xlabel('PtM',fontsize=1)
plt.ylabel(r'$\rm  \Delta E_O \ (eV)$',fontsize=16)
plt.xticks(x, categories)
plt.legend(frameon=False,fontsize=12,ncol=3)
plt.axhline(y=0, color='black', linestyle='-',linewidth=0.5)

# 显示图形
plt.savefig('Eo_decomposed.eps',dpi=300,bbox_inches='tight')
plt.show()