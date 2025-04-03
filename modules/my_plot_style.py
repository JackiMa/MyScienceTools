"""
科学绘图风格设置模块

提供一组预设的绘图风格设置，用于创建具有一致性和专业性的科学图表
主要特点：
1. 定义了统一的字体大小和样式
2. 设置了良好的网格线和刻度线样式
3. 定义了适合科学绘图的颜色循环
4. 自适应设置了图表大小和DPI

使用方法：
from my_plot_style import *  # 导入所有样式设置
import matplotlib.pyplot as plt

# 现在plt将使用这些设置绘图
plt.figure()
plt.plot(x, y)
plt.show()

# 自定义部分设置
plt.rcParams['figure.figsize'] = (12, 8)  # 修改默认图表大小
"""

import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np

# ===== 字体设置 =====
fontsize_main = 14  # 主要字体大小
fontsize_label = fontsize_main * 1.1  # 轴标签字体大小
fontsize_title = fontsize_main * 1.4  # 标题字体大小
fontsize_legend = fontsize_main * 0.8  # 图例字体大小
fontsize_annotate = fontsize_main * 0.7  # 注释字体大小
fontsize_tick = fontsize_main * 0.8  # 刻度字体大小

# ===== 颜色设置 =====
# 主要颜色列表 - 选择高对比度、适合科学绘图的颜色
colors = [
    '#1f77b4',  # 蓝色
    '#ff7f0e',  # 橙色
    '#2ca02c',  # 绿色
    '#d62728',  # 红色
    '#9467bd',  # 紫色
    '#8c564b',  # 棕色
    '#e377c2',  # 粉色
    '#7f7f7f',  # 灰色
    '#bcbd22',  # 黄绿色
    '#17becf',  # 青色
]

# ===== 线型设置 =====
linestyles = [
    '-',                # Solid line
    '--',               # Dashed line
    '-.',               # Dash-dot line
    ':',                # Dotted line
    (0, (1, 1)),       # Densely dotted
    (0, (5, 1)),       # Densely dashed
    (0, (3, 1, 1, 1)), # Densely dash-dotted
    (0, (5, 5)),       # Dashed
    (0, (3, 5, 1, 5)), # Dash-dot
    (0, (1, 5)),       # Loosely dotted
    (0, (5, 10)),      # Loosely dashed
    (0, (3, 10, 1, 10)), # Loosely dash-dotted
    (0, (5, 1, 1, 1)), # Dash-dot-dot
    (0, (1, 1, 1, 1)), # Densely dash-dot-dot
    (0, (3, 1, 1, 1, 1, 1)), # Dash-dot-dot-dot
    (0, (5, 2, 5, 2, 5, 10)), # Complex pattern 1
    (0, (3, 2, 1, 2, 1, 2)), # Complex pattern 2
    (0, (2, 2, 2, 1, 2, 1)), # Complex pattern 3
]

# ===== 标记设置 =====
markers = [
    'o',        # Circle
    's',        # Square
    '^',        # Triangle up
    'v',        # Triangle down
    '<',        # Triangle left
    '>',        # Triangle right
    'D',        # Diamond
    'p',        # Pentagon
    'h',        # Hexagon1
    'H',        # Hexagon2
    '8',        # Octagon
    '*',        # Star
    '+',        # Plus
    'x',        # X
    'd',        # Thin diamond
    'P',        # Plus (filled)
    'X',        # X (filled)
    '.',        # Point
    '|',        # Vertical line
    '_',        # Horizontal line
]

# ===== 创建线型循环器 =====
# 确保所有循环的长度相同
n_styles = len(colors)  # 使用颜色列表的长度作为基准

# 创建循环器，确保每个循环器长度相同
line_cycler = (cycler(color=colors) + 
              cycler(linestyle=linestyles[:len(colors)]) + 
              cycler(marker=([''] + markers)[:len(colors)]))  # 第一个标记为空，这样第一条线没有标记

# ===== 设置为全局默认值 =====
# 字体设置
plt.rcParams['font.size'] = fontsize_main
plt.rcParams['axes.labelsize'] = fontsize_label
plt.rcParams['axes.titlesize'] = fontsize_title
plt.rcParams['legend.fontsize'] = fontsize_legend
plt.rcParams['xtick.labelsize'] = fontsize_tick
plt.rcParams['ytick.labelsize'] = fontsize_tick

# 设置sans-serif字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif']
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# 网格线设置
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.7

# 刻度设置
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 0.8
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.width'] = 0.8
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True   # 在顶部也显示刻度
plt.rcParams['ytick.right'] = True  # 在右侧也显示刻度

# 轴设置
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.axisbelow'] = True  # 网格线在数据后面
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.prop_cycle'] = line_cycler

# 图表设置
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.titlesize'] = fontsize_title
plt.rcParams['figure.titleweight'] = 'bold'

# 图例设置
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.8
plt.rcParams['legend.edgecolor'] = 'gray'
plt.rcParams['legend.fancybox'] = True

# 其他设置
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 6
plt.rcParams['axes.formatter.use_mathtext'] = True  # 使用数学文本格式化器
plt.rcParams['mathtext.fontset'] = 'stix'  # 使用STIX字体渲染数学符号


# ===== 自定义配色方案 =====
# 用于热图的颜色映射
heat_colors = plt.cm.viridis

# 用于表示正负值的颜色映射
diverging_colors = plt.cm.coolwarm

# ===== 辅助函数 =====
def set_size(width_cm, height_cm=None, fraction=1.0):
    """设置图形尺寸，基于厘米
    
    参数:
        width_cm (float): 宽度，厘米
        height_cm (float, optional): 高度，厘米。如果为None，则按黄金比例计算
        fraction (float, optional): 缩放因子
    
    返回:
        tuple: 宽和高，英寸
    """
    if height_cm is None:
        # 使用黄金比例 1:0.618
        height_cm = width_cm * (1-0.618)
        
    width_in = width_cm / 2.54 * fraction
    height_in = height_cm / 2.54 * fraction
    
    return (width_in, height_in)

def reset_to_defaults():
    """重置为matplotlib的默认设置"""
    plt.rcdefaults()
    print("已重置为matplotlib默认设置")

def enable_minor_ticks():
    """启用次要刻度"""
    plt.minorticks_on()
    
def set_science_style():
    """设置为科学绘图样式，启用网格并设置合适的刻度"""
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['grid.alpha'] = 0.7
    enable_minor_ticks()
    
def set_presentation_style():
    """设置为演示文稿风格，加大字体和线宽"""
    plt.rcParams['font.size'] = fontsize_main * 1.2
    plt.rcParams['axes.labelsize'] = fontsize_label * 1.2
    plt.rcParams['axes.titlesize'] = fontsize_title * 1.2
    plt.rcParams['legend.fontsize'] = fontsize_legend * 1.2
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['lines.markersize'] = 8

# ===== LeCroy 模块的图形设置函数 =====
def setup_figure(figsize=(5, 4), style=None):
    """Setup figure with style"""
    if style:
        if isinstance(style, list):
            try:
                plt.style.use(style)
            except:
                print(f"Warning: Style {style} not found, using default.")
                plt.style.use(['science', 'no-latex'])
        else:
            try:
                plt.style.use(style)
            except:
                print(f"Warning: Style {style} not found, using default.")
                plt.style.use(['science', 'no-latex'])
    else:
        plt.style.use(['science', 'no-latex'])
        
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 
                                      'Bitstream Vera Sans', 'sans-serif']
    
    fig = plt.figure(figsize=figsize, dpi=100)
    ax = plt.subplot(111)
    
    # 添加次要刻度
    ax.minorticks_on()
    ax.grid(True, linestyle='--', alpha=0.5, which='major')
    ax.grid(True, linestyle=':', alpha=0.3, which='minor')
    
    return fig

def configure_legend(fontsize=None, framealpha=0.7, loc='upper right'):
    """
    配置图例的样式
    
    参数:
        fontsize (float, optional): 图例字体大小，默认使用全局设置
        framealpha (float): 图例背景透明度
        loc (str): 图例位置
    
    返回:
        matplotlib.legend.Legend: 配置好的图例对象
    """
    if fontsize is None:
        fontsize = fontsize_legend * 0.9
    
    legend = plt.legend(fontsize=fontsize, framealpha=framealpha, loc=loc)
    legend.get_frame().set_linewidth(0.5)
    return legend

def set_axes_labels(xlabel=None, ylabel=None, title=None):
    """
    设置坐标轴标签和标题
    
    参数:
        xlabel (str, optional): x轴标签文本
        ylabel (str, optional): y轴标签文本
        title (str, optional): 图表标题文本
    """
    if xlabel:
        plt.xlabel(xlabel, fontsize=fontsize_label)
    if ylabel:
        plt.ylabel(ylabel, fontsize=fontsize_label)
    if title:
        plt.title(title, fontsize=fontsize_title)

def finalize_figure(tight=True, save_path=None, dpi=300):
    """
    完成图形设置并保存或显示
    
    参数:
        tight (bool): 是否使用tight_layout()
        save_path (str, optional): 保存图形的路径，如果为None则显示图形
        dpi (int): 保存图形的分辨率
    """
    if tight:
        plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def format_fit_label(mean, mean_err, sigma, sigma_err, chi2, ndof):
    """Format peak fit results for legend"""
    return f"$\\mu$={mean:.2f}$\\pm${mean_err:.2f}, $\\sigma$={sigma:.2f}$\\pm${sigma_err:.2f}\n$\\chi^2$/ndof={chi2:.2f}/{ndof}"

# 默认启用科学风格
set_science_style()