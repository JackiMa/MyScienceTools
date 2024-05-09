# version 1.0 2024年5月9日

# 修改了打开文件的逻辑，能自动处理一些sep，并且打开出错时会显示文件前10行

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os

class InterpolationFunction:
    def __init__(self, x, y, kind, name):
        self.func = interp1d(x, y, kind=kind)
        self.name = name
        # Copy all attributes from the interp1d object to this object
        self.__dict__.update(self.func.__dict__)

    def __call__(self, x):
        return self.func(x)
    
class myInterpolation:
    def __init__(self, fileName, skiprows=None, sep = None,smooth_window=5, plot=False, isregularize='none'):
        """
        初始化参数

        Args:
            fileName (str): 文件名
            skiprows (int, optional): 读取文件时跳过的行数. Defaults to None.
            decimal (str, optional): 文件中的小数点符号. Defaults to ','.
            smooth_window (int, optional): 平滑窗口的大小. Defaults to 5.
            plot (bool, optional): 是否绘制图像. Defaults to False.
            isregularize (str, optional): 数据规范化的方法，可选值为'MaxY', 'Area', 'MinMax', 'Z-Score', 'TotalY', 'none'. Defaults to 'none'.
        """
        self.fileName = fileName
        self.skiprows = skiprows
        self.smooth_window = smooth_window
        self.plot = plot
        self.sep = sep if sep is not None else '[;,;，；。\t]'
        self.isregularize = isregularize

        baseName = os.path.basename(self.fileName)
        self.name, _ = os.path.splitext(baseName)


        # 自动执行一些操作
        self.read_file()
        if self.isregularize != 'none':
            print(f"Data normalization method is {self.isregularize}.")
            self.scidata_process()
        self.smooth_data()
        self.do_interpolat()


    def scidata_process(self):
        self.x = self.data[0].values
        self.y = self.data[1].values

        x = self.x
        y = self.y
        isregularize = self.isregularize
        if isregularize == 'MaxY':
            y = y / np.max(y)
        elif isregularize == 'Area':
            y = y / np.trapz(y, x)
        elif isregularize == 'MinMax':
            y = (y - np.min(y)) / (np.max(y) - np.min(y))
        elif isregularize == 'Z-Score':
            y = (y - np.mean(y)) / np.std(y)
        elif isregularize == 'TotalY':
            y = y / np.sum(y)
        elif isregularize != 'none':
            print(f"Unknown normalization method {isregularize}, no normalization applied.")

        # 更新self.data
        self.data = pd.DataFrame({0: x, 1: y})

    def read_file(self):
        try:
            # 尝试读取文件并对数据进行分组平均
            self.data = pd.read_csv(self.fileName, header=None, skiprows=self.skiprows, sep=self.sep, engine='python')
            self.data = self.data.groupby(0, as_index=False).mean()
        except Exception as e:
            # 如果读取文件失败，打印文件的前10行
            with open(self.fileName, 'r') as file:
                lines = file.readlines()[:10]
            print("First 10 lines of the file:")
            print("".join(lines))
            
            # 重新抛出异常
            raise Exception(f"打开文件{self.fileName}错误")

    def smooth_data(self):
        # 在做变换之前，保存原始数据
        self.rawdata = self.data.copy()

        # 对数据进行对称滑动平均以平滑数据，保持原始数据的位置不变
        self.data[1] = self.data[1].rolling(window=self.smooth_window, center=True, min_periods=1).mean()

    def do_interpolat(self):
        # 对平滑后的数据进行插值
        # 首先，确保数据都是数字
        self.data[0] = pd.to_numeric(self.data[0], errors='coerce')
        self.data[1] = pd.to_numeric(self.data[1], errors='coerce')
        # 然后，删除所有的非数字（NaN）值
        self.data = self.data.dropna()
        # 最后，进行插值
        self.interpolation = InterpolationFunction(self.data[0], self.data[1], kind='cubic', name=self.name)

    def getInterpolation(self):
        # 获取插值函数
        return self.interpolation

    def plot_data(self):
        # 如果需要，绘制原始数据和插值曲线
        if self.plot:
            plt.figure(figsize=(10, 6))
            plt.title('Interpolation of data: ' + self.fileName)
            plt.plot(self.rawdata[0], self.rawdata[1], 'o', label='Original data')
            plt.plot(self.data[0], self.interpolation(self.data[0]), '-', label='Interpolated curve')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.show()

    def save_file(self, new_file_name=None):
        # 如果需要，保存处理后的数据到新的文件
        if new_file_name is not None:
            self.data.to_csv(new_file_name, index=False)

    def process(self):
        # 执行所有步骤

        self.plot_data()
        return self.getInterpolation() # 返回插值函数



def calculate_weighted_average(func1, func2):
    """
    计算两个函数的加权平均值
    用来计算func2为光电器件量子效率，func1为光谱的情况下，计算平均量子效率
    """

    # 确定两个函数的共同范围
    start = max(func1.x[0], func2.x[0])
    end = min(func1.x[-1], func2.x[-1])

    # 创建在共同范围内的等间隔的x值数组
    x = np.linspace(start, end, num=1000)

    # 计算加权的func2的值
    weighted_func2_values = np.trapz(func1(x) * func2(x),x) / np.trapz(func1(x), x)

    # 计算加权的func2值的平均值
    average_weighted_func2 = np.mean(weighted_func2_values)

    print(f"The average of {func2.name} weighted by {func1.name} is: {average_weighted_func2}")

    return average_weighted_func2