"""
LeCroy Data Processing Package

Features:
1. Read LeCroy oscilloscope waveform and spectrum data
2. Auto-adjust units and scales based on data type
3. Visualization for waveforms and spectra
4. Peak finding and Gaussian fitting
5. Auto-detection of data types
6. Support for logarithmic axis display

Usage:
- Read and display waveform/spectrum:
  data = LeCroyDATA('file_path.txt')
  data.plot(save_path=None, logPlot='logY')

- Get processed data:
  x, y, unitX, unitY = data.get_data()

- Find and fit peaks in a specific range:
  analyzer = SpectrumAnalyzer('spectrum_file.txt')
  analyzer.fit_peak(fit_range=[2000, 2500])
  
- Find and fit all peaks:
  analyzer = SpectrumAnalyzer('spectrum_file.txt')
  analyzer.plot_with_fit('save_path.png')
  
- Plot waveform without fitting:
  analyzer = SpectrumAnalyzer('waveform_file.txt')
  analyzer.plot(logPlot='logY')  # Optional: logPlot = 'logX', 'logY', 'logXY'
"""

# Version 2.0.0
# 2025-4-3
# Author: Ge Ma

# 修订记录
# 重构代码

import re
import numpy as np
import matplotlib.pyplot as plt
import os 
import scienceplots
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from my_plot_style import *


def get_datalist(data_dir, file_type=['txt']):
    """获取指定目录下的所有符合类型的数据文件"""
    datalists = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            for ftype in file_type:
                if file.endswith(ftype):
                    datalists.append(os.path.join(root, file))
    return datalists

class LeCroyDATA:
    """
    读取LeCroy的数据文件，返回数据和文件名，可以画图。
    
    参数:
    data_dir (str): 数据文件路径
    delimiters (list): 尝试使用的分隔符列表
    skiprows (int): 跳过文件头的行数
    xfactor (float, optional): x轴数据的比例因子
    yfactor (float): y轴数据的比例因子
    isSpectrum (bool, optional): 是否为能谱数据
    
    使用方法:
    data = LeCroyDATA('文件路径.txt')
    data.plot(save_path=None, logPlot='logY')
    """
    def __init__(self, data_dir, delimiters=[',', '\t', ' ', ';'], skiprows=5, xfactor=None, yfactor=1, isSpectrum=None):
        self.data_dir = data_dir
        self.delimiters = delimiters
        self.skiprows = skiprows
        self.raw_data = self._LeCroy_data_read()
        self.data = self.raw_data.copy() if self.raw_data is not None else None
        self.title = self.get_name()
        self.metrics = self._get_metrics()
        self.xfactor = xfactor
        self.yfactor = yfactor
        self.is_lecroy = False
        self.device = None
        self.data_type = None
        self.acquisition_time = None
        self.bins = None
        
        # 存储处理后的数据
        self.processed_data = None
        self.processed_x = None
        self.processed_y = None
        self.unitX = None
        self.unitY = None
        
        # 尝试解析LeCroy文件头
        self._parse_lecroy_header()
        
        # 自动判断数据类型
        if isSpectrum is not None:
            self.isSpectrum = isSpectrum
        else:
            if self.is_lecroy:
                self.isSpectrum = (self.data_type == 'Histogram')
            else:
                # 非LeCroy数据，根据数据特征判断
                if len(self.data) > 0 and len(self.data[0]) == 2:
                    # 检查数据是否看起来像能谱（离散的y值）
                    y_values = self.data[:, 1]
                    unique_y = np.unique(y_values)
                    if len(unique_y) < len(y_values) * 0.1:  # 如果y值重复较多，可能是能谱
                        self.isSpectrum = True
                    else:
                        self.isSpectrum = False


        # 初始化处理一次数据
        self.process_data()

    def _parse_lecroy_header(self):
        """解析LeCroy文件头信息"""
        try:
            with open(self.data_dir, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 5:
                    # 解析第一行：设备信息
                    first_line = lines[0].strip().split()
                    if len(first_line) >= 3:
                        self.device = first_line[0]
                        self.data_type = first_line[2]
                        self.is_lecroy = True
                    
                    # 解析第四行：采数时间
                    if len(lines) >= 4:
                        time_line = lines[3].strip()
                        time_match = re.search(r'(\d+\.?\d*)\s*(s|ms|us|ns|ps)', time_line)
                        if time_match:
                            value, unit = time_match.groups()
                            value = float(value)
                            if unit == 'ms':
                                value *= 1e-3
                            elif unit == 'us':
                                value *= 1e-6
                            elif unit == 'ns':
                                value *= 1e-9
                            elif unit == 'ps':
                                value *= 1e-12
                            self.acquisition_time = value
        except Exception as e:
            print(f"Warning: Could not parse LeCroy header: {str(e)}")
            self.is_lecroy = False

    def _LeCroy_data_read(self):
        """读取LeCroy数据文件，尝试不同的分隔符"""
        for delimiter in self.delimiters:
            try:
                data = np.loadtxt(self.data_dir, skiprows=self.skiprows, delimiter=delimiter)
                break
            except ValueError:
                pass
        else:
            print(f"Warning: Could not read file {self.data_dir} with provided delimiters.")
            return None

        if data.ndim != 2 or data.shape[1] != 2:
            print(f"Warning: Data in file {self.data_dir} is not 2D. Returning first 6 rows.")
            return data[:6]

        return data

    def process_data(self, isregularize='none', rebin_factor=None):
        """处理数据，每次处理都基于原始数据"""
        if self.raw_data is None:
            return None, None, None, None
            
        # 数据处理总是从原始数据开始
        self.data = self.raw_data.copy()
        
        # 确定适当的单位
        self._determine_units()
        
        # 处理数据
        x = self.data[:,0] * self.xfactor
        y = self.data[:,1] * self.yfactor

        # 归一化处理
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
        
        # Rebin处理（仅针对能谱）
        if self.isSpectrum and rebin_factor is not None:
            if self.bins is None:
                self.bins = len(x) // rebin_factor if rebin_factor else len(x)
            
            # 这里我们不实际重新分bin，只是设置bins参数，在绘图时使用
            
        # 存储处理后的数据
        self.processed_x = x
        self.processed_y = y
        self.processed_data = np.column_stack((x, y))
        
        return x, y, self.unitX, self.unitY

    def _determine_units(self):
        """确定适当的单位"""
        if self.isSpectrum:  # 能谱数据处理
            self.unitY = 'counts'
            if self.xfactor is None:  # 自动判断
                deltaX = self.data[3,0] - self.data[2,0]
                if deltaX < 1E-12:
                    self.xfactor = 1E12
                    self.unitX = 'Areas [pV$\\cdot$s]'
                elif deltaX < 1E-7:
                    self.xfactor = 1E9
                    self.unitX = 'Areas [nV$\\cdot$s]'
                elif deltaX < 1E-3:  # 1E-7 < deltaX < 1E-3
                    self.xfactor = 1
                    self.unitX = 'Areas [$\\mu$V$\\cdot$s]'
                elif deltaX < 1:
                    self.xfactor = 1E3
                    self.unitX = 'Voltage [mV]'
                else:
                    self.xfactor = 1
                    self.unitX = 'Voltage [V]'
        else:  # 波形数据处理
            self.unitY = 'voltage [V]'
            if self.xfactor is None:
                deltaX = self.data[3,0] - self.data[2,0]
                if deltaX < 1E-12:
                    self.xfactor = 1E12
                    self.unitX = 'Time [ps]'
                elif deltaX < 1E-7:
                    self.xfactor = 1E9
                    self.unitX = 'Time [ns]'
                elif deltaX < 1E-3: 
                    self.xfactor = 1E6
                    self.unitX = 'Time [$\\mu$s]'
                elif deltaX < 1:
                    self.xfactor = 1E3
                    self.unitX = 'Time [ms]'
                else:
                    self.xfactor = 1
                    self.unitX = 'Time [s]'

    def scidata_process(self, isregularize='none', rebin_factor=None):
        """处理数据并返回处理后的数据"""
        return self.process_data(isregularize, rebin_factor)

    def plot(self, isregularize='none', rebin_factor=10, save_path=None, logPlot=False, title=None, style=None, figsize=(5, 4), **kwargs):
        """Plot waveform or spectrum
        
        Parameters:
        -----------
        isregularize : str, optional
            Normalization method ('none', 'MaxY', 'Area', 'MinMax', 'Z-Score', 'TotalY')
        rebin_factor : int, optional
            Factor to rebin the data (only for spectra)
        save_path : str, optional
            Path to save the plot
        logPlot : str, optional
            Log scale setting ('logY', 'logX', 'logXY', or False)
        title : str, optional
            Custom title for the plot (if None, no title will be shown)
        style : list or str, optional
            Matplotlib style to use (e.g. ['science', 'ieee'])
        figsize : tuple, optional
            Figure size in inches (width, height)
        **kwargs : 
            Additional keyword arguments to pass to plt functions
        """
        print("Normalization method: ", isregularize)
        
        # 处理数据
        x, y, unitX, unitY = self.process_data(isregularize, rebin_factor)
        
        # Create figure with style
        fig = setup_figure(figsize=figsize, style=style)
        
        # Plot based on data type
        if self.isSpectrum: 
            if self.bins is None:
                self.bins = len(self.data)  # Use number of data points as bins
            plt.hist(x, weights=y, bins=self.bins, histtype='step', **kwargs)
        else:
            plt.plot(x, y, **kwargs)
            
        # Set labels - 可被后续调用覆盖
        set_axes_labels(xlabel=unitX, ylabel=unitY, title=title)
        
        # Configure additional plot settings
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Set log scale
        if logPlot:
            if logPlot == 'logY':
                plt.yscale('log')
            elif logPlot == 'logX':
                plt.xscale('log')
            elif logPlot == 'logXY':
                plt.yscale('log')
                plt.xscale('log')
        
        # 返回当前figure，让用户可以继续自定义
        return fig, plt.gca()

    def get_rawdata(self):
        """获取原始数据"""
        return self.raw_data
    
    def get_data(self, processed=True):
        """获取数据
        
        Parameters:
        -----------
        processed : bool
            如果为True，返回处理后的数据；否则返回原始数据
        """
        if processed:
            if self.processed_x is None or self.processed_y is None:
                self.process_data()
            return self.processed_x, self.processed_y, self.unitX, self.unitY
        else:
            # 返回原始数据
            x = self.raw_data[:,0]
            y = self.raw_data[:,1]
            return x, y, None, None

    def get_name(self):
        """获取文件名"""
        return self.data_dir.split('\\')[-1]

    def _get_metrics(self):
        """从文件名中提取相关信息"""
        metrics = {}
        temperature_match = re.search(r'tmp(n?\d+)', self.title)
        if temperature_match:
            metrics['temperature'] = temperature_match.group(1)
            if 'n' in metrics['temperature']:  # 负温度
                metrics['temperature'] = '-' + metrics['temperature'].replace('n', '')
        # 这里可以添加更多信息的提取
        return metrics

    def set_bins(self, bins=None):
        """设置能谱的bin数量"""
        if self.isSpectrum:
            if bins is None:
                # 使用原始数据的分bin方式
                self.bins = len(self.data)
            else:
                self.bins = bins
        else:
            print("Warning: set_bins() is only applicable for spectrum data.")

class SpectrumAnalyzer:
    """
    使用LeCroyDATA获取数据文件，并分析能谱、寻峰、拟合
    
    参数:
    data_dir (str): 数据文件路径
    sigma (float, optional): 自定义sigma值
    
    使用方法:
    analyzer = SpectrumAnalyzer('能谱文件路径.txt')
    analyzer.fit_peak(fit_range=[2000, 2500])
    analyzer.plot_with_fit('保存路径.png')
    analyzer.plot(logPlot='logY')
    """
    def __init__(self, data_dir, sigma=None):
        self.data_dir = data_dir
        self.lecroy_data = LeCroyDATA(data_dir)
        self.xdata, self.ydata, self.unitX, self.unitY = self.lecroy_data.get_data(processed=True)
        self.peaks_index = None
        self.data_width = None
        self.sigma = sigma  # 自定义sigma值
        self.isSpectrum = self.lecroy_data.isSpectrum
        self.fitted_peaks = {} # 存储拟合后的峰位，key为峰位，value为(popt, perr)

    def _getFilterSigma(self):
        """根据数据结构找到合适的滤波参数sigma"""
        spectrum = self.ydata
        centroid = np.average(self.xdata, weights=spectrum)

        # 初始化包含的ydata计数的总和
        count_sum = 0

        # 找到重心在xdata中的位置
        centroid_index = (np.abs(self.xdata - centroid)).argmin()

        # 从重心处开始，向两边扩展
        left_index = right_index = centroid_index
        delta_step = len(self.xdata) // 200
        while count_sum < sum(spectrum) * 0.95:
            if left_index > delta_step:
                left_index -= delta_step
            if right_index < len(self.xdata) - delta_step:
                right_index += delta_step
            if left_index <= delta_step and right_index >= len(self.xdata) - delta_step:
                break
            count_sum = sum(spectrum[left_index:right_index+1])

        # 始终计算数据宽度，无论是否使用自定义sigma
        self.data_width = self.xdata[right_index] - self.xdata[left_index]
        
        # 使用自定义sigma或计算sigma
        if self.sigma is not None:
            return self.sigma
        
        # 计算滤波参数
        sigma = (right_index - left_index) * 0.2
        return sigma

    def _gaussian(self, x, amplitude, mean, stddev):
        """高斯函数"""
        return amplitude * np.exp(-((x - mean)**2 / (2 * stddev**2)))

    def _gaussian_plus_linear(self, x, amplitude, mean, stddev, slope, intercept):
        """高斯函数加线性背景"""
        return amplitude * np.exp(-((x - mean)**2 / (2 * stddev**2))) + slope * x + intercept

    def _value_to_index(self, value):
        """找到最接近value的索引位置"""
        return np.abs(self.xdata - value).argmin()

    def _fit_gaussian(self, peak, spectrum, x_index):
        """对单个峰进行高斯拟合"""
        # 定义拟合范围
        fit_range = float(0.5 * np.sqrt(abs(peak))) * np.sqrt(self.xdata[-1] - self.xdata[0]) * 0.1
        if self.data_width is None:
            self._getFilterSigma()  # 确保data_width已计算
        fit_range = max(fit_range, self.data_width * 0.01)  # 最小取数据宽度的1%
        
        start = max(self.xdata[0], peak - fit_range)
        end = min(self.xdata[-1], peak + fit_range)
        
        start_index = self._value_to_index(start)
        end_index = max(self._value_to_index(end), start_index+1)
        peak_index = self._value_to_index(peak)
        
        # 设置拟合边界和初值
        lb = [0, 0.5*start, -np.inf, -np.inf, -np.inf]
        ub = [1.5*max(self.ydata), 2*end, np.inf, np.inf, np.inf]
        x0 = [spectrum[peak_index], peak, 1/2*fit_range, 0, 0]
        
        # 第一次拟合
        try:
            popt1, pcov1 = curve_fit(self._gaussian_plus_linear, self.xdata[start_index:end_index], 
                                     spectrum[start_index:end_index], p0=x0, bounds=(lb, ub), maxfev=5000)
        except RuntimeError:
            print(f"峰位 {peak} 第一次拟合失败")
            return None

        # 计算第一次拟合的chi2/ndof
        expected_value1 = self._gaussian_plus_linear(self.xdata[start_index:end_index], *popt1)
        residuals1 = spectrum[start_index:end_index] - expected_value1
        chi2_1 = np.sum(residuals1**2 / expected_value1)
        ndof1 = len(residuals1) - len(popt1)
        perr1 = np.sqrt(np.diag(pcov1))

        # 保存第一次拟合的范围
        start_index1 = start_index
        end_index1 = end_index

        # 根据第一次拟合结果调整拟合范围，进行第二次拟合
        peak = popt1[1]
        fit_range = min(abs(7 * popt1[2]), 1.5*fit_range)
        
        start = max(self.xdata[0], popt1[1] - 0.9*fit_range)
        end = min(self.xdata[-1], popt1[1] + 1.1*fit_range)
        start_index = self._value_to_index(start)
        end_index = self._value_to_index(end)
        
        # 第二次拟合
        try:
            popt2, pcov2 = curve_fit(self._gaussian_plus_linear, self.xdata[start_index:end_index], 
                                     spectrum[start_index:end_index], 
                                     p0=[spectrum[peak_index], peak, 1/2*fit_range, 0, 0], 
                                     bounds=(lb, ub), maxfev=5000)
        except (RuntimeError, ValueError):
            print(f"峰位 {peak} 第二次拟合失败，使用第一次拟合结果")
            return popt1, perr1, chi2_1, ndof1, start_index1, end_index1

        # 计算第二次拟合的chi2/ndof
        expected_value2 = self._gaussian_plus_linear(self.xdata[start_index:end_index], *popt2)
        residuals2 = spectrum[start_index:end_index] - expected_value2
        chi2_2 = np.sum(residuals2**2 / expected_value2)
        ndof2 = len(residuals2) - len(popt2)
        perr2 = np.sqrt(np.diag(pcov2))

        # 比较两次拟合结果，返回较好的一次
        if np.sum(perr1) < np.sum(perr2):
            return popt1, perr1, chi2_1, ndof1, start_index1, end_index1
        else:
            return popt2, perr2, chi2_2, ndof2, start_index, end_index

    def _find_all_peaks(self, spectrum, x_index):
        """寻找并拟合所有峰"""
        # 获取滤波参数
        sigma = self._getFilterSigma()
        
        # 应用高斯滤波平滑谱
        spectrum_smooth = gaussian_filter1d(spectrum, sigma=sigma)
        
        # 在平滑后的谱中寻找峰
        peaks_index, _ = find_peaks(spectrum_smooth, distance=10, width=3, height=10, prominence=2)
        peaks_lists = self.xdata[peaks_index]
        self.peaks_index = peaks_index
        
        print(f"找到 {len(peaks_lists)} 个峰")
        
        # 拟合结果列表 (popt, perr, chi2, ndof, start, end)
        fit_results = []
        
        # 对每个峰进行拟合
        for peak in peaks_lists:
            # 拟合峰
            result = self._fit_gaussian(peak, spectrum, x_index)
            
            # 如果拟合失败，跳过
            if result is None:
                print(f"峰 {peak} 拟合失败，跳过")
                continue
                
            popt, perr, chi2, ndof, start_index, end_index = result
            chi2_by_ndof = chi2/ndof
            
            # 如果峰数量较多，根据条件筛选
            if len(peaks_lists) > 3:
                print(f"发现 {len(peaks_lists)} 个峰，数量较多。谱可能存在噪声，请检查拟合结果。")
                
                # 根据拟合优度筛选
                if chi2_by_ndof > 20 or chi2_by_ndof < 0.5:
                    print(f"由于chi2/ndof = {chi2_by_ndof:.2f}不符合要求，移除峰 {peak}")
                    continue
                
                # 根据能量分辨率筛选
                if (popt[2]**2 / popt[1]) > 0.5:
                    print(f"由于sigma^2/mean = {popt[2]**2 / popt[1]:.2f}过大，移除峰 {peak}")
                    continue
            
            fit_results.append((popt, perr, chi2, ndof, start_index, end_index))
            
        return fit_results

    def _plot_spectrum_with_peaks(self, spectrum, x_index, fit_results, save_path=None, logPlot=False, title=None, style=None, figsize=(5, 4), show_annotations=True, **kwargs):
        """Plot spectrum with peak fitting results
        
        Parameters:
        -----------
        show_annotations : bool, optional
            Whether to display peak annotations (arrows and stars)
        **kwargs : 
            Additional keyword arguments to pass to plt functions
        """
        # Create figure with style
        fig = setup_figure(figsize=figsize, style=style)
        
        # Plot spectrum (without label)
        if self.isSpectrum:
            plt.hist(self.xdata, weights=spectrum, bins=len(self.xdata), histtype='step', **kwargs)
        else:
            plt.plot(self.xdata, spectrum, **kwargs)
        
        # Plot each fit result (使用纯实线，不添加额外标记)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, result in enumerate(fit_results):
            popt, perr, chi2, ndof, start, end = result
            fit_label = format_fit_label(popt[1], perr[1], popt[2], perr[2], chi2, ndof)
            color = colors[(i + 1) % len(colors)]  # 使用不同于数据的颜色
            plt.plot(self.xdata[start:end], self._gaussian_plus_linear(self.xdata[start:end], *popt), 
                    '-', color=color, linewidth=1, label=fit_label)
            
            # Store the fitted peak for future reference
            self.fitted_peaks[popt[1]] = (popt, perr)
            
            # 直接标注每个拟合峰
            if show_annotations:
                peak_index = np.abs(self.xdata - popt[1]).argmin()
                peak_x = popt[1]  # 使用拟合得到的峰值
                peak_y = spectrum[peak_index]
                
                # 创建箭头注释
                arrow_y = min(peak_y + 0.1*plt.gca().get_ylim()[1], 0.9*plt.gca().get_ylim()[1])
                plt.annotate(f'{peak_x:.2f}', 
                            xy=(peak_x, peak_y), 
                            xycoords='data', 
                            xytext=(peak_x + 0.02 * (plt.gca().get_xlim()[1] - plt.gca().get_xlim()[0]), arrow_y), 
                            textcoords='data',
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3", clip_on=True),
                            fontsize=fontsize_annotate)
                
                # 绘制标记
                plt.plot(peak_x, peak_y, "*", color='red')
        
        # Create legend with custom formatting
        configure_legend()
        
        # Set axes labels and title
        set_axes_labels(xlabel=self.unitX, ylabel=self.unitY, title=title)
        
        # Set log scale
        if logPlot:
            if logPlot == 'logY':
                plt.yscale('log')
            elif logPlot == 'logX':
                plt.xscale('log')
            elif logPlot == 'logXY':
                plt.yscale('log')
                plt.xscale('log')
        
        # 返回当前图形和坐标轴，允许用户进一步自定义
        ax = plt.gca()
        
        # Finalize and show/save
        if save_path:
            finalize_figure(tight=True, save_path=save_path)
            return None, None
        return fig, ax

    def fit_peak(self, fit_range, p0=[1,0,1,0,0], title=None, style=None, figsize=(5, 4), save_path=None, **kwargs):
        """
        Fit peak using gaussian_plus_linear
        
        Parameters:
        fit_range (list): x-axis range for fitting [min, max]
        p0 (list): Initial parameters [amplitude, mean, stddev, slope, intercept]
        title (str, optional): Custom title for the plot
        style (list or str, optional): Matplotlib style to use
        figsize (tuple, optional): Figure size in inches (width, height)
        save_path (str, optional): Path to save the plot, if None or empty the plot will be displayed
        **kwargs: Additional keyword arguments to pass to plt functions
        
        Returns:
        tuple: (popt, perr) Fit parameters and errors
        """
        left = self._value_to_index(fit_range[0])
        right = self._value_to_index(fit_range[1])

        if right - left < 1:
            raise ValueError("Fit range too small, please select a larger range.")

        # Set initial parameters
        amp, mean, stddev, slope, intercept = p0
        if amp == 1:
            amp = 0.6 * max(self.ydata[left:right])
        if mean == 0:
            mean = self.xdata[left + np.argmax(self.ydata[left:right])]
        if stddev == 1:
            stddev = (right - left) * 0.1

        p0 = [amp, mean, stddev, slope, intercept]

        try:
            # Perform fit
            popt, pcov = curve_fit(self._gaussian_plus_linear, self.xdata[left:right], self.ydata[left:right], p0=p0)
            perr = np.sqrt(np.diag(pcov))

            # Calculate goodness of fit
            residuals = self.ydata[left:right] - self._gaussian_plus_linear(self.xdata[left:right], *popt)
            chi2 = np.sum((residuals ** 2) / self._gaussian_plus_linear(self.xdata[left:right], *popt))
            Ndof = len(self.xdata[left:right]) - len(popt)

            # Store fitted peak
            self.fitted_peaks[popt[1]] = (popt, perr)
            
            # Create fit label and title
            fit_label = format_fit_label(popt[1], perr[1], popt[2], perr[2], chi2, Ndof)
            plot_title = title if title is not None else f"Peak at {popt[1]:.2f}$\\pm${perr[1]:.2f}"
        except RuntimeError:
            plot_title = title if title is not None else "Fit failed"
            fit_label = "Fit failed"

        # Create figure with style
        fig = setup_figure(figsize=figsize, style=style)
        
        # Plot data
        if self.isSpectrum:
            plt.hist(self.xdata, weights=self.ydata, bins=len(self.xdata), histtype='step', **kwargs)
        else:
            plt.plot(self.xdata, self.ydata, **kwargs)
        
        # Plot fit
        if 'popt' in locals():
            plt.plot(self.xdata[left:right], self._gaussian_plus_linear(self.xdata[left:right], *popt), 
                     '-', linewidth=2.5, label=fit_label)
        else:
            plt.plot(self.xdata[left:right], self.ydata[left:right], 'r', label='Fit range')
        
        # Configure legend and labels
        configure_legend()
        set_axes_labels(xlabel=self.unitX, ylabel=self.unitY, title=plot_title)

        # Finalize and show/save
        finalize_figure(tight=True, save_path=save_path)

        if 'popt' in locals():
            return popt, perr

    def plot_with_fit(self, save_path=None, logPlot=False, title=None, style=None, figsize=(5, 4), filter_sigma=None, show_annotations=True, **kwargs):
        """
        Find and fit all peaks, generate plot
        
        Parameters:
        save_path (str, optional): Path to save the plot
        logPlot (str, optional): Log scale setting ('logY', 'logX', 'logXY', or False)
        title (str, optional): Custom title for the plot
        style (list or str, optional): Matplotlib style to use
        figsize (tuple, optional): Figure size in inches (width, height)
        filter_sigma (float, optional): Custom sigma value for peak detection filter
        show_annotations (bool, optional): Whether to display peak annotations (arrows and stars)
        **kwargs: Additional keyword arguments to pass to plt functions
        
        Returns:
        tuple: (fig, ax) matplotlib figure and axes objects if save_path is None
        """
        if filter_sigma is not None:
            self.sigma = filter_sigma
            
        spectrum = self.ydata
        x_index = np.arange(len(spectrum))
        fit_results = self._find_all_peaks(spectrum, x_index)    
        return self._plot_spectrum_with_peaks(spectrum, x_index, fit_results, save_path, logPlot=logPlot, 
                                      title=title, style=style, figsize=figsize, 
                                      show_annotations=show_annotations, **kwargs)

    def plot(self, save_path=None, logPlot=False, title=None, style=None, figsize=(5, 4), **kwargs):
        """
        Plot waveform or spectrum without peak fitting
        
        Parameters:
        save_path (str, optional): Path to save the plot
        logPlot (str, optional): Log scale setting ('logY', 'logX', 'logXY', or False)
        title (str, optional): Custom title for the plot
        style (list or str, optional): Matplotlib style to use
        figsize (tuple, optional): Figure size in inches (width, height)
        **kwargs: Additional keyword arguments to pass to plt functions
        
        Returns:
        tuple: (fig, ax) matplotlib figure and axes objects if save_path is None
        """
        spectrum = self.ydata
        x_index = np.arange(len(spectrum))
        
        # Create figure with style
        fig = setup_figure(figsize=figsize, style=style)
        
        # Plot based on data type
        if self.isSpectrum:
            plt.hist(self.xdata, weights=spectrum, bins=len(self.xdata), histtype='step', **kwargs)
        else:
            plt.plot(self.xdata, spectrum, **kwargs)
            
        # Set axes labels and title
        set_axes_labels(xlabel=self.unitX, ylabel=self.unitY, title=title)
        
        # Set log scale
        if logPlot:
            if logPlot == 'logY':
                plt.yscale('log')
            elif logPlot == 'logX':
                plt.xscale('log')
            elif logPlot == 'logXY':
                plt.yscale('log')
                plt.xscale('log')
        
        # 返回当前图形和坐标轴，允许用户进一步自定义
        ax = plt.gca()
        ax.minorticks_on()
        # Finalize and show/save
        if save_path:
            finalize_figure(tight=True, save_path=save_path)
            return None, None
        return fig, ax


if __name__ == "__main__":
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(current_dir, 'spectrum_test.txt')
    
    # 创建测试图像保存目录
    test_dir = os.path.join(current_dir, 'testFigure')
    os.makedirs(test_dir, exist_ok=True)
    
    # 示例 1: 使用 plt 函数自定义绘图样式
    print("\n示例 1: 自定义绘图样式")
    analyzer = SpectrumAnalyzer(test_file)
    
    # 使用 plot 并自定义 plt 样式
    fig, ax = analyzer.plot(figsize=(8, 5))
    plt.xlabel('Custom X Label')
    plt.ylabel('Custom Y Label')
    plt.title('Custom Plot Title')
    plt.savefig(os.path.join(test_dir, 'example1_custom_style.png'), dpi=300)
    plt.close()
    
    # 使用 plot_with_fit 并自定义 plt 样式
    fig, ax = analyzer.plot_with_fit(filter_sigma=10, show_annotations=True)
    plt.xlabel('Energy Scale')
    plt.ylabel('Intensity (counts)')
    plt.title('Spectrum Analysis with Custom Labels')
    plt.savefig(os.path.join(test_dir, 'example1_custom_fit.png'), dpi=300)
    plt.close()
    
    # 示例 2: 使用 kwargs 直接传递样式参数
    print("\n示例 2: 使用参数设置样式")
    analyzer = SpectrumAnalyzer(test_file)
    
    # 传递 color, linestyle 等参数
    analyzer.plot(
        save_path=os.path.join(test_dir, 'example2_kwargs.png'),
        color='blue',
        linewidth=2,
        alpha=0.7,
        title='Styled with kwargs'
    )
    
    # 拟合时传递样式参数
    analyzer.plot_with_fit(
        save_path=os.path.join(test_dir, 'example2_fit_kwargs.png'),
        filter_sigma=5,
        title='Fit with Style Parameters',
        color='green',
        alpha=0.8
    )
    
    # 示例 3: 访问原始数据和处理后数据
    print("\n示例 3: 处理数据示例")
    data = LeCroyDATA(test_file)
    
    # 获取原始数据
    raw_x, raw_y, _, _ = data.get_data(processed=False)
    print(f"原始数据形状: {raw_x.shape}")
    
    # 获取处理后的数据（默认处理）
    proc_x, proc_y, unit_x, unit_y = data.get_data(processed=True)
    print(f"处理后数据形状: {proc_x.shape}")
    print(f"单位: {unit_x}, {unit_y}")
    
    # 应用不同的处理并获取数据
    data.process_data(isregularize='MaxY', rebin_factor=20)
    norm_x, norm_y, _, _ = data.get_data(processed=True)
    print(f"归一化数据: min={norm_y.min():.2f}, max={norm_y.max():.2f}")
    
    # 测试1：基本数据读取和显示
    print("\n测试1：基本显示功能")
    data = LeCroyDATA(test_file)
    print(f"设备: {data.device}")
    print(f"数据类型: {data.data_type}")
    print(f"采集时间: {data.acquisition_time}")
    print(f"是否为能谱: {data.isSpectrum}")
    
    fig, ax = data.plot()
    plt.savefig(os.path.join(test_dir, 'basic_plot.png'))
    plt.close()
    
    # 测试2：能谱分析
    print("\n测试2：能谱分析功能")
    for sigma in [None, 10]:
        analyzer = SpectrumAnalyzer(test_file, sigma=sigma)
        analyzer.plot_with_fit(save_path=os.path.join(test_dir, f'spectrum_analysis_sigma{sigma}.png'))
    
    # 测试3：自定义bin数量
    print("\n测试3：自定义bin功能")
    for bins in [50, 200]:
        data = LeCroyDATA(test_file)
        data.set_bins(bins)
        data.plot(save_path=os.path.join(test_dir, f'custom_bins_{bins}.png'))
    
    # 测试4：单峰拟合
    print("\n测试4：单峰拟合功能")
    analyzer = SpectrumAnalyzer(test_file)
    fit_ranges = [[100, 150], [200, 250]]
    for i, fit_range in enumerate(fit_ranges):
        analyzer.fit_peak(fit_range=fit_range, save_path=os.path.join(test_dir, f'fit_peak_{i}.png'))
    
    # 测试5：对数坐标显示
    print("\n测试5：对数显示功能")
    data = LeCroyDATA(test_file)
    for log_scale in ['logY', 'logXY']:
        data.plot(save_path=os.path.join(test_dir, f'log_scale_{log_scale}.png'), logPlot=log_scale)