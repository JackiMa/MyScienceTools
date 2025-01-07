# version 1.0.4
# 2024-04-25
# Author: Ge Ma

# 修订记录 2024年4月25日
# 1. 增加了画对数图的功能，在画图代码中使用命令 logPlot = 'logX', 'logY', 'logXY'

# 修订记录 2024年4月13日
# 1. 修改了画图相关代码，让箭头的位置更加合适
# 2. 修改画图相关代码，使所有画图代码都能在无文件名时直接show
# 3. 修改了对detector的指定，避免有时没有指定detector的报错
# 4. 修改部分局部方法，使之无法从外部访问，增加代码补全的可靠性

# 修改记录 2024年4月9日
# 1. 修改了画图的代码，适用wave_save() 可以直接画图：如果没有指定保存文件名，就直接show

# 修改记录 2024年2月25日
# 1. 增加了保存全部波形而不拟合的选项
# 2. 修改推断文件是能谱还是波形的bug


# 修改记录 2024-01-28
# 1. 修改了接受峰的判断条件，目前只在峰>3时再判断。其实应当将各种判据都糅合起来统一判断
# 2. 将chi2/ndof的值写成了具体是多少chi2/ndof，而不是给出化简后的值。注意，这一步修改了很多关于chi2，ndof传值的过程，可能有bug。比如将chi2_by_ndof,替换成了chi2, ndof,


import re
import numpy as np
import matplotlib.pyplot as plt
import os 
import scienceplots
from matplotlib import font_manager 
font = font_manager.FontProperties(fname=r'C:\Windows\Fonts\segoeui.ttf')
font_manager.fontManager.addfont('C:\\Windows\\Fonts\\segoeui.ttf')
plt.rcParams['font.family'] = 'Segoe UI'  # 使用的字体名字需要和字体文件中的名字匹配


def get_datalist(data_dir, file_type = ['txt']):
    datalists = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            for ftype in file_type:
                if file.endswith(ftype):
                    datalists.append(os.path.join(root, file))


    return datalists

class LeCroyDATA:
    '''
    读取LeCroy的数据文件，返回数据和文件名， 可以画图，常见用法：
    data = LeCroyDATA('F7--PMT--00000.txt')
    data.draw_figure(save_path = None)
    获取数据：data.get_data()
    获取原始数据: data.get_rawdata()
    '''
    def __init__(self, data_dir, delimiters= [',', '\t', ' ', ';'], skiprows=5, xfactor = None, yfactor=1, isSpectrum = None):
        self.data_dir = data_dir
        self.delimiters = delimiters
        self.skiprows = skiprows
        self.raw_data = self.LeCroy_data_read()
        self.data = self.LeCroy_data_read()
        self.title = self.get_name()
        self.xfactor = xfactor
        self.yfactor = yfactor
        if isSpectrum is not None:
            # 可以通过文件名来指定是否是能谱还是波形
            self.isSpectrum = isSpectrum
        else:
            if os.path.basename(self.data_dir).startswith('F'):
                self.isSpectrum = True
            elif os.path.basename(self.data_dir).startswith('C'):
                self.isSpectrum = False
            else:
                print("Unknown data type. Please check the file name. The default type is spectrum")
                self.isSpectrum = True

    def LeCroy_data_read(self):
        for delimiter in self.delimiters:
            try:
                data = np.loadtxt(self.data_dir, skiprows=self.skiprows, delimiter=delimiter)
                self.raw_data = data
                break
            except ValueError:
                pass
        else:
            print(f"Failed to read file {self.data_dir} with provided delimiters.")
            return

        if data.ndim != 2 or data.shape[1] != 2:
            print(f"Data in file {self.data_dir} is not 2D. Returning the first 6 rows of the original data.")
            return data[:6]

        return data

    def scidata_process(self, isregularize = 'none'):

        unitX = ''
        unitY = ''
        if self.isSpectrum: # handle the spectrum histogram data
            unitY = 'counts'
            if self.xfactor is None: # if xfactor is not set, use the auto-generated xfactor
                deltaX = self.data[3,0] - self.data[2,0]
                if deltaX < 1E-12:
                    self.xfactor = 1E12
                    unitX = 'Areas [pV·s]'
                elif deltaX < 1E-7:
                    self.xfactor = 1E9
                    unitX = 'Areas [nV·s]'
                elif deltaX < 1E-3: # 1E-7 < deltaX < 1E-3, 只有时间尺度在us量级才容易有这样的结果
                    self.xfactor = 1
                    unitX = 'Areas [μV·s]'
                elif deltaX < 1:  # 比较大的值肯定是以V为单位了
                    self.xfactor = 1E3
                    unitX = 'Voltage [mV]'
                else:
                    self.xfactor = 1
                    unitX = 'Voltage [V]'
        else:
            unitY = 'voltage [V]'
            if self.xfactor is None: # if xfactor is not set, use the auto-generated xfactor
                deltaX = self.data[3,0] - self.data[2,0]
                if deltaX < 1E-12:
                    self.xfactor = 1E12
                    unitX = 'Time [ps]'
                elif deltaX < 1E-7:
                    self.xfactor = 1E9
                    unitX = 'Time [ns]'
                elif deltaX < 1E-3: 
                    self.xfactor = 1E6
                    unitX = 'Time [μs]'
                elif deltaX < 1:
                    self.xfactor = 1E3
                    unitX = 'Time [ms]'
                else:
                    self.xfactor = 1
                    unitX = 'Time [s]'


        x = self.data[:,0] * self.xfactor
        y = self.data[:,1] * self.yfactor

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

        return x, y, unitX, unitY

    def draw_figure(self, isregularize='none', rebin_factor=10, fontsize = 14, save_path = None):
        print("regularize method: ", isregularize)
        x, y, unitX, unitY = self.scidata_process(isregularize)
        plt.figure()
        if self.isSpectrum: 
            bins = len(x) // rebin_factor
            plt.hist(x, weights=y, bins=bins)  # set bins manually
        else:
            plt.plot(x, y)
        plt.xlabel(unitX, fontsize=fontsize+2)
        plt.ylabel(unitY, fontsize=fontsize+2)
        plt.title(os.path.basename(self.title))
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.grid(True, linestyle='--', alpha=0.5)
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path, dpi = 300)


    def get_rawdata(self):
        return self.raw_data
    
    def get_data(self):
        x, y, unitX, unitY = self.scidata_process(isregularize = 'none')
        return x, y, unitX, unitY

    def get_name(self):
        return self.data_dir.split('\\')[-1]

class Detector:
    '''
    record the basic information about detector
    SiPM
    PMT
    SiPIN
    '''
    def __init__(self, name = "S3590", type = "SiPD", gain = 1, resolution = 0.1, efficiency_spectrum = 1):
        self.name = name
        self.type = type
        self.gain = gain
        self.resolution = resolution
        self.efficiency_spectrum = efficiency_spectrum
    
class Crystal:
    '''
    record the basic information about crystal
    '''
    def __init__(self, name = "GAGG", type = "scintillator", density = 6.63, light_yield = 54, decay_time = 88, emission_spectrum = 1):
        self.name = name
        self.type = type
        self.density = density
        self.light_yield = light_yield
        self.decay_time = decay_time
        self.emission_spectrum = emission_spectrum



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import logging
import os
import shutil

logging.getLogger('matplotlib').setLevel(logging.WARNING)  # 单独设置matplotlib的日志等级，不要显示查询字体的信息

class SpectrumAnalyzer:
    '''
    使用LecroyData获取数据文件，并自动分析能谱，寻峰

    '''
    def __init__(self, data_dir, detector = Detector(name = "R6233", type = "PMT"), crystal = None, filter_window = 1):
        self.data_dir = data_dir
        self.xdata, self.ydata, self.unitX, self.unitY = LeCroyDATA(data_dir).get_data()
        self.ydata =  np.convolve(self.ydata , np.ones(filter_window) / filter_window, mode='same')
        self.detector = detector
        self.crystal = crystal

    def getFilterSigma(self):
        # 根据数据结构找到合适的滤波参数sigma
        # 计算xdata的重心
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
            # logging.debug(f"left_index = {left_index}, right_index = {right_index}, count_sum = {count_sum}, count_sum/sum(spectrum) = {count_sum/sum(spectrum)}")
            if left_index > delta_step:
                left_index -= delta_step
            if right_index < len(self.xdata) - delta_step:
                right_index += delta_step
            if left_index <= delta_step and right_index >= len(self.xdata) - delta_step:
                break
            count_sum = sum(spectrum[left_index:right_index+1])

        self.data_width = self.xdata[right_index] - self.xdata[left_index]
        # 计算这个范围的长度的1%
        sigma = (right_index - left_index) * 0.2
        logging.debug(f"\nthe filter used sigma = {sigma}")
        return sigma 

    def gaussian(self, x, amplitude, mean, stddev):
        return amplitude * np.exp(-((x - mean)**2 / (2 * stddev**2)))

    def gaussian_plus_linear(self, x, amplitude, mean, stddev, slope, intercept):
        return amplitude * np.exp(-((x - mean)**2 / (2 * stddev**2))) + slope * x + intercept

    def value_to_index(self, value):
        # return the index of the closest value in the xdata
        return np.abs(self.xdata - value).argmin()
        
    def fit_peak(self, fit_range, p0=[1,0,1,0,0]):
        '''
        适用gaussian_plus_linear拟合峰，
        手动指定拟合范围fit_range，和给出拟合初值p0
        画图并在图中给出拟合结果
        p0 = [amplitude, mean, stddev, slope, intercept]
        '''
        left = self.value_to_index(fit_range[0])
        right = self.value_to_index(fit_range[1])

        if right - left < 1:
            raise ValueError("Fit range is too small. Please select a larger range.")

        amp, mean, stddev, slope, intercept = p0
        if amp == 1:
            amp = 0.6*max(self.ydata[left:right])
        if mean == 0:
            mean = self.xdata[np.argmax(self.ydata[left:right])] + self.xdata[left]
            # print("left = ", left, "right = ", right, "mean = ", mean, "xdata = ", self.xdata[left:right])
            # mean = (self.xdata[left] + self.xdata[right]) / 2
            
        if stddev == 1:
            stddev = (right - left) * 0.1

        p0 = [amp, mean, stddev, slope, intercept]

        try:
            popt, pcov = curve_fit(self.gaussian_plus_linear, self.xdata[left:right], self.ydata[left:right], p0=p0)
            perr = np.sqrt(np.diag(pcov))

            # Calculate chi2
            residuals = self.ydata[left:right] - self.gaussian_plus_linear(self.xdata[left:right], *popt)
            chi2 = np.sum((residuals ** 2) / self.gaussian_plus_linear(self.xdata[left:right], *popt))
            Ndof = len(self.xdata[left:right]) - len(popt)

            title = os.path.basename(self.data_dir) + f" peak at {popt[1]:.2f}±{popt[2]:.2f}"
            fit_label = f'Gauss Fit: peak at {popt[1]:.2f}±{popt[2]:.2f}'
        except RuntimeError:
            title = os.path.basename(self.data_dir) + " Fit Failed"
            fit_label = "Fit Failed"

        with plt.style.context(['science', 'ieee','std-colors','grid','no-latex']):
            plt.rcParams['font.family'] = 'Segoe UI'
            plt.figure(figsize=(10, 6))
            plt.plot(self.xdata, self.ydata, label='Measured Spectrum')
            if 'popt' in locals():
                plt.plot(self.xdata[left:right], self.gaussian_plus_linear(self.xdata[left:right], *popt), label=fit_label)
            else:
                plt.plot(self.xdata[left:right], self.ydata[left:right], 'r', label='Fit Range')
            plt.legend(fontsize=12)
            plt.title(title,fontsize=20)
            plt.xlabel(self.unitX,fontsize=16)
            plt.ylabel(self.unitY,fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            # Add fit results to the plot
        if 'popt' in locals():
            fit_function = f'Fit function:\nGaussian: $f(x) = {popt[0]:.2f} \cdot e^{{-0.5 \cdot ((x - {popt[1]:.2f}) / {popt[2]:.2f})^2}}$\nLinear: $f(x) = {popt[3]:.2f} \cdot x + {popt[4]:.2f}$'
            plt.text(0.05, 0.95, fit_function + f'\nChi2: {chi2:.2f}\nNdof: {Ndof}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

            plt.show()

        if 'popt' in locals():
            return popt, perr


    def _fit_gaussian(self, peak, spectrum, x_index):
        # Define fitting range
        fit_range = float(0.5 * np.sqrt(abs(peak)))*np.sqrt(self.xdata[-1] - self.xdata[0])*0.1 # 以峰值为中心，向两边取0.5倍的标准差
        logging.debug(f"the raw fit range is {fit_range}")
        fit_range = max(fit_range, self.data_width * 0.01) # 最小取10个点
        logging.debug(f"the real fit range is {fit_range}")
        start = max(self.xdata[0], peak - fit_range)
        end = min(self.xdata[-1], peak + fit_range)
        # record the peak and start, end
        logging.debug(f"peak = {peak}, start = {start}, end = {end}")

        start_index = self.value_to_index(start)
        end_index = max(self.value_to_index(end), start_index+1)
        peak_index = self.value_to_index(peak)

        # Fit Gaussian function
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            # 画出拟合范围，用透明度为0.3的红色色块
            plt.axvspan(start, end, alpha=0.2, color='red')
        
        # bounds: [amplitude, mean, stddev, slope, intercept]
        lb = [0,                       0.5*start,           -np.inf,        -np.inf,    -np.inf]
        ub = [1.5*max(self.ydata),     2*end,               np.inf,         np.inf,     np.inf]
        x0 = [spectrum[peak_index],    peak,                1/2*fit_range,  0,          0]
        x0 = [round(x, 4) for x in x0]
        lb = [round(x, 4) for x in lb]
        ub = [round(x, 4) for x in ub]

        logging.debug(f"The bounds of fit for peak at {self.xdata[peak_index]}: {lb}, {ub}")
        logging.debug(f"Initial guess for peak at {self.xdata[peak_index]}: {x0}")
        logging.debug(f"start = {start}, end = {end}, start_index = {start_index}, end_index = {end_index}")
        logging.debug(f"self.xdata[start_index:end_index] = {self.xdata[start_index:end_index]}")
        logging.debug(f"spectrum[start_index:end_index] = {spectrum[start_index:end_index]}")
        
        # First fit
        try:
            popt1, pcov1 = curve_fit(self.gaussian_plus_linear, self.xdata[start_index:end_index], spectrum[start_index:end_index], 
                                    p0=x0, bounds=(lb, ub), maxfev=5000)
        except RuntimeError as e:
            logging.error(f"Error in first curve fitting: {str(e)}")
            return None

        # Calculate chi2/ndof for the first fit
        expected_value1 =  self.gaussian_plus_linear(self.xdata[start_index:end_index], *popt1)
        residuals1 = spectrum[start_index:end_index] - expected_value1
        chi2_1 = np.sum(residuals1**2 / expected_value1)
        ndof1 = len(residuals1) - len(popt1)
        chi2_by_ndof1 = chi2_1 / ndof1
        perr1 = np.sqrt(np.diag(pcov1))
        logging.debug(f"The first Fit results: popt={popt1}, chi2/ndof={chi2_1} / {ndof1} = {chi2_by_ndof1:.2f}")
        logging.debug(f"Standard errors for peak at {self.xdata[peak_index]}: {perr1}")

        # Save the indices for the first fit
        start_index1 = start_index
        end_index1 = end_index

        # Reset the fit_range and fit again according to the sigma
        peak = popt1[1]
        if self.detector.type == "PMT":
            fit_range = abs(5 * popt1[2])
        else:
            fit_range = min(abs(7 * popt1[2]), 1.5*fit_range) # 以峰值为中心，向两边总共取7倍的标准差
        logging.debug(f"The detector type is {self.detector.type}, New fit range for peak at {self.xdata[peak_index]} = {fit_range}")
        start = max(self.xdata[0], popt1[1] - 0.9*fit_range)
        end = min(self.xdata[-1], popt1[1] + 1.1*fit_range)
        start_index = self.value_to_index(start)
        end_index = self.value_to_index(end)
        peak_index = self.value_to_index(peak)

        # Second fit
        try:
            popt2, pcov2 = curve_fit(self.gaussian_plus_linear, self.xdata[start_index:end_index], spectrum[start_index:end_index], 
                                    p0=[spectrum[peak_index], self.xdata[peak_index], 1/2*fit_range, 0, 0], bounds = (lb, ub),maxfev=5000)
        except (RuntimeError, ValueError)  as e:
            perr2 = np.sqrt(np.diag(pcov1))
            chi2_by_ndof2 = 0
            logging.error(f"Error occurred during the more detailed curve fitting: {str(e)}")
            return popt1, perr1, chi2_1, ndof1, start_index1, end_index1

        # Calculate chi2/ndof for the second fit
        expected_value2 =  self.gaussian_plus_linear(self.xdata[start_index:end_index], *popt2)
        residuals2 = spectrum[start_index:end_index] - expected_value2
        chi2_2 = np.sum(residuals2**2 / expected_value2)
        ndof2 = len(residuals2) - len(popt2)
        chi2_by_ndof2 = chi2_2 / ndof2
        perr2 = np.sqrt(np.diag(pcov2))
        logging.debug(f"The second Fit results: popt={popt2}, chi2/ndof={chi2_by_ndof2:.2f}")
        logging.debug(f"Standard errors for peak at {self.xdata[peak_index]}: {perr2}")

        # Compare the errors and return the best fit
        if np.sum(perr1) < np.sum(perr2):
            return popt1, perr1, chi2_1, ndof1, start_index1, end_index1
        else:
            return popt2, perr2, chi2_2, ndof2, start_index, end_index


    def _find_matching_peaks(self, peaks_list = [], expected_ratios = [], fluctuation = 0.1):
        '''
        寻找匹配的峰
        :param peaks_list: 峰的位置, 例如 [12.1, 30]
        :param expected_ratios: 期望的峰值比例, 例如 [1274/511]
        :param fluctuation: 允许的峰值比例的波动范围
        :return: 匹配的峰的位置, 例如 [(12.1, 30)]
        '''
        matching_pairs = []
        debug = logging.getLogger().getEffectiveLevel() == logging.DEBUG
        for i in range(len(peaks_list)):
            for j in range(i+1, len(peaks_list)):
                ratio = peaks_list[j] / peaks_list[i]
                for expected_ratio in expected_ratios:
                    diff = abs(ratio - expected_ratio)
                    limit = fluctuation * expected_ratio
                    if debug:
                        logging.debug(f"For peaks {peaks_list[i]} and {peaks_list[j]}: Their ratio is {ratio}, Expected ratio is {expected_ratio}, Difference is {diff}, Limit is {limit}")
                    if np.all(diff <= limit):
                        matching_pairs.append((peaks_list[i], peaks_list[j]))
                        logging.info(f"Matching peaks found: {peaks_list[i]}, {peaks_list[j]}")
                        break
                else:
                    continue
                break
        return matching_pairs


    def _find_and_fit_peaks(self, spectrum, x_index, expected_ratios):
        # Function name for logging
        func_name = "find_and_fit_peaks: "
        logging.debug(func_name + "Starting...")

        # Get the sigma for the Gaussian filter
        sigma = self.getFilterSigma()
        # Apply the Gaussian filter to the spectrum
        spectrum_smooth = gaussian_filter1d(spectrum, sigma=sigma)
        # Find the peaks in the smoothed spectrum
        peaks_index, _ =  find_peaks(spectrum_smooth, distance=20, width = 10, height = 18, prominence=2)  
        
        # Calculate the centroid of the image
        centroid = np.average(self.xdata, weights=self.ydata)
        # Only consider peaks to the right of the centroid
        peaks_index = peaks_index[self.xdata[peaks_index] > centroid*0.3]
        logging.debug(func_name + f"0.3 * centroid is {0.3 * centroid}, peaks_index is {peaks_index}")

        # Get the x values of the peaks
        peaks_lists = self.xdata[peaks_index]
        self.peaks_index = peaks_index

        # If the log level is set to DEBUG, save and plot the original and smoothed spectra
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            plt.figure(figsize=(10, 6))
            plt.plot(self.xdata, spectrum, label='Original')
            plt.plot(self.xdata, spectrum_smooth, label='Smoothed')
            plt.plot(peaks_lists, spectrum_smooth[peaks_index], "x")
            plt.legend()
            logging.debug(func_name + f'Found peaks at: {peaks_lists}')

        # Find the matching peaks
        matching_pairs =  self._find_matching_peaks(peaks_lists, expected_ratios, fluctuation=0.15)

        # Initialize the results list
        fit_results = []
        logging.debug(func_name + f"Matching pairs: {matching_pairs}")

        # If no matching pairs were found, log a message and return
        if len(matching_pairs) == 0:
            logging.info(func_name + f"Expected peaks not found in file {self.data_dir}")
            return matching_pairs, fit_results

        # Copy the matching pairs list
        matching_pairs_copy = matching_pairs.copy()

        # Initialize the list to store the results for this pair
        peak_results = []
        num = 0

        # For each pair in the matching pairs
        for pair in matching_pairs_copy:
            # For each peak in the pair
            for peak in pair:
                num += 1

                # Only consider the first few peaks, as there may be very small interference peaks in the high energy area
                if num > 6:
                    break

                # Fit a Gaussian to the peak
                result = self._fit_gaussian(peak, spectrum, x_index)

                # If the result is None, skip this iteration
                if result is None:
                    logging.error("Error occurred during the more detailed curve fitting. Skipping this iteration.")
                    continue

                popt, perr, chi2, ndof, start_index, end_index = result
                
                # Add the results to the peak results list
                peak_results.append((popt, perr, chi2, ndof, start_index, end_index, peak))

        # Sort the results by peak position
        peak_results.sort(key=lambda x: x[5])

        # Get the first three results
        first_three_results = peak_results[:3]

        # Find the result with the smallest sigma
        min_sigma_result = min(first_three_results, key=lambda x: abs(x[0][2]))
        min_sigma_peak = min_sigma_result[5]

        for pair in matching_pairs_copy:
            logging.debug(func_name + f"Processing pair: {pair}")
            pair_is_valid = True  # Assume the pair is valid until proven otherwise
            pair_results = []  # Store the results for this pair

            for peak in pair:
                # Fit a Gaussian to the peak and get standard errors
                result = self._fit_gaussian(peak, spectrum, x_index)

                # If the result is None, skip this iteration
                if result is None:
                    logging.error("Error occurred during the more detailed curve fitting. Skipping this iteration.")
                    continue
                popt, perr, chi2, ndof, start_index, end_index = result

                # Check if chi2/ndof > 2 or < 0.5 (according to goodness of fit)
                chi2_by_ndof = chi2/ndof
                if chi2_by_ndof > 20 or chi2_by_ndof < 0.5:
                    pair_is_valid = False
                    logging.info(func_name + f"Removed pair {pair} due chi2/ndof fit at peak {peak} is {chi2_by_ndof:.2f}, which means the fit is not good enough")
                    break  # No need to check the other peaks in this pair

                # Check if sigma^2/mean > 0.12 (according to energy resolution)
                if (popt[2]**2 / popt[1]) > 0.5:
                    pair_is_valid = False
                    logging.info(func_name + f"Removed pair {pair} due to sigma^2/mean ratio at peak {peak} is {popt[2]**2 / popt[1]:.2f}, which means the energy resolution is too low")
                    break  # No need to check the other peaks in this pair

                # Check if the current peak's sigma is the smallest
                # If the current peak is to the left of the peak with the smallest sigma, discard it
                if peak < min_sigma_peak and abs(popt[2]) > peak_results[0][0][2]:
                    pair_is_valid = False
                    logging.debug(func_name + f"min_sigma_peak is {min_sigma_peak}, current peak is {peak}, sigma is {popt[2]}, smallest sigma is {peak_results[0][0][2]}")
                    logging.info(func_name + f"Removed pair {pair} due to peak {peak} is to the left of the peak with the smallest sigma")
                    break

                # If the peak passed all checks, store its results
                pair_results.append((popt, perr, chi2, ndof, start_index, end_index))

            # If the pair is still valid after checking all peaks, save its results
            if pair_is_valid:
                fit_results.extend(pair_results)
                for result in pair_results:
                    popt, perr, chi2, ndof, start_index, end_index = result
                    logging.debug(func_name + f"Accepted pair {pair} with peak at {popt[1]}, chi2/ndof = {chi2_by_ndof:.2f}, sigma^2/mean = {popt[2]**2 / popt[1]:.2f}")
                    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                        plt.plot(self.xdata[start_index:end_index],  self.gaussian_plus_linear(self.xdata[start_index:end_index], *popt), label=f'Gauss Fit: peak at {popt[1]:.2f}±{popt[2]:.2f}')
                        plt.annotate(f'peak = {popt[1]:.2f}±{popt[2]:.2f}', xy=(0.4, 0.9 - annotation_counter * 0.04), xycoords='axes fraction')
                        annotation_counter += 1
            else:
                matching_pairs.remove(pair)

        logging.debug(func_name + f"Matching pairs after filtering: {matching_pairs}")
        logging.debug(func_name + f"Fit results: {fit_results}")
        logging.debug(func_name + "Finished.\n\n")
        # Add legend after the loop
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            plt.legend()

        return matching_pairs, fit_results

    def _find_all_peaks(self, spectrum, x_index):
        func_name = "find_all_peaks: "
        logging.debug(func_name + "Starting...")

        sigma = self.getFilterSigma()
        logging.debug(func_name + f"Filter sigma: {sigma}")

        spectrum_smooth = gaussian_filter1d(spectrum, sigma=sigma)
        peaks_index, _ =  find_peaks(spectrum_smooth, distance=10, width = 3, height = 10, prominence=2)  

        peaks_lists = self.xdata[peaks_index]
        self.peaks_index = peaks_index

        # 如果日志级别设置为DEBUG，则保存和绘制滤波前后的光谱
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            plt.figure(figsize=(10, 6))
            plt.plot(self.xdata, spectrum, label='Original')
            plt.plot(self.xdata, spectrum_smooth, label='Smoothed')
            plt.plot(peaks_lists, spectrum_smooth[peaks_index], "x")
            plt.legend()
            logging.debug(func_name + f'Found peaks at: {peaks_lists}')
            logging.debug(func_name + "-"*50)  # Add a separator line

        fit_results = [] # (popt, perr, chi2, ndof, start, end)
        annotation_counter = 0
        for peak in peaks_lists:
            logging.debug(f"\n")
            logging.debug(func_name + f"Processing peak: {peak}")
            # record the brief information about peak, spectrum, x_index
            logging.debug(func_name + f"peak: {peak}, spectrum: {spectrum[:5]}..., x_index: {x_index[:5]}...")
            result = self._fit_gaussian(peak, spectrum, x_index)
            # If the result is None, skip this iteration
            if result is None:
                logging.error("Error occurred during the more detailed curve fitting. Skipping this iteration.")
                continue
            popt, perr, chi2, ndof, start_index, end_index = result
            
            chi2_by_ndof = chi2/ndof
            logging.debug(func_name + f"Fit results: popt={popt}, perr={perr}, chi2_by_ndof={chi2}/{ndof}={chi2_by_ndof}, start_index={start_index}, end_index={end_index}")

            if len(peaks_lists) > 3:
                logging.info(func_name + f"There are {len(peaks_lists)} peaks in the spectrum, which is too many. The spectrum may be noisy. Please check the spectrum and the fit results.")
                # 只有峰的数量太多时，才根据下面的规则删除。
                # 应当改成设置某个判据，以下面这些参数和峰的数量作为标准来判断是否要删除
                if chi2_by_ndof > 20 or chi2_by_ndof < 0.5:
                    logging.info(func_name + f"!!! Removed peak {peak} due chi2/ndof fit at peak {peak} is {chi2}/{ndof}={chi2_by_ndof:.2f}, which means the fit is not good enough")
                    continue
                if (popt[2]**2 / popt[1]) > 0.5:
                    logging.info(func_name + f"!!! Removed peak {peak} due to sigma^2/mean ratio at peak {peak} is {popt[2]**2 / popt[1]:.2f}, which means the energy resolution is too low")
                    continue
            fit_results.append((popt, perr, chi2, ndof, start_index, end_index))
            logging.debug(func_name + f"Accepted peak {peak} with peak at {popt[1]}, chi2/ndof = {chi2}/{ndof}={chi2_by_ndof:.2f}, sigma^2/mean = {popt[2]**2 / popt[1]:.2f}")
            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                plt.plot(self.xdata[start_index:end_index],  self.gaussian_plus_linear(self.xdata[start_index:end_index], *popt), label=f'Gauss Fit: peak at {popt[1]:.2f}±{popt[2]:.2f}')
                plt.annotate(f'peak = {popt[1]:.2f}±{popt[2]:.2f}', xy=(0.4, 0.9 - annotation_counter * 0.04), xycoords='axes fraction')
                annotation_counter += 1
            logging.debug(func_name + "-"*50)  # Add a separator line

        logging.debug(func_name + f"Final fit results: {fit_results}")
        # Add legend after the loop
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            plt.legend()

        logging.debug(func_name + "Finished.\n\n")
        return fit_results


    def _plot_spectrum_with_peaks(self, spectrum, x_index, fit_results, save_path=None, fontsize = 16, logPlot = False):
        with plt.style.context(['science', 'ieee','std-colors','grid','no-latex']):
            plt.rcParams['font.family'] = 'Segoe UI'
            plt.figure(figsize=(10, 6))
            plt.plot(self.xdata, spectrum, label='Measured Spectrum')
            for result in fit_results:
                popt, perr, chi2, ndof, start, end = result  # get standard errors
                # Use gaussian_plus_linear function instead of gaussian
                plt.plot(self.xdata[start:end],  self.gaussian_plus_linear(self.xdata[start:end], *popt), label=f'μ={popt[1]:.2f}±{perr[1]:.2f}, σ={popt[2]:.2f}±{perr[2]:.2f}, res={2.355*popt[2] / popt[1]:.2f}, χ2/ndof={chi2:.2f}/{ndof}')  # use standard error for error bar
            plt.legend(fontsize=fontsize-4)
            # titelString = 'Energy Spectrum with Gaussian Fits'
            titelString = os.path.basename(self.data_dir) + ' Gaussian Fits'
            if save_path:
                titelString = os.path.basename(save_path)
            plt.title(titelString,fontsize=fontsize+4)
            # 画出所有的peaks值，并在旁边标上坐标
            for peak_index in self.peaks_index:
                arrow_y = min(spectrum[peak_index]+0.1*plt.gca().get_ylim()[1], 0.9*plt.gca().get_ylim()[1])
                plt.annotate(f'{self.xdata[peak_index]:.2f}', 
                            xy=(self.xdata[peak_index], spectrum[peak_index]), 
                            xycoords='data', 
                            xytext=(self.xdata[peak_index] + 0.02 * (plt.gca().get_xlim()[1] - plt.gca().get_xlim()[0]), arrow_y ), 
                            textcoords='data',
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3", clip_on=True))
                plt.plot(self.xdata[peak_index], self.ydata[peak_index], "*", color = 'red')
            plt.xlabel(self.unitX,fontsize=fontsize+4)
            plt.ylabel(self.unitY,fontsize=fontsize+4)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            if logPlot != False:
                if logPlot == 'logY':
                    plt.yscale('log')
                elif logPlot == 'logX':
                    plt.xscale('log')
                elif logPlot == 'logXY':
                    plt.yscale('log')
                    plt.xscale('log')
            if save_path:
                plt.savefig(save_path, dpi = 300)
            else:
                plt.show()
            if logging.getLogger().getEffectiveLevel() != logging.DEBUG:
                plt.close()

    def _plot_spectrum(self, spectrum, x_index, save_path=None, fontsize = 16, logPlot=False):
        with plt.style.context(['science', 'ieee','std-colors','grid','no-latex']):
            plt.rcParams['font.family'] = 'Segoe UI'
            plt.figure(figsize=(10, 6))
            plt.plot(self.xdata, spectrum, label='Measured Spectrum')
            plt.legend(fontsize=fontsize-4)
            if save_path:
                titelString = os.path.basename(save_path)
            else:
                titelString = os.path.basename(self.data_dir)
            plt.title(titelString,fontsize=fontsize+4)
            plt.xlabel(self.unitX,fontsize=fontsize+4)
            plt.ylabel(self.unitY,fontsize=fontsize+4)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            if logPlot != False:
                if logPlot == 'logY':
                    plt.yscale('log')
                elif logPlot == 'logX':
                    plt.xscale('log')
                elif logPlot == 'logXY':
                    plt.yscale('log')
                    plt.xscale('log')
            if save_path:
                plt.savefig(save_path, dpi = 300)
            else:
                plt.show()
            if logging.getLogger().getEffectiveLevel() != logging.DEBUG:
                plt.close()

    def analysis_twopeaks_save(self, png_path=None):
        '''
        保存所有波形、能谱。如果是能谱，会尝试使用511/1274的关系拟合找到峰位
        '''
        spectrum = self.ydata
        x_index = np.arange(len(spectrum))
        # 期望的峰值比例，这里以511峰为基准
        expected_ratios = [1274/511]
        # 寻峰并拟合
        matching_pairs, fit_results = self._find_and_fit_peaks(spectrum, x_index, expected_ratios)
        logging.debug(f"matching_pairs: {matching_pairs}")
        
        # 如果发现的pair == 1，保存图片，并做好记录
        if len(matching_pairs) == 1:
            logging.info(f"All expected peaks found in file {self.data_dir}")
            self._plot_spectrum_with_peaks(spectrum, x_index, fit_results, png_path)
        # 如果发现的峰数量不匹配，复制文件到新的文件夹内，并保存图片到这个文件夹，便于后续人工处理
        else:
            try:
                # Split the file name and extension
                file_name, file_extension = os.path.splitext(png_path)
                # If the file extension is .png, add WARNING before the extension
                if file_extension == '.png':
                    png_path = f"{file_name}_WARNING{file_extension}"
            except TypeError:
                # If png_path is None, do nothing
                pass
            logging.info(f"Expected peaks not found in file {self.data_dir}")
            self._plot_spectrum_with_peaks(spectrum, x_index, fit_results, png_path)

    def allpeaks_save(self, png_path=None, logPlot = False):
        '''
        保存所有波形、能谱。如果是能谱也保存所有拟合的峰信息
        logPlot = 'logY' or 'logX' or 'logXY'
        '''
        spectrum = self.ydata
        x_index = np.arange(len(spectrum))
        fit_results = self._find_all_peaks(spectrum, x_index)    
        self._plot_spectrum_with_peaks(spectrum, x_index, fit_results, png_path, logPlot = logPlot)

    def wave_save(self, png_path = None, logPlot = False):
        '''
        保存且仅保存所有波形图
        logPlot = 'logY' or 'logX' or 'logXY' 
        '''
        spectrum = self.ydata
        x_index = np.arange(len(spectrum))
        self._plot_spectrum(spectrum, x_index, png_path, logPlot = logPlot)



if __name__ == "__main__":

    # 设置日志文件
    # 清除所有现有的处理器
    # logging.getLogger().handlers = []
    # # 创建一个处理器，将日志消息输出到标准输出
    # # handler = logging.StreamHandler()
    # handler = logging.FileHandler('log.txt')
    # # 将处理器添加到根日志记录器
    # logging.getLogger().addHandler(handler)
    # # 设置日志级别为DEBUG
    # logging.getLogger().setLevel(logging.DEBUG)


    # detector = Detector(name = "R6233", type = "PMT")
    # fileName = r'data\MGTest\R2083waves\C3---10dB+TVS+R2083@2500V+LYSO225+Na22--00007.txt'
    # data1 = SpectrumAnalyzer(fileName, detector=detector)
    # # data1.analysis_twopeaks_save('test.png')
    # data1.wave_save('test.png') # 与右边基本等价 LeCroyDATA(fileName, isSpectrum=False).draw_figure(save_path = 'test2.png')
    # # data1.allpeaks_save('test.png')

    # data2 = LeCroyDATA(fileName, isSpectrum=False)
    # data2.draw_figure(save_path = 'test2.png')

    SiPIN_Am241 = SpectrumAnalyzer(r'240408+SiPIN\241Am+SiPIN--00000.csv')

    SiPIN_Am241.fit_peak(fit_range=[2000, 2500])