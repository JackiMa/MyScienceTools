import re
import numpy as np
import matplotlib.pyplot as plt
import os 

def get_datalist(data_dir, file_type = ['txt']):
    datalists = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            for ftype in file_type:
                if file.endswith(ftype):
                    datalists.append(os.path.join(root, file))


    return datalists

class LeCroyDATA:
    def __init__(self, data_dir, delimiters= [',', '\t', ' ', ';'], skiprows=5,xfactor = 1, yfactor=1):
        self.data_dir = data_dir
        self.delimiters = delimiters
        self.skiprows = skiprows
        self.raw_data = self.LeCroy_data_read()
        self.data = self.LeCroy_data_read()
        self.title = self.get_name()
        self.metrics = self.get_metrics()
        self.xfactor = xfactor
        self.yfactor = yfactor


        self.rebin_factor = 10 # rebinning factor, 将原来的bin数除以10

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

        return x, y

    def draw_figure(self, isregularize='none'):
        print("regularize method: ", isregularize)
        x,y = self.scidata_process(isregularize)

        plt.figure()
        if os.path.basename(self.data_dir).startswith('F'): # 'F' means histogram
            bins = len(x) // self.rebin_factor
            plt.hist(x, weights=y, bins=bins)  # set bins manually
            plt.xlabel('Areas [nV·s]')
            plt.ylabel('counts')
        else:
            plt.plot(x, y)
            plt.xlabel('x-axis')
            plt.ylabel('y-axis')
        plt.title(os.path.basename(self.title))

    def get_rawdata(self):
        return self.raw_data
    
    def get_data(self):
        return self.data

    def get_name(self):
        return self.data_dir.split('\\')[-1]

    def get_metrics(self):
        metrics = {}
        temperature_match = re.search(r'tmp(n?\d+)', self.title)
        if temperature_match:
            metrics['temperature'] = temperature_match.group(1)
            if 'n' in metrics['temperature']: # negative temperature
                metrics['temperature'] = '-' + metrics['temperature'].replace('n', '')
        # Add more metrics here if needed
        return metrics