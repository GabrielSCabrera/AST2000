import numpy as np
import matplotlib.pylab as plt
import matplotlib.pyplot as plt2
import matplotlib.patches as Patches
import matplotlib.axes as Axes
import math

def integrate_function_vectorized(f, data, a, b, dt = 2e-3):
    x = np.linspace(float(a), float(b), (b-a)/dt)
    x_step_low = dt*f(x[1:], data)
    x_step_high = dt*f(x[:-1], data)
    return np.sum((x_step_low + x_step_high)/2.)

def get_data(filename):
    with open(filename, 'r') as doc:
        lines = doc.readlines()[2:]
    years = np.zeros(len(lines))
    rainfall_totals = years.copy()
    rainfall_per_month = np.zeros((len(lines),12))
    for n,i in enumerate(lines):
        lines[n] = i.replace('\n', ' ')
        lines[n] = lines[n].replace('\t', ' ')
        lines[n] = lines[n].split()
        years[n] = lines[n][0]
        rainfall_totals[n] = lines[n][-1]
        rainfall_per_month[n] = lines[n][1:-1]
    return years, rainfall_totals, rainfall_per_month

def histogram(data, title = 'Histogram', xlabel = 'x', ylabel = 'y', last_width = None):
    x,y = data
    widths = np.diff(a = x)
    fig1 = plt2.figure()
    fig1.canvas.set_window_title(title)
    ax1 = fig1.add_subplot(111, aspect = 'auto')
    ax1.axis([x[0],x[-1],0.,1.1*np.max(y)])
    for n,(i,j,k) in enumerate(zip(x,y,widths)):
        ax1.add_patch(Patches.Rectangle((i,0.), k, j))
    plt2.xlabel(xlabel)
    plt2.ylabel(ylabel)
    plt2.title(title)
    plt2.xticks(x[:-1])
    plt2.show()

def get_month_data(month, rainfall_per_month):
    return rainfall_per_month[:,month-1]

def sort_histogram_data(data, n = 20):
    bins = np.zeros(n+1)
    bins[:-1] = np.linspace(np.min(data), np.max(data), n)
    bins[-1] = np.max(data) + 1.
    data = np.sort(data)
    bin_index = 0
    sorted_data = np.zeros(n)
    for n,i in enumerate(data):
        while True:
            if i >= bins[bin_index] and i <= bins[bin_index + 1]:
                sorted_data[bin_index] += 1
                break
            else:
                bin_index += 1
    return bins, sorted_data

def get_mean_rainfall(rainfall_per_month):
    return np.sum(rainfall_per_month, axis = 1)/12.

def integrate_mean_rainfall(bins, sorted_data, rainfall):
    total_rain = np.sum(rainfall)
    probabilities = (bins[:-1]/total_rain)*sorted_data

def probability_function(x):


years, rainfall_totals, rainfall_per_month = get_data('sydney_rainfall_data.txt')
print get_mean_rainfall(rainfall_per_month)
data = []
for i in range(1,13):
    rainfall = get_month_data(i, rainfall_per_month)
    bins, sorted_data = sort_histogram_data(rainfall)
    print integrate_mean_rainfall(bins, sorted_data, rainfall)
