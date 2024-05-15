import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats as st
from os.path import basename
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# USER INPUT #
lipid_type = 'DPPC'
excel_file_names = [ 'Nanodisc_spectra_DPPC_10282023.xlsx', 'Nanodisc_Spectra_DPPC_12132023.xlsx', 'Nanodisc_Spectra_DPPC_020624.xlsx'  ]
membrane_platforms = [ 'LUV', 'spMSP1D1', 'spNW15', 'spNW25', 'spNW50' ] # the names of the sheets in excel
color_main = [ '#BBBBBB', '#66CCEE', '#228833', '#CCBB44', '#AA3377'  ]
filetype = 'pdf'
expected_melting_temp = 41.0 # The experimental melting temperature of the lipid

# PROCEDURES #
def read_in_excel_files(excel_file_names):
	# excel_file_names: names of the excel files, which 
	# are input by user above in the 'USER INPUT' section
	df_list = {basename(f) : pd.read_excel(f,engine='openpyxl',keep_default_na=False,sheet_name=None) for f in excel_file_names }	
	return df_list

def get_clean_column(datasheet,column):
	# Takes the column in question and strips out any non-numeric
	# entries.
	# datasheet: Specific datasheet in the workbook you want evaluated
	# column : Header of the column to be evaluated
	dirty_column = datasheet[column]
	clean_column = list( pd.to_numeric(dirty_column,errors='coerce').dropna() ) # 'coerce' changes non-numeric to NaN and 'dropna()' removes NaN from list
	clean_column = np.unique(clean_column).tolist()
	return clean_column

def get_clean_headers(datasheet):
	# Gets the headers from the datasheet and strips any 
	# non-numeric values. Puts this into a list for data analysis.
	# datasheet: Specific datasheet in the workbook you want evaluated
	dirty_headers = datasheet.keys()
	clean_headers = list( pd.to_numeric(dirty_headers,errors='coerce').dropna() )
	return clean_headers

def get_column_data(datasheet,column_header,independent_dimension):
	# Collects the raw data in the column with the appropriate column
	# header
	column_data = datasheet[column_header]
	if len(column_data)==independent_dimension:
		return column_data
	else:
		column_data = column_data[0:independent_dimension]
		return column_data

def plot_spectrum_data(xaxis,data,keys,figure_name):
	# Plots the spectrum data for one membrane type and one
	# worksheet
	# xaxis : A list of the independent variable
	# data : A dictionary of the spctrum data
	# keys : A list of keys to access the data from the dictionary
	# filename : Name in which the file will be saved
	number_of_traces = len(keys)
	colors = plt.cm.rainbow(np.linspace(0,1,number_of_traces))
	for i in range(len(keys)):
		iteration = keys[i]
		trace_label = str(iteration) + ' C'
		plt.plot(xaxis,data[iteration],color=colors[i],label=trace_label)
	plt.xlabel('Wavelength (nm)')
	plt.ylabel('Intensity (cps)')
	plt.title(figure_name)
	plt.legend(frameon=False,ncol=4)
	plt.savefig(figure_name,format=filetype)
	#plt.show()
	plt.clf()

def get_gp_row(datasheet,column):
	# Takes the row with generalized polarization values in
	# the datasheet. In the case that there are more than one
	# row of GP values, it takes the first.
	column_values = datasheet[column]
	gp_row_index = column_values[column_values=='GP'].index[0]
	gp_row = datasheet.iloc[142]
	gp_row_clean = list( pd.to_numeric(gp_row,errors='coerce').dropna() )
	return gp_row_clean

def plot_gp_data(xaxis,data,number_of_runs,membrane_type):
	# Takes the generalized polarization data and plots each run
	# individually, along with average +/- std dev
	# xaxis : The independent axis values as a list
	# data : The generalized polarization data to be plotted
	# number_of_runs : The number of independent experiments performed
	# membrane_type : Nanodisc construct or LUV for which data is being plotted
	colors = [ '#EECC66', '#6699CC', '#994455', '#000000'  ]
	figure_name = membrane_type + '_GenPol_' + lipid_type + '.' + str(filetype)
	for i in range(number_of_runs):
		trace_label = 'n=' + str(i+1)
		plt.plot(xaxis,data[i],color=colors[i],marker='o',linestyle='None',label=trace_label)
	i += 1
	plt.errorbar(xaxis,data[i],yerr=data[i+1],color=colors[i],ecolor=colors[i],marker='o',linestyle='None',label='Average')
	plt.xlabel('Temperature (C)')
	plt.ylabel('Generalized Polarization')
	plt.title(membrane_type)
	plt.legend(frameon=False)
	plt.ylim(-0.51,0.51)
	plt.savefig(figure_name,format=filetype)
	#plt.show()
	plt.clf()

def calc_first_derivative_central(xdata,ydata):
	# Calculates the first derivative of the data provided
	# and returns a list with the derivative. This uses the
	# central finite element difference. Therefore, the first 
	# and last independent varaible points will not have 
	# a derivative.
	# xdata : The independent variable which you
	# are differentiating against
	# ydata : The dependent variable you are differentiating
	step_size = 2 * ( xdata[1] - xdata[0] )
	xdata_deriv = [ ]
	ydata_deriv = [ ]
	for i in range(1,(len(xdata)-1)):
		#print(xdata[i])
		xdata_deriv.append(xdata[i])
		difference = ydata[i+1] - ydata[i-1]
		derivative = difference / step_size
		#print(derivative)
		ydata_deriv.append(derivative)
	return xdata_deriv,ydata_deriv

def boltzmann_curve_fit_func(T,GP1,GP2,Tmelt,Tslope,asymm_param):
	return GP2 + ( ( GP1 - GP2 ) / ( 1 + np.exp((T-Tmelt)*Tslope) )**asymm_param )

def boltzmann_curve_fit(temp_data,gp_data,expected_melting_temp):
	# Takes in the temperature and gp data
	# to obtain GP1 and GP2 for the curve fitting 
	# procedure.

	epsilon = 0.00001
	GP1 = np.max(gp_data)
	GP2 = np.min(gp_data)
	guess = [ GP1, GP2, expected_melting_temp, 1.0,1.0 ]
	gp_bounds = ( [GP1-epsilon,GP2-epsilon,-np.inf,-np.inf,-np.inf], [GP1+epsilon,GP2+epsilon,np.inf,np.inf,np.inf] )
	popt,pconv = curve_fit(boltzmann_curve_fit_func,temp_data,gp_data,p0=guess,bounds=gp_bounds,ftol=1e-10)
	print(popt)
	return popt,pconv

# EXECUTION #
# Reads in excel files #
excel_files = read_in_excel_files(excel_file_names)

# Creates spectrum plots for each nanodisc condition
for i in range(len(membrane_platforms)):
	membrane_type = membrane_platforms[i]
	k = 1 # Numerical placeholder for figure names
	data_dictionary = dict()
	for j in range(len(excel_file_names)):
		print('Examining',membrane_type,'in',excel_file_names[j],'and plotting temperature-dependent spectra')
		workbook = excel_files[excel_file_names[j]]
		datasheet = workbook[membrane_type]
		wavelengths = get_clean_column(datasheet,'Wave')
		temperatures = get_clean_headers(datasheet)
		for l in range(len(temperatures)):
			# Gathers the spectrum at each temperature and adds it to the dictionary
			spectrum_at_temperature = get_column_data(datasheet,int(temperatures[l]),len(wavelengths))
			data_dictionary[int(temperatures[l])] = spectrum_at_temperature
			#print(spectrum_at_temperature)
		figure_name = str(membrane_type) + '_' + str(lipid_type) + '_' + str(k) + '.' + str(filetype)
		plot_spectrum_data(wavelengths,data_dictionary,temperatures,figure_name)
		k += 1
		
# Creates GP plots for each nanodisc condition
averages_dictionary = dict()
stdv_dictionary = dict()
for i in range(len(membrane_platforms)):
	membrane_type = membrane_platforms[i]
	data_dictionary = dict()
	k = 0
	for j in range(len(excel_file_names)):
		print('Examining',membrane_type,'in',excel_file_names[j],'and plotting generalized polarization')
		workbook = excel_files[excel_file_names[j]]
		datasheet = workbook[membrane_type]
		gp_data = get_gp_row(datasheet,'Wave')
		data_dictionary[k] = gp_data
		k += 1
	gp_avg = [ np.mean(x) for x in zip(*data_dictionary.values() ) ]
	data_dictionary[k] = gp_avg
	averages_dictionary[membrane_type] = gp_avg
	k+= 1
	gp_std = [ np.std(x) for x in zip(*data_dictionary.values() ) ]
	data_dictionary[k] = gp_std
	stdv_dictionary[membrane_type] = gp_std
	plot_gp_data(temperatures,data_dictionary,len(excel_file_names),membrane_type)

# Creates GP plots with average+std.dev for lipid type
#curve_fit_point_to_exclude = int ( ( curve_fit_exclude / 2 ) * len(temperatures) )
#print (curve_fit_point_to_exclude)
#curve_fit_start = curve_fit_point_to_exclude
#curve_fit_end = len(temperatures) - curve_fit_point_to_exclude
for i in range(len(membrane_platforms)):
	popt,pconv = boltzmann_curve_fit(temperatures,averages_dictionary[membrane_platforms[i]],expected_melting_temp)
	x_fit = np.linspace(20,70,560)
	plt.plot(x_fit,boltzmann_curve_fit_func(x_fit,*popt),linestyle='-',color=color_main[i])
	plt.errorbar(temperatures,averages_dictionary[membrane_platforms[i]],yerr=stdv_dictionary[membrane_platforms[i]],color=color_main[i],marker='o',elinewidth=3,linestyle='None',label=membrane_platforms[i])
plt.xlabel('Temperature (C)')
plt.ylabel('Generalized Polarization')
plt.ylim(-0.51,0.51)
plt.axes().xaxis.set_minor_locator(AutoMinorLocator(5))
plt.axes().yaxis.set_minor_locator(AutoMinorLocator(2))
plt.legend(frameon=False)
figure_name = lipid_type + '_GenPol.' + str(filetype) 
plt.savefig(figure_name,format=filetype)
#plt.show()
plt.clf()

# Creates negative derivative plot to deterimine the melting temperature
print('Calculating first derivative of generalized polarization')
for i in range(len(membrane_platforms)):
	temp_deriv,gp_deriv = calc_first_derivative_central(temperatures,averages_dictionary[membrane_platforms[i]])
	gp_deriv = -1.0 * np.float_(gp_deriv)
	#print(temp_deriv,gp_deriv)
	# Generating curve of best fit
	#x = np.linspace(5,59,540)
	#dist_data = np.stack([temp_deriv,gp_deriv],axis=-1)
	#print(dist_data)
	#params = st.lognorm.fit(dist_data)
	#print(*params)
	# Gets peak of derivative plot
	max_index = np.argmax(gp_deriv)
	t_melt = temp_deriv[max_index]
	print(t_melt)
	plt.plot(temp_deriv,gp_deriv,marker='o',linestyle='None',color=color_main[i],label=membrane_platforms[i])
plt.xlabel('Temperature (C)')
plt.ylabel('-d(GP)/dT (C^-1)')
plt.axes().xaxis.set_minor_locator(AutoMinorLocator(5))
plt.legend(frameon=False)
#plt.show()
figure_name = lipid_type + '_dGPdT.' + str(filetype)
plt.savefig(figure_name,format=filetype)
plt.show()
plt.clf()

# Fits averages to Boltzmann sigmoid for analysis of Tm, cooperativity, and deltaH
