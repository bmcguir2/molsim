import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def plot_mf(spectrum,params={}):
	'''
	Plots a matched filter spectrum, with optional annotations and other parameters
	as defined in the parameters dictionary.  The keywords that can be specified,
	and any defaults, are as follows:	
	'''
	
	#load in options from the params dictionary, and any defaults	
	name = params['name'] if 'name'	in params else 'Matched Filter'
	figsize = params['figsize'] if 'figsize' in params else (9,6)
	fontsize = params['fontsize'] if 'fontsize' in params else 16
	xlabel = params['xlabel'] if 'xlabel' in params else 'Relative Velocity (km/s)'
	ylabel = params['ylabel'] if 'ylabel' in params else 'Impulse Response (' + r'$\sigma$' + ')'
	xlimits = params['xlimits'] if 'xlimits' in params else None
	ylimits = params['ylimits'] if 'ylimits' in params else None
	plot_color = params['plot_color'] if 'plot_color' in params else 'black'
	drawstyle = params['drawstyle'] if 'drawstyle' in params else 'steps'
	linewidth = params['linewidth'] if 'linewidth' in params else 1.0
	label = params['label'] if 'label' in params else None
	label_xy = params['label_xy'] if 'label_xy' in params else (0.95,0.75)
	label_color = params['label_color'] if 'label_color' in params else 'black'
	display_sigma = params['display_sigma'] if 'display_sigma' in params else True
	sigma_xy = params['sigma_xy'] if 'sigma_xy' in params else (0.95,0.90)
	sigma_color = params['sigma_color'] if 'sigma_color' in params else 'black'
	save_plot = params['save_plot'] if 'save_plot' in params else False
	file_out = params['file_out'] if 'file_out' in params else 'matched_filter.pdf'
	
	#plot shell making
	plt.ion()
	plt.close(name)
	fig = plt.figure(num=name,figsize=figsize)
	fontparams = {'size':fontsize, 'family':'sans-serif','sans-serif':['Helvetica']}	
	plt.rc('font',**fontparams)
	plt.rc('mathtext', fontset='stixsans')	
	ax = fig.add_subplot(111)
	
	#axis labels
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	
	#fix ticks
	ax.tick_params(axis='x', which='both', direction='in')
	ax.tick_params(axis='y', which='both', direction='in')
	ax.yaxis.set_ticks_position('both')
	ax.xaxis.set_ticks_position('both')	
	
	#plot
	plt.plot(spectrum.velocity,spectrum.snr,color=plot_color,drawstyle=drawstyle,linewidth=linewidth)
	
	#xlimits
	if xlimits is not None:
		ax.set_xlim(xlimits)
	
	#ylimits	
	snr = np.nanmax(spectrum.snr)	
	if ylimits is None:
		if snr < 10.:
			ax.set_ylim([-4,12])
		else:
			if 0.1*snr < 4:
				ymin = -4
			else:
				ymin = -0.1*snr
			ax.set_ylim([ymin,1.3*snr])
	else:
		ax.set_ylim(ylimits)		
	
	#annotate if desired
	if display_sigma is True:
		plt.annotate('Peak Impulse Response: {:.1f}' .format(snr) + r'$\sigma$', xy = sigma_xy, xycoords = 'axes fraction', color = sigma_color, ha = 'right')
	if label is not None:
		plt.annotate(label, xy = label_xy, xycoords = 'axes fraction', color = label_color, ha = 'right')
		
	#show it
	fig.canvas.draw()
	
	#save it if desired
	if save_plot is True:
		plt.savefig(file_out,format='pdf',transparent=True,bbox_inches='tight')
		
	return
	
def plot_stack(spectrum,params={}):
	'''
	Plots a stacked spectrum, with optional annotations and other parameters
	as defined in the parameters dictionary.  The keywords that can be specified,
	and any defaults, are as follows:	
	'''
	
	#load in options from the params dictionary, and any defaults	
	name = params['name'] if 'name'	in params else 'Stack'
	figsize = params['figsize'] if 'figsize' in params else (9,6)
	fontsize = params['fontsize'] if 'fontsize' in params else 16
	xlabel = params['xlabel'] if 'xlabel' in params else 'Relative Velocity (km/s)'
	ylabel = params['ylabel'] if 'ylabel' in params else 'Signal-to-Noise Ratio (' + r'$\sigma$' + ')'
	xlimits = params['xlimits'] if 'xlimits' in params else None
	ylimits = params['ylimits'] if 'ylimits' in params else None
	plot_color = params['plot_color'] if 'plot_color' in params else 'black'
	drawstyle = params['drawstyle'] if 'drawstyle' in params else 'steps'
	linewidth = params['linewidth'] if 'linewidth' in params else 1.0
	label = params['label'] if 'label' in params else None
	label_xy = params['label_xy'] if 'label_xy' in params else (0.95,0.75)
	label_color = params['label_color'] if 'label_color' in params else 'black'
	plot_sim = params['plot_sim'] if 'plot_sim' in params else True
	sim_color = params['sim_color'] if 'sim_color' in params else 'red'
	sim_linewidth = params['sim_linewidth'] if 'sim_linewidth' in params else 1.0
	sim_drawstyle = params['sim_drawstyle'] if 'sim_drawstyle' in params else 'steps'
	save_plot = params['save_plot'] if 'save_plot' in params else False
	file_out = params['file_out'] if 'file_out' in params else 'stacked_spectrum.pdf'

	#plot shell making
	plt.ion()
	plt.close(name)
	fig = plt.figure(num=name,figsize=figsize)
	fontparams = {'size':fontsize, 'family':'sans-serif','sans-serif':['Helvetica']}	
	plt.rc('font',**fontparams)
	plt.rc('mathtext', fontset='stixsans')	
	ax = fig.add_subplot(111)
	
	#axis labels
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	
	#fix ticks
	ax.tick_params(axis='x', which='both', direction='in')
	ax.tick_params(axis='y', which='both', direction='in')
	ax.yaxis.set_ticks_position('both')
	ax.xaxis.set_ticks_position('both')	
	
	#plot
	plt.plot(spectrum.velocity,spectrum.snr,color=plot_color,drawstyle=drawstyle,linewidth=linewidth)
	if plot_sim is True:
		plt.plot(spectrum.velocity,spectrum.int_sim,color=sim_color,drawstyle=sim_drawstyle,linewidth=sim_linewidth)
	
	#xlimits
	if xlimits is not None:
		ax.set_xlim(xlimits)
	
	#ylimits	
	snr = max([np.nanmax(spectrum.snr),np.nanmax(spectrum.int_sim)])
	if ylimits is None:
		if snr < 10.:
			ax.set_ylim([-4,12])
		else:
			if 0.1*snr < 4:
				ymin = -4
			else:
				ymin = -0.1*snr
			ax.set_ylim([ymin,1.3*snr])
	else:
		ax.set_ylim(ylimits)		
	
	#annotate if desired
	if label is not None:
		plt.annotate(label, xy = label_xy, xycoords = 'axes fraction', color = label_color, ha = 'right')
		
	#show it
	fig.canvas.draw()
	
	#save it if desired
	if save_plot is True:
		plt.savefig(file_out,format='pdf',transparent=True,bbox_inches='tight')

	return		