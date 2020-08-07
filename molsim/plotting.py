import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def plot_mf(spectrum,params={}):
	'''
	Plots a matched filter spectrum, with optional annotations and other parameters
	as defined in the parameters dictionary.  The keywords that can be specified,
	and their defaults, are as follows:	
	
	'name' 			: 	'Matched Filter', #string
	'figsize'		: 	(9,6), #tuple
	'fontsize'		: 	16, #integer
	'xlabel'		: 	'Relative Velocity (km/s)', #string
	'ylabel'		: 	'Impulse Response (' + r'$\sigma$' + ')', #string
	'xlimits'		: 	None, #list [lowerlimit,upperlimit]
	'ylimits'		:	None, #list [lowerlimit,upperlimit]
	'plot_color'	:	'black', #matplotlib color
	'drawstyle'		:	'steps', #string: 'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'
	'linewidth'		:	1.0, #float
	'label'			:	None, #string
	'label_xy'		:	(0.95,0.75), #tuple, (vertical axis fraction, horizontal axis fraction)
	'label_color'	:	'black', #matplotlib color
	'display_sigma'	:	True, #True or False
	'sigma_xy'		:	(0.95,0.90), #tuple, (vertical axis fraction, horizontal axis fraction)
	'sigma_color'	:	'black', #matplotlib color
	'save_plot'		:	False, #True or False
	'file_out'		:	'matched_filter.pdf', #string
	'''
	
	#load in options from the params dictionary, and any defaults
	settings = {'name' 			: 	'Matched Filter',
				'figsize'		: 	(9,6),
				'fontsize'		: 	16,
				'xlabel'		: 	'Relative Velocity (km/s)',
				'ylabel'		: 	'Impulse Response (' + r'$\sigma$' + ')',
				'xlimits'		: 	None,
				'ylimits'		:	None,
				'plot_color'	:	'black',
				'drawstyle'		:	'steps',
				'linewidth'		:	1.0,
				'label'			:	None,
				'label_xy'		:	(0.95,0.75),
				'label_color'	:	'black',
				'display_sigma'	:	True,
				'sigma_xy'		:	(0.95,0.90),
				'sigma_color'	:	'black',
				'save_plot'		:	False,
				'file_out'		:	'matched_filter.pdf',
				}
				
	for x in params:
		if x in settings:
			settings[x] = params[x]			
		
	name = settings['name']			
	figsize	= settings['figsize']
	fontsize = settings['fontsize']	
	xlabel = settings['xlabel']		
	ylabel = settings['ylabel']		
	xlimits = settings['xlimits']	
	ylimits	= settings['ylimits']		
	plot_color = settings['plot_color']
	drawstyle = settings['drawstyle']	
	linewidth = settings['linewidth']
	label = settings['label']
	label_xy = settings['label_xy']
	label_color = settings['label_color']
	display_sigma = settings['display_sigma']
	sigma_xy = settings['sigma_xy']
	sigma_color = settings['sigma_color']
	save_plot = settings['save_plot']
	file_out = settings['file_out']
	
	#plot shell making
	plt.ion()
	plt.close(name)
	fig = plt.figure(num=name,figsize=figsize)
	fontparams = {'size':fontsize, 'family':'sans-serif','sans-serif':['Helvetica']}	
	plt.rc('font',**fontparams)
	plt.rc('mathtext', fontset='stixsans')
	matplotlib.rcParams['pdf.fonttype'] = 42	
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
	and their defaults, are as follows:	
	
	'name' 			: 	'Stack', #string
	'figsize'		: 	(9,6), #tuple
	'fontsize'		: 	16, #integer
	'xlabel'		: 	'Relative Velocity (km/s)', #string
	'ylabel'		: 	'Signal-to-Noise Ratio (' + r'$\sigma$' + ')', #string
	'xlimits'		: 	None, #list [lowerlimit,upperlimit]
	'ylimits'		:	None, #list [lowerlimit,upperlimit]
	'plot_color'	:	'black', #matplotlib color
	'drawstyle'		:	'steps', #string: 'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'
	'linewidth'		:	1.0, #float
	'label'			:	None, #string
	'label_xy'		:	(0.95,0.75), #tuple, (vertical axis fraction, horizontal axis fraction)
	'label_color'	:	'black', #matplotlib color
	'plot_sim'		:	True, #True or False
	'sim_color'		:	'red', #matplotlib color
	'sim_linewidth'	:	1.0, #float
	'sim_drawstyle'	:	'steps', #string: 'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'
	'save_plot'		:	False, #True or False
	'file_out'		:	'stacked_spectrum.pdf', #string	
	'''

	#load in options from the params dictionary, and any defaults
	settings = {'name' 			: 	'Stack',
				'figsize'		: 	(9,6),
				'fontsize'		: 	16,
				'xlabel'		: 	'Relative Velocity (km/s)',
				'ylabel'		: 	'Signal-to-Noise Ratio (' + r'$\sigma$' + ')',
				'xlimits'		: 	None,
				'ylimits'		:	None,
				'plot_color'	:	'black',
				'drawstyle'		:	'steps',
				'linewidth'		:	1.0,
				'label'			:	None,
				'label_xy'		:	(0.95,0.75),
				'label_color'	:	'black',
				'plot_sim'		:	True,
				'sim_color'		:	'red',
				'sim_linewidth'	:	1.0,
				'sim_drawstyle'	:	'steps',
				'save_plot'		:	False,
				'file_out'		:	'stacked_spectrum.pdf'
				}
				
	for x in params:
		if x in settings:
			settings[x] = params[x]			
		
	name = settings['name']			
	figsize	= settings['figsize']
	fontsize = settings['fontsize']	
	xlabel = settings['xlabel']		
	ylabel = settings['ylabel']		
	xlimits = settings['xlimits']	
	ylimits	= settings['ylimits']		
	plot_color = settings['plot_color']
	drawstyle = settings['drawstyle']	
	linewidth = settings['linewidth']
	label = settings['label']
	label_xy = settings['label_xy']
	label_color = settings['label_color']
	plot_sim = settings['plot_sim']
	sim_color = settings['sim_color']
	sim_linewidth = settings['sim_linewidth']
	sim_drawstyle = settings['sim_drawstyle']
	save_plot = settings['save_plot']
	file_out = settings['file_out']

	#plot shell making
	plt.ion()
	plt.close(name)
	fig = plt.figure(num=name,figsize=figsize)
	fontparams = {'size':fontsize, 'family':'sans-serif','sans-serif':['Helvetica']}	
	plt.rc('font',**fontparams)
	plt.rc('mathtext', fontset='stixsans')	
	matplotlib.rcParams['pdf.fonttype'] = 42
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
		print('Test')
		plt.savefig(file_out,format='pdf',transparent=True,bbox_inches='tight')

	return		
	
def plot_sim(spectra,params={}):
	'''
	Plots a simulation, with optional annotations and other parameters
	as defined in the parameters dictionary.  The keywords that can be specified,
	and their defaults, are as follows:

	'name' 			: 	'Simulation', #string
	'figsize'		: 	(9,6), #tuple
	'fontsize'		: 	16, #integer
	'xlabel'		: 	'Frequency (MHz)', #string
	'ylabel'		: 	r'T$_{b}$ (K)', #string
	'xlimits'		: 	None, #list [lowerlimit,upperlimit]
	'ylimits'		:	None, #list [lowerlimit,upperlimit]
	'sim_colors'	:	['red'], #list of matplotlib colors
	'sim_drawstyles':	['steps'], #list of drawstyles; string: 'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'
	'sim_linewidths':	[1.0], #list of floats
	'sim_orders'	:	[2], #list of integers
	'obs_colors'	:	['black'], #list of matplotlib colors
	'obs_drawstyles':	['steps'], #list of drawstyles; string: 'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'
	'obs_linewidths':	[1.0], #list of floats
	'obs_orders'	:	[1], #list of strings
	'save_plot'		:	False, #True or False
	'file_out'		:	'simulated_spectrum.pdf' #string
		
	'''
	
	#load in options from the params dictionary, and any defaults
	settings = {'name' 			: 	'Simulation',
				'figsize'		: 	(9,6),
				'fontsize'		: 	16,
				'xlabel'		: 	'Frequency (MHz)',
				'ylabel'		: 	r'T$_{b}$ (K)',
				'xlimits'		: 	None,
				'ylimits'		:	None,
				'sim_colors'	:	['red'] * len(spectra),
				'sim_drawstyles':	['steps'] * len(spectra),
				'sim_linewidths':	[1.0] * len(spectra),
				'sim_orders'	:	[2] * len(spectra),
				'obs'			:	None,
				'obs_colors'	:	['black'],
				'obs_drawstyles':	['steps'],
				'obs_linewidths':	[1.0],
				'obs_orders'	:	[1],				
				'save_plot'		:	False,
				'file_out'		:	'simulated_spectrum.pdf'
				}
				
	for x in params:
		if x in settings:
			settings[x] = params[x]			
		
	name = settings['name']			
	figsize	= settings['figsize']
	fontsize = settings['fontsize']	
	xlabel = settings['xlabel']		
	ylabel = settings['ylabel']		
	xlimits = settings['xlimits']	
	ylimits	= settings['ylimits']		
	sim_colors = settings['sim_colors']
	sim_drawstyles = settings['sim_drawstyles']
	sim_linewidths = settings['sim_linewidths']
	sim_orders = settings['sim_orders']
	obs = settings['obs']
	obs_colors = settings['obs_colors']
	obs_drawstyles = settings['obs_drawstyles']
	obs_linewidths = settings['obs_linewidths']
	obs_orders = settings['obs_orders']
	save_plot = settings['save_plot']
	file_out = settings['file_out']	
	
	#plot shell making
	plt.ion()
	plt.close(name)
	fig = plt.figure(num=name,figsize=figsize)
	fontparams = {'size':fontsize, 'family':'sans-serif','sans-serif':['Helvetica']}	
	plt.rc('font',**fontparams)
	plt.rc('mathtext', fontset='stixsans')
	matplotlib.rcParams['pdf.fonttype'] = 42	
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
	for spectrum,color,drawstyle,linewidth,order in zip(spectra,sim_colors,sim_drawstyles,sim_linewidths,sim_orders):
		plt.plot(spectrum.freq_profile,spectrum.int_profile,color=color,drawstyle=drawstyle,linewidth=linewidth,zorder=order)
	if obs is not None:
		for spectrum,color,drawstyle,linewidth,order in zip(obs,obs_colors,obs_drawstyles,obs_linewidths,obs_orders):
			plt.plot(spectrum.frequency,spectrum.Tb,color=color,drawstyle=drawstyle,linewidth=linewidth,zorder=order)	
	
	#xlimits
	ax.get_xaxis().get_major_formatter().set_scientific(False) #Don't let the x-axis go into scientific notation
	ax.get_xaxis().get_major_formatter().set_useOffset(False)	
	if xlimits is not None:
		ax.set_xlim(xlimits)
	
	#ylimits	
	if ylimits is not None:
		ax.set_ylim(ylimits)		
			
	#show it
	fig.canvas.draw()
	
	#save it if desired
	if save_plot is True:
		plt.savefig(file_out,format='pdf',transparent=True,bbox_inches='tight')
		
	return	