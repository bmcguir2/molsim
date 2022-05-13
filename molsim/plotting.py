import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def plot_mf(spectrum,params={}):
	'''
	Plots a matched filter spectrum or spectra, with optional annotations and other parameters
	as defined in the parameters dictionary.  The keywords that can be specified,
	and their defaults, are as follows:	
	
	'name' 			: 	'Matched Filter', #string
	'figsize'		: 	(9,6), #tuple
	'fontsize'		: 	16, #integer
	'xlabel'		: 	'Relative Velocity (km/s)', #string
	'ylabel'		: 	'Impulse Response (' + r'$\sigma$' + ')', #string
	'xlimits'		: 	None, #list [lowerlimit,upperlimit]
	'ylimits'		:	None, #list [lowerlimit,upperlimit]
	'nxticks'		:	None, #integer number of xtick marks
	'nyticks'		:	None, #integer number of ytick marks
	'colors'		:	['black'], #list of matplotlib colors
	'drawstyles'	:	['steps'], #list of strings: 'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'
	'linewidths'	:	[1.0], #list of floats
	'alphas'		:	[1.0], #list of floats
	'orders'		:	[1], #list of integers
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
				'nxticks'		:	None,
				'nyticks'		:	None,
				'colors'		:	['black'],
				'drawstyles'	:	['steps'],
				'linewidths'	:	[1.0],
				'alphas'		:	[1.0], 
				'orders'		:	[1],
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
	
	if isinstance(spectrum,list) is False:
		spectrum = [spectrum]
		
	name = settings['name']			
	figsize	= settings['figsize']
	fontsize = settings['fontsize']	
	xlabel = settings['xlabel']		
	ylabel = settings['ylabel']		
	xlimits = settings['xlimits']	
	ylimits	= settings['ylimits']		
	nxticks = settings['nxticks']
	nyticks = settings['nyticks']
	colors = settings['colors']
	drawstyles = settings['drawstyles']	
	linewidths = settings['linewidths']
	alphas = settings['alphas']
	orders = settings['orders']
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
	
	if nxticks is not None:
		ax.xaxis.set_major_locator(plt.MaxNLocator(nxticks))
	if nyticks is not None:
		ax.yaxis.set_major_locator(plt.MaxNLocator(nyticks))
	
	#plot
	for spectrum,color,drawstyle,linewidth,alpha,zorder in zip(spectrum,colors,drawstyles,linewidths,alphas,orders):
		plt.plot(spectrum.velocity,spectrum.snr,color=color,drawstyle=drawstyle,linewidth=linewidth,alpha=alpha,zorder=zorder)
	
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
	'nxticks'		:	None, #integer number of xtick marks
	'nyticks'		:	None, #integer number of ytick marks	
	'plot_color'	:	'black', #matplotlib color
	'drawstyle'		:	'steps', #string: 'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'
	'linewidth'		:	1.0, #float
	'label'			:	None, #string
	'label_xy'		:	(0.95,0.75), #tuple, (vertical axis fraction, horizontal axis fraction)
	'label_color'	:	'black', #matplotlib color.
	'stack_alpha'	:	1.0, #float
	'plot_sim'		:	True, #True or False
	'sim_color'		:	'red', #matplotlib color
	'sim_linewidth'	:	1.0, #float
	'sim_drawstyle'	:	'steps', #string: 'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'
	'sim_alpha'		:	0.1, #float
	'save_plot'		:	False, #True or False
	'file_out'		:	'stacked_spectrum.pdf', #string	
	'save_format'	:	'pdf', #string: 'pdf', 'png', 'svg', 'jpg'
	'save_dpi'		:	300, #int
	'show_plot'		:	True, #True or False
	'''

	#load in options from the params dictionary, and any defaults
	settings = {'name' 			: 	'Stack',
				'figsize'		: 	(9,6),
				'fontsize'		: 	16,
				'xlabel'		: 	'Relative Velocity (km/s)',
				'ylabel'		: 	'Signal-to-Noise Ratio (' + r'$\sigma$' + ')',
				'xlimits'		: 	None,
				'ylimits'		:	None,
				'nxticks'		:	None,
				'nyticks'		:	None,				
				'plot_color'	:	'black',
				'drawstyle'		:	'steps',
				'linewidth'		:	1.0,
				'label'			:	None,
				'label_xy'		:	(0.95,0.75),
				'label_color'	:	'black',
				'stack_alpha'	:	1.0,
				'plot_sim'		:	True,
				'sim_color'		:	'red',
				'sim_linewidth'	:	1.0,
				'sim_drawstyle'	:	'steps',
				'sim_alpha'		:	1.0,
				'save_plot'		:	False,
				'file_out'		:	'stacked_spectrum.pdf',
				'save_format'	:	'pdf',
				'save_dpi'		:	300,
				'show_plot'		:	True,
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
	nxticks = settings['nxticks']
	nyticks = settings['nyticks']	
	plot_color = settings['plot_color']
	drawstyle = settings['drawstyle']	
	linewidth = settings['linewidth']
	label = settings['label']
	label_xy = settings['label_xy']
	label_color = settings['label_color']
	stack_alpha = settings['stack_alpha']
	plot_sim = settings['plot_sim']
	sim_color = settings['sim_color']
	sim_linewidth = settings['sim_linewidth']
	sim_drawstyle = settings['sim_drawstyle']
	sim_alpha = settings['sim_alpha']
	save_plot = settings['save_plot']
	file_out = settings['file_out']
	save_format = settings['save_format']
	save_dpi = settings['save_dpi']
	show_plot = settings['show_plot']

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
	
	if nxticks is not None:
		ax.xaxis.set_major_locator(plt.MaxNLocator(nxticks))
	if nyticks is not None:
		ax.yaxis.set_major_locator(plt.MaxNLocator(nyticks))	
	
	#plot
	plt.plot(spectrum.velocity,spectrum.snr,color=plot_color,drawstyle=drawstyle,linewidth=linewidth,alpha=stack_alpha)
	if plot_sim is True:
		plt.plot(spectrum.velocity,spectrum.int_sim,color=sim_color,drawstyle=sim_drawstyle,linewidth=sim_linewidth,alpha=sim_alpha)
	
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
	if show_plot is True:
		fig.canvas.draw()
	
	#save it if desired
	if save_plot is True:
		if save_format != 'pdf':
			plt.savefig(file_out,format=save_format,dpi=save_dpi,transparent=True,bbox_inches='tight')
		else:
			plt.savefig(file_out,format=save_format,transparent=True,bbox_inches='tight')

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
	'nxticks'		:	None, #integer number of xtick marks
	'nyticks'		:	None, #integer number of ytick marks		
	'sim_colors'	:	['red'], #list of matplotlib colors
	'sim_drawstyles':	['steps'], #list of drawstyles; string: 'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'
	'sim_linewidths':	[1.0], #list of floats
	'sim_orders'	:	[2], #list of integers
	'sim_alphas'	:	[1.0], #list of floats
	'obs_colors'	:	['black'], #list of matplotlib colors
	'obs_drawstyles':	['steps'], #list of drawstyles; string: 'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'
	'obs_linewidths':	[1.0], #list of floats
	'obs_orders'	:	[1], #list of integers
	'obs_alphas'	:	[1.0], #list of floats
	'plot_Iv'		:	False, #bool
	'plot_Tb'		:	True, #bool
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
				'nxticks'		:	None,
				'nyticks'		:	None,					
				'sim_colors'	:	['red'] * len(spectra),
				'sim_drawstyles':	['steps'] * len(spectra),
				'sim_linewidths':	[1.0] * len(spectra),
				'sim_orders'	:	[2] * len(spectra),
				'sim_alphas'	:	[1.0] * len(spectra),
				'obs'			:	None,
				'obs_colors'	:	['black'],
				'obs_drawstyles':	['steps'],
				'obs_linewidths':	[1.0],
				'obs_orders'	:	[1],			
				'obs_alphas'	:	[1.0],	
				'plot_Iv'		:	False,
				'plot_Tb'		:	True,
				'save_plot'		:	False,
				'file_out'		:	'simulated_spectrum.pdf'
				}
				
	for x in params:
		if x in settings:
			settings[x] = params[x]			
	
	#check if parameters conflict
	if settings['plot_Iv'] == settings['plot_Tb']:
		raise ValueError('Please confirm the spectral unit, plot_Iv and plot_Tb can not be both True or False.')
	
	#set appropriate default ylabel if not set by user
	if 'ylabel' not in params:
		if settings['plot_Iv']:
			settings['ylabel'] = r'$I_\nu$ (Jy/beam)'
		
	name = settings['name']			
	figsize	= settings['figsize']
	fontsize = settings['fontsize']	
	xlabel = settings['xlabel']		
	ylabel = settings['ylabel']		
	xlimits = settings['xlimits']	
	ylimits	= settings['ylimits']	
	nxticks = settings['nxticks']
	nyticks = settings['nyticks']		
	sim_colors = settings['sim_colors']
	sim_drawstyles = settings['sim_drawstyles']
	sim_linewidths = settings['sim_linewidths']
	sim_orders = settings['sim_orders']
	sim_alphas = settings['sim_alphas']
	obs = settings['obs']
	obs_colors = settings['obs_colors']
	obs_drawstyles = settings['obs_drawstyles']
	obs_linewidths = settings['obs_linewidths']
	obs_orders = settings['obs_orders']
	obs_alphas = settings['obs_alphas']
	plot_Iv = settings['plot_Iv']
	plot_Tb = settings['plot_Tb']
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
	
	if nxticks is not None:
		ax.xaxis.set_major_locator(plt.MaxNLocator(nxticks))
	if nyticks is not None:
		ax.yaxis.set_major_locator(plt.MaxNLocator(nyticks))		
	
	#plot
	for spectrum,color,drawstyle,linewidth,order,alpha in zip(spectra,sim_colors,sim_drawstyles,sim_linewidths,sim_orders,sim_alphas):
		plt.plot(spectrum.freq_profile,spectrum.int_profile,color=color,drawstyle=drawstyle,linewidth=linewidth,alpha=alpha,zorder=order)
	if obs is not None:
		for spectrum,color,drawstyle,linewidth,order,alpha in zip(obs,obs_colors,obs_drawstyles,obs_linewidths,obs_orders,obs_alphas):
			if plot_Tb:
				plt.plot(spectrum.frequency,spectrum.Tb,color=color,drawstyle=drawstyle,linewidth=linewidth,alpha=alpha,zorder=order)
			if plot_Iv:
				plt.plot(spectrum.frequency,spectrum.Iv,color=color,drawstyle=drawstyle,linewidth=linewidth,alpha=alpha,zorder=order)
	
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
	
def plot_highest_snr(sims,obs,params={}):

	'''
	Makes a grid plot of the highest snr lines from a simulation using parameters
	as defined in the parameters dictionary.  The keywords that can be specified,
	and their defaults, are as follows:

	'name' 			: 	'Simulation', #string
	'figsize'		: 	(9,6), #tuple
	'fontsize'		: 	16, #integer
	'xlabel'		: 	'Frequency (MHz)', #string
	'ylabel'		: 	r'T$_{b}$ (K)', #string
	'xlimits'		: 	None, #list [lowerlimit,upperlimit]
	'ylimits'		:	None, #list [lowerlimit,upperlimit]
	'nxticks'		:	None, #integer number of xtick marks
	'nyticks'		:	None, #integer number of ytick marks		
	'sim_colors'	:	['red'], #list of matplotlib colors
	'sim_drawstyles':	['steps'], #list of drawstyles; string: 'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'
	'sim_linewidths':	[1.0], #list of floats
	'sim_orders'	:	[2], #list of integers
	'sim_alphas'	:	[1.0], #list of floats
	'obs_colors'	:	['black'], #list of matplotlib colors
	'obs_drawstyles':	['steps'], #list of drawstyles; string: 'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'
	'obs_linewidths':	[1.0], #list of floats
	'obs_orders'	:	[1], #list of integers
	'obs_alphas'	:	[1.0], #list of floats
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
				'nxticks'		:	None,
				'nyticks'		:	None,					
				'sim_colors'	:	['red'] * len(spectra),
				'sim_drawstyles':	['steps'] * len(spectra),
				'sim_linewidths':	[1.0] * len(spectra),
				'sim_orders'	:	[2] * len(spectra),
				'sim_alphas'	:	[1.0] * len(spectra),
				'obs'			:	None,
				'obs_colors'	:	['black'],
				'obs_drawstyles':	['steps'],
				'obs_linewidths':	[1.0],
				'obs_orders'	:	[1],			
				'obs_alphas'	:	[1.0],	
				'save_plot'		:	False,
				'file_out'		:	'simulated_spectrum.pdf'
				}
				
	for x in params:
		if x in settings:
			settings[x] = params[x]			
		
	name = params['name'] if 'name' in params else 'Highest SNR Lines'
	figsize	= params['figsize'] if 'figsize' in params else ''
	fontsize = params['fontsize']	
	xlabel = params['xlabel']		
	ylabel = params['ylabel']		
	xlimits = params['xlimits']	
	ylimits	= params['ylimits']	
	nxticks = params['nxticks']
	nyticks = params['nyticks']		
	sim_colors = params['sim_colors']
	sim_drawstyles = params['sim_drawstyles']
	sim_linewidths = params['sim_linewidths']
	sim_orders = params['sim_orders']
	sim_alphas = params['sim_alphas']
	obs = params['obs']
	obs_colors = params['obs_colors']
	obs_drawstyles = params['obs_drawstyles']
	obs_linewidths = params['obs_linewidths']
	obs_orders = params['obs_orders']
	obs_alphas = params['obs_alphas']
	save_plot = params['save_plot']
	file_out = params['file_out']		

	return
	
# def interactive_plot(params = {}):
# 
# 
# 	plot_name = params['plot_name'] if params['plot_name'] else 'Interactive Plot'
# 	figsize = params['figsize'] if params['figsize'] else (9,6)
# 	xlabel = params['xlabel'] if params['xlabel'] else 'Freqeuncy (MHz)'
# 	ylabel = params['ylabel'] if params['ylabel'] else 'Intensity (Probably K)'
# 	nxticks = params['nxticks'] if params['nxticks'] else None
# 	nyticks = params['nyticks'] if params['nyticks'] else None
# 	xlimits = params['xlimits'] if params['xlimits'] else None
# 	ylimits = params['ylimits'] if params['ylimits'] else None
# 	save_plot = params['save_plot'] if params['save_plot'] else False
# 	file_out = params['file_out'] if params['file_out'] else None
# 	file_format = params['file_format'] if params['file_format'] else 'pdf'
# 	file_dpi = params['file_dpi'] if params['file_dpi'] else 300
# 	transparent = params['transparent'] if params['transparent'] else True
# 	
# 	
# 
# 	#plot shell making
# 	plt.ion()
# 	plt.close(plot_name)
# 	fig = plt.figure(num=plot_name,figsize=figsize)
# 	fontparams = {'size':fontsize, 'family':'sans-serif','sans-serif':['Helvetica']}	
# 	plt.rc('font',**fontparams)
# 	plt.rc('mathtext', fontset='stixsans')
# 	matplotlib.rcParams['pdf.fonttype'] = 42	
# 	ax = fig.add_subplot(111)
# 
# 	#axis labels
# 	plt.xlabel(xlabel)
# 	plt.ylabel(ylabel)
# 	
# 	#fix ticks
# 	ax.tick_params(axis='x', which='both', direction='in')
# 	ax.tick_params(axis='y', which='both', direction='in')
# 	ax.yaxis.set_ticks_position('both')
# 	ax.xaxis.set_ticks_position('both')	
# 	
# 	if nxticks is not None:
# 		ax.xaxis.set_major_locator(plt.MaxNLocator(nxticks))
# 	if nyticks is not None:
# 		ax.yaxis.set_major_locator(plt.MaxNLocator(nyticks))
# 		
# 	#xlimits
# 	ax.get_xaxis().get_major_formatter().set_scientific(False) #Don't let the x-axis go into scientific notation
# 	ax.get_xaxis().get_major_formatter().set_useOffset(False)	
# 	if xlimits is not None:
# 		ax.set_xlim(xlimits)
# 	
# 	#ylimits	
# 	if ylimits is not None:
# 		ax.set_ylim(ylimits)	
# 	
# 	
# 	
# 	for trace in traces:
# 	
# 			
# 
# 	#show it
# 	fig.canvas.draw()
# 	
# 	#save it if desired
# 	if save_plot is True:
# 		if file_format == 'pdf':
# 			plt.savefig(file_out,format=file_format,transparent=transparent,bbox_inches='tight')
# 		if file_format == 'png':
# 			plt.savefig(file_out,format=file_format,dpi=file_dpi,transparent=transparent,bbox_inches='tight')
# 			
# 			
# 
# 
# 	return
	
	