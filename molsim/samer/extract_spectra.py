#Functions besides extract_spectra() copied from toddTools as the module is not publicly available with the permission of Todd Hunter
#Does not contain any analysisUtils functions that are required as the module is publicly available
#To install analysisUtils see https://casaguides.nrao.edu/index.php/Analysis_Utilities

#Run in CASA with execfile() - doesn't recognize CASA commands with %run

from __future__ import print_function  # prevents adding old-style print statements

import glob as glob
import sys
# sys.path.insert(0,"/users/thunter/test/AIV/science/analysis_scripts/")
import analysisUtils as au
import numpy as np
import os
import json
import pathlib

import csv
import collections

try:
	from scipy.stats import nanstd
except:
	from numpy import nanstd
	
from analysisUtils import createCasaTool, numberOfChannelsInCube, findSpectralAxis, findPixel, getFitsBeam, pixelsPerBeam, iatool

#BEFORE RUNNING THIS SCRIPT
#ALL CUBES AND CONTINUUM FILES SHOULD BE ALIGNED IN PIXEL SPACE AND CONVOLVED TO THE SAME BEAM SIZE

###################################################
#EDIT THESE VARIABLES TO SUIT YOUR DATA
###################################################

#The directory that holds the final ALMA data cubes from which the spectra will be extracted
image_path='/lustre/cv/projects/cbrogan/From_Lustre/CoCCoA_HotCoreSurvey/PRODUCTS/G22.04+0.22/TuningB/CASA_Format/'

#List that holds all of the filepaths to the cubes as strings - set manually if desired
cubes = glob.glob(image_path+'*GHz*')

#Continuum
#Have to make sure that the files in cont_images are in the order you want for the continuum object (probably ascending frequency)
#Convert to Planck: tt.brightnessImage(cont_file,cont_file.split('/')[-1],rayleighJean=False,planck=True)

#The directory that holds the final continuum images from which the values will be extracted
cont_path = '/lustre/cv/users/selabd/spectra_extraction/g22.04_files/'
#List that holds all of the filepaths to the continuum images as strings - set manually if desired
cont_images = glob.glob(cont_path+'*.planck')

#Name of the source for output filenames - should match Fitting_Variables.source to simplify things
core = 'g22.04'

#output_dir will have the final spectra and contain a subdirectory with intermediate files
output_dir = '/lustre/cv/users/selabd/spectra_extraction/'+core+'_complete/'

#Define a box in pixel space for the region of interest
#Here defined starting with the pixel in the bottom left corner
ra_pixels = np.arange(360,413,1)
dec_pixels = np.arange(355,413,1)

# print(ra_pixels.size)
# print(dec_pixels.size)

###################################################
#SHOULD NOT HAVE TO EDIT ANYTHING BEYOND THIS POINT
###################################################

ARCSEC_PER_RAD = 206264.80624709636
AU = 1.49597e13 # cm
CM_PER_KILOPARSEC = ARCSEC_PER_RAD*AU*1000

def assertFile(filename = None):
	""" 
	Ensure a file or directory exists, else report and fail
	"""
	
	if filename is None: return
	if type(filename) == type([]):
		for f in filename:
			assert os.path.exists(f),  "tt.assertFile: %s does not exist" % f
	else:
		assert os.path.exists(filename),  "tt.assertFile: %s does not exist" % filename
	return

def computeStatisticalSpectrumFromMask(cube, jointmask, statistic='mean', suffix='', region=''):
    """
    Simpler version of the function in findContinuumCycle6.py (without the normalizeByMAD option).
    Uses ia.getprofile to compute the mean spectrum of a cube within a 
    masked area.
    jointmask: a 2D or 3D mask indicating which pixels shall be used
    statistic: passed to ia.getprofile via the 'function' parameter
    suffix: inequality suffix for jointmask, e.g. "<0.5"
    Returns: three lists: channels, freqs(Hz), intensities
    """
    chanInfo = numberOfChannelsInCube(cube, returnChannelWidth=True, returnFreqs=True) 
    nchan, firstFreq, lastFreq, channelWidth = chanInfo # freqs are in Hz
    frequency = np.linspace(firstFreq, lastFreq, nchan)  # lsrk

    myia = iatool()
    myia.open(cube)
    axis = findSpectralAxis(myia)
    if jointmask == '':
        jointmaskQuoted = jointmask
    else:
        print("Using jointmask = %s" % (jointmask))
        jointmaskQuoted = '"'+jointmask+'"'
    jointmaskQuoted += suffix
    print("cssfm: Running ia.getprofile(axis=%d, function='%s', mask='%s', region='%s', stretch=True)" % (axis,statistic, jointmaskQuoted, region))
    print(" on the cube: %s" % (cube))
    try:
        avgIntensity = myia.getprofile(axis=axis, function=statistic, mask=jointmaskQuoted, region=region, stretch=True)['values']
    except:
        avgIntensity = myia.getprofile(axis=axis, function=statistic, mask=jointmaskQuoted, region=region, stretch=True)['values']
    myia.close()
    channels = range(len(avgIntensity))
    if nchan != len(channels):
        print("Discrepant number of channels! %d vs %d" % (nchan,len(channels)))
    return channels, frequency, avgIntensity

def greybodyFluxDensity(parameters, frequency, Tbrightness=0):
	"""
	Given a dust model, and a list of frequencies, compute list of flux densities.
	parameters: {'Tdust': K, 'dust_angular_diameter': arcsec, 'beta':1.7, 
				 'reference_frequency':GHz, 'tau0': tau at reference_frequency}}
		 temperature: physical temperature
	frequency: list or array of values, in GHz
	Tbrightness: if specified, determine physical temperature and tau0 from 
				  this value instead of 'Tdust'
	Returns: 
	flux density in Jy
	"""
	temperature = parameters['Tdust']
	referenceFrequency = parameters['reference_frequency']
	beta = parameters['beta']
	if (Tbrightness > 0):
		tau0 = tauFromBrightness(Tbrightness, temperature)
	elif ('tau0' in parameters):
		tau0 = parameters['tau0']
	else:
		print("no tau0 in the parameters and Tbrightness <= 0: ", Tbrightness)
	solidAngle = beamSolidAngle(parameters['dust_angular_diameter'])
	tau = tau0 * pow(frequency/referenceFrequency, beta)
#    print "tau0 = %g, solidAngle=%g, tau=%s, Tb=%f, temp=%f" % (tau0, solidAngle, tau, Tbrightness, temperature)
	result = greybodyFluxDensityJy(temperature, tau, solidAngle, frequency)
	return(result)

def greybodyFluxDensityJy(temperature, tau, solidAngle, frequency):
    """
    temperature: in K
    tau: (can be a scalar or array)
    solidAngle: in sr
    frequency: in GHz (can be a scalar or array)
    Returns flux density in Jy
    """
    return(greybodyIntensity(temperature,tau,frequency) * solidAngle * 1e23)

def greybodyIntensity(temperature, tau, frequency):
    """
    temperature: in K
    tau: (can be a scalar or array)
    frequency: in GHz (can be a scalar or array)
    returns: in cgs
    """
    mytau = []
    if (type(tau) != list and type(tau) != np.ndarray):
        tau = [tau]
    for t in tau:
        mytau.append(t)
    tau = np.array(mytau, dtype=np.float64)
    result = (1-np.exp(-tau)) * planckIntensity(temperature,frequency)
    return(result)

def ispec(image, outfile=None, region='', sep=' ', format='%e',
		  overwrite=True, blc=None, trc=None, pol=0, 
		  showplot=False, source=None, velocity=False, startchan=None,
		  endchan=None, plotfile=None, plotrange=[0,0,0,0],
		  fontsize=11, labelCH3CN=False, xoffsetKms=0,label='',
		  labelCH313CN=False, fitCH3CN_K=-1, debug=False, title='',
		  drawstyle='steps-mid', grid=False, overlayFindcont='', linecolor='k',
		  findcontColor='r', writeHeader=True, frequnits='GHz', intensityUnits='Jy',
		  planck=False, mask='', minorTickInterval=None, includeChannelNumbers=False, 
		  restfreq=None, outpath='', radec='', overlayFindcontMode='frequency',
		  findcontLW=4):
	"""
	Emulates AIPS' ISPEC function to produce an ASCII spectrum from a
	CASA image, using the region tool (rg) and the image tool (ia).
	Operates on a single pixel and a single polarization only.
	* To operate on two cubes at once, see tt.ispecOverlay
	image: name of CASA image
	outfile: default is <image>+'blc[0].blc[1].ispec'
	outpath: only relevant if outfile is not set; combines os.path.basename of 
			 image and this outpath to define the outfile
	source: append this source name to the filename
	region: the region to use (alternative to blc,trc), e.g. 'circle[[100pix,100pix],10pix]'
			or a CRTF file, or 'whole' to use whole image
		 (an arbitrary CRTF shape can be used, as it is passed to ia.getprofile)
	sep: the column separator string to use in the output file
	blc, trc: the pixel range to use as a list (i.e. [40,50] or '40,50')
			  If trc is not specified, it is assumed to equal blc.
			  If neither are specified, nor radec, then use the peak.
	radec: sexagesimal RA Dec to use to find a single pixel to dump (using tt.findRADec)
	velocity: if True, then convert Hz to km/s via restfreq keyword
	restfreq: a value in Hz; if None, then use the restfreq keyword 
		in the image header
	startchan: the first spectral channel to consider
	endchan: the final spectral channel to consider (if negative, then ignore this many channels at end)
	showplot: make a graphics plot
	plotfile: save plot to this png name (True --> use automatic naming), implies showplot=True
	labelCH3CN: if True, label the J=12-11 lines
	xoffsetKms: value to shift the CH3CN lines
	drawstyle: 'steps' (stairsteps) or 'default' (connect the dots)
	grid: if True then draw dotted grid lines at major ticks
	overlayFindcont: draw the channel ranges used in the specified *_findContinuum.dat file
		 created by fc.findContinuum, or the file created by tt.findContinuumHistogram;
		 if True, then simply append '_findContinuum.dat' to the image name
	overlayFindcontMode: 'channel' or 'frequency' ('channel' will read from the
		   first line of the overlayFindcont file which contains LSRK channels, 
		   while 'frequency' will read
		   from the second line which contains LSRK frequencies)
	findcontColor: color of findcont ranges
	fintcontLW: linewidth of findcont ranges
	frequnits: output x-column units: 'GHz' or 'MHz'
	intensityUnits: 'Jy' (native from image) or 'K' (convert Jy to K using tt.brightness)
	   WARNING: it uses the conversion factor for the brightest pixel in the cube for
		 all channels, so the profiles are not strictly in Kelvin, but it is impossible
		 to create a Planck spectrum that shows negative noise values due to the non-
		 linearity of the Jy->K conversion (because Band 10 0.2" 1uJy=3K, 1nJy=2K, etc.)
	planck: if True, use 'brightnessTemperaturePlanck' in tt.brightness if intensityUnits=='k'
		   do not use if specifying img and the peak is a small number of Kelvin
	mask: mask to use in the ia.getregion call
	minorTickInterval: in frequency units, or in km/s (for velocity mode)
	includeChannelNumbers: if True, then third column of the output file will contain channel number
	Returns: two or three lists:
	  x-axis in GHz (or MHz or km/s), flux density (in image units), blc or regionXcenter, regionYCenter (i.e the pixel used)
	  if neither region nor blc are specified, then the third list is None
	Todd Hunter
	"""
	assertFile(image)
	if type(overlayFindcont) == bool:
		print("overlayFindcont must be a filename, not a boolean")
		return
	if overlayFindcont != '':
		assertFile(overlayFindcont)
		if overlayFindcontMode not in ['frequency', 'channel']:
			print("overlayFindcontMode must be either 'frequency' or 'channel'.")
			return
	if frequnits.lower()=='ghz':
		freqMultiplier = 1.0
	elif frequnits.lower()=='mhz':
		freqMultiplier = 1000.0
	else:
		print("Unrecognized frequnits (must be ghz or mhz, case insensitive)")
		return 
	shape = imhead(image,mode='get',hdkey='shape')
	if restfreq is None:
		restfreq = imhead(image,mode='get',hdkey='restfreq')['value']
	bunit = imhead(image,mode='get',hdkey='bunit') # normally this is 'Jy/beam'
	ctype3 = imhead(image,mode='get',hdkey='ctype3')
	naxes = len(shape)
	if intensityUnits.lower() == 'k':
		if planck:
			intensityMultiplier = brightness(img=image, verbose=False)['brightnessTemperaturePlanck'] / imagePeak(image)
		else:
			intensityMultiplier = brightness(img=image, verbose=False)['brightnessTemperature'] / imagePeak(image)
		bunit = 'K'
	elif intensityUnits.lower() == 'jy':
		intensityMultiplier = 1.0
		print("Set intensityMultiplier = ", intensityMultiplier)
		if bunit == '':
			bunit = 'Jy/beam'  # .residual images do not have a unit, so set it
	else:
		print("unrecognized intensity units (must be Jy or K, case insensitive)")
		return
	myrg = createCasaTool(rgtool)
	if type(ctype3) == dict:
		# CASA 4.0.1
		if ctype3['value'].upper().find('FREQ') >= 0:
			spectralAxis = 2
		else:
			ctype4 = imhead(image,mode='get',hdkey='ctype4')
			if ctype4['value'].upper().find('FREQ') >= 0:
				spectralAxis = 3
			else:
				print("Could not find frequency axis")
				return
	elif (ctype3.upper().find('FREQ') >= 0):
		spectralAxis = 2
	else:
		ctype4 = imhead(image,mode='get',hdkey='ctype4')
		if (ctype4.upper().find('FREQ') >= 0):
			spectralAxis = 3
		else:
			print("Could not find frequency axis")
			return
	if type(shape) == dict:
		shape = shape['value']
	nchan = shape[spectralAxis]
	if debug:
		print("spectralAxis = ", spectralAxis)
	if (endchan is not None):
		if (endchan <= 0):
			endchan = nchan+endchan-1
	if (region=='' and blc is None and mask == '' and radec == ''):
		print("Looking for peak pixel using imstat")
		if (startchan is not None):
			blc = imstat(image,chans='%d~%d'%(startchan,endchan))['maxpos']
		else:
			blc = imstat(image)['maxpos']
		print("Found peak at ", str(blc))
		blc = list(blc[:2])
		trc = blc[:]
		region = myrg.box(blc=blc,trc=trc)
	if (blc is not None):
		if (type(blc) == str):
			blc = [int(round(float(i))) for i in blc.split(',')]
		if (type(trc) == str):
			trc = [int(round(float(i))) for i in trc.split(',')]
		if (len(blc) == 2):
			# avoid appending onto the object that got passed to avoid screwing
			# up the user's script.
			newblc = blc[:]
			if spectralAxis == 3:
				newblc.append(pol)
				if (startchan is not None):
					newblc.append(startchan)
				else:
					newblc.append(0)
			else:
				if (startchan is not None):
					newblc.append(startchan)
				else:
					newblc.append(0)
				if naxes > 3:
					newblc.append(pol)
			blc = newblc
		else:
			print("blc must be a list of length 2: [x,y]")
			return
	elif (startchan is not None):
		if (naxes > 3):
			blc = [0,0,0,0]
			trc = [0,0,0,0]
		else:
			blc = [0,0,0]
			trc = [0,0,0]
		blc[spectralAxis] = startchan
		trc[spectralAxis] = endchan
		print("blc = ", blc)
		print("trc = ", trc)
	if (trc is not None):
		if (len(trc) == 2):
			newtrc = trc[:]
			if (spectralAxis==3):
				newtrc.append(pol)
				if (endchan is not None):
					newtrc.append(endchan)
				else:
					newtrc.append(nchan-1)
			else:
				if (endchan is not None):
					newtrc.append(endchan)
				else:
					newtrc.append(nchan-1)
				if naxes > 3:
					newtrc.append(pol)
			trc = newtrc
			region = myrg.box(blc=blc,trc=trc)
		else:
			print("type(trc)=%s, len(trc)=%d, trc=%s" % (type(trc),len(trc),str(trc)))
			print("trc must be a list of length 2:  [x,y]")
			return
	elif (region == '' and mask == '' and radec == ''):
		trc = blc[:]
		if endchan is not None:
			trc[spectralAxis] = endchan
		else:
			trc[spectralAxis] = nchan-1
		if debug:
			print("blc = ", blc)
			print("trc = ", trc)
		region = myrg.box(blc=blc,trc=trc)
#        Setting the type makes no difference because rg.box changes it to float
#        region = myrg.box(blc=np.array(blc,dtype=int),trc=np.array(trc,dtype=int))
#        region['trc'] = np.array(trc,dtype=int)
#        region['blc'] = np.array(blc,dtype=int)
#        region['inc'] = np.array(region['inc'],dtype=int)
#        print("region = ", region)
	elif (os.path.exists(region) or region=='whole'):
		if region == 'whole':
			print("Using whole image")
		else:
			print("Using whole image with region applied")
		regionUnderscored = os.path.basename(region)
	elif (region != ''):
		regionUnderscored = region.split('[[')[1].replace(']','').replace(',','_').replace('_[','_')
		if debug:
			print("regionUnderscored = ", regionUnderscored)
		regionXCenterIsRADec = False
		if (regionUnderscored.find('pix_')>0):
			# spatial part of region has been specified in pixels
			regionXCenter = regionUnderscored.split('_')[-3].split('pix')[0]
			if (regionXCenter.isdigit()):
				regionXCenter = float(regionXCenter)
		else:
			# spatial part of region has been specified in sexagesimal
			regionXCenter = regionUnderscored.split('_')[0]
			regionXCenterIsRADec = True
		if debug:
			print("regionXCenter = ", regionXCenter)
		if (regionUnderscored.find('pix_')>0):
			# spatial part of region has been specified in pixels
			regionYCenter = regionUnderscored.split('_')[-2].split('pix')[0]
			if (regionYCenter.isdigit()):
				regionYCenter = float(regionYCenter)
		else:
			# spatial part of region has been specified in sexagesimal
			regionYCenter = regionUnderscored.split('_')[1]
		if debug:
			print("regionXCenterIsRADec = ", regionXCenterIsRADec)
			print("regionYCenter = ", regionYCenter)
	elif (mask != ''):
		print("Using whole image with mask applied.")
	elif radec != '':
		blc = findRADec(image, radec)
		trc = blc
		print("Found pixel = ", blc)
		region = myrg.box(blc=blc,trc=trc)
	else:
		print("Using whole image.")
	if (outfile is None):
		if (not os.access(os.path.dirname(image), os.W_OK)):
			outfile = os.path.basename(image)
		else:
			outfile = image  
		if (blc is None):
			outfile += '.' + regionUnderscored
		else:
			outfile += '.%d.%d' % (blc[0],blc[1])
		if (source is not None):
			outfile += '.' + source
		outfile += '.ispec'
		if outpath != '':
			outfile = os.path.join(outpath, os.path.basename(outfile))
	myia = createCasaTool(iatool)
	myia.open(image)

#    print "Calling toASCII(outfile='%s',region='%s',mask='%s',sep='%s',format='%s',maskvalue=%d,overwrite=%s)" % (outfile,str(region),mask,sep,format,maskvalue,overwrite)
#    This only writes out the Jansky value, not the frequency.
#    myia.toASCII(outfile=outfile, region=region, mask=mask, sep=sep, 
#                 format=format, maskvalue=maskvalue,overwrite=overwrite)
	if debug:
		print("region = ", region)
	if region == 'whole':
		wholeImage = True
		region = ''
	else:
		wholeImage = False
	if len(mask) == 0:
#        Old method:
#        print("Calling ia.getregion(region='%s', mask='%s')" % (region,mask))
#        myshape = myia.shape()
#        if blc is not None:
#            if blc[0] >= myshape[0] or blc[1] >= myshape[1]:
#                print("Pixel (%d,%d) is outside of the image (%dx%d)." % (blc[0],blc[1],myshape[0],myshape[1]))
#                return
#        pixels = myia.getregion(region=region, mask=mask)
#        pixelmask = myia.getregion(region=region, mask=mask, getmask=True)
#        print("Done")
		print("Calling tt.computeStatisticalSpectrumFromMask(mask='%s', region='%s')" % (mask, region))
		channels, frequency, pixels = computeStatisticalSpectrumFromMask(image, mask, 'mean', region=region)
		spectralAxis = 0
		myshape = [len(channels)]
		freqHz = frequency
		print("Freq range: %f - %f" % (np.min(freqHz),np.max(freqHz)))
	else:
		print("Calling tt.ncomputeStatisticalSpectrumFromMask(mask='%s')" % (mask))
		channels, frequency, pixels = computeStatisticalSpectrumFromMask(image, mask, 'mean')
		spectralAxis = 0
		myshape = [len(channels)]
		freqHz = frequency
		print("Freq range: %f - %f" % (np.min(freqHz),np.max(freqHz)))
	if spectralAxis != 0:
		freqHz = []
		npixels = myshape[spectralAxis]
		if startchan is None:
			startchan = 0
		if endchan is None:
			endchan = npixels-1
		if debug:
			print("Got %d pixels" % (np.prod(np.shape(pixels))))
			print("nchan=%d  nchanSelected=%d" % (nchan, npixels))
		for p,pixel in enumerate(range(startchan,endchan+1)):
			mypixel = np.zeros(endchan-startchan+1)
			mypixel[spectralAxis] = pixel
			freqHz.append(myia.coordmeasures(pixel=mypixel)['measure']['spectral']['frequency']['m0']['value'])
		if debug:
			print("Built frequency array")
		freqHz = np.array(freqHz)
	myia.close()
	if (velocity):
		xaxis = -ckms*(freqHz-restfreq)/restfreq
	else:
		xaxis = np.array(freqHz)*1e-9
	f = open(outfile,'w')
	# Write the header information to the new output file
	if writeHeader:
		f.write('#title: Spectral profile - %s\n' % image)
		if (blc is None):
			if region == '' and mask != '':
				f.write('#region (mask): %s\n' % (mask))
			elif os.path.exists(region) or wholeImage:
				f.write('#region (region): %s\n' % (region))
			else:
				print("Calling findPixel('%s',pixel=[%s,%s],verbose=False)" % (image, str(regionXCenter), str(regionYCenter)))
				if regionXCenterIsRADec:
					result = regionXCenter+' '+regionYCenter
					f.write('#region (world): Point[[%s]]\n' % (result))
				else:
					result = findPixel(image,pixel=[regionXCenter,regionYCenter],verbose=False)
					f.write('#region (world): Point[[%s]]\n' % (result))
					f.write('#region (pixel): Point[[%.1f,%.1f]]\n' % (regionXCenter,regionYCenter))
		else:
			f.write('#region (world): Point[[%s]]\n' % (findPixel(image,pixel=blc,verbose=False)))
			f.write('#region (pixel): Point[[%.1f,%.1f]]\n' % (blc[0],blc[1]))
		if (velocity):
			f.write('#xLabel: velocity (km/s)\n')
		else:
			f.write('#xLabel: frequency (%s)\n'%frequnits)
		if intensityUnits.lower() == 'k':
			if planck:
				f.write('#yLabel: [K]  # Planck\n')
			else:
				f.write('#yLabel: [K]  # Rayleigh-Jean\n')
		else:
			f.write('#yLabel: [Jy/beam]\n')
		f.write('\n#%s\n'%(image))

	channelNumber = ''
	for i in range(len(freqHz)):
		if includeChannelNumbers:
			channelNumber = ' %d' % i
		if (len(np.shape(pixels)) == 4):
			idx = np.where(pixelmask == True)
			if spectralAxis == 3:
#                spectrum = pixels[0,0,0,i]*intensityMultiplier
				spectrum = np.mean(pixels[idx],axis=(0,1,2))*intensityMultiplier
			else:
#                f.write('%f %f%s\n' % (xaxis[i]*freqMultiplier, pixels[0,0,i,0]*intensityMultiplier, channelNumber))
				spectrum = np.mean(pixels[idx],axis=(0,1,3))*intensityMultiplier
			if debug:
				print("channel = %d (of %d)" % (i+1, len(freqHz)))
		elif (len(np.shape(pixels)) == 3):
#            f.write('%f %f%s\n' % (xaxis[i]*freqMultiplier, pixels[0,0,i]*intensityMultiplier, channelNumber))
			idx = np.where(pixelmask == True)
			spectrum = np.mean(pixels[idx],axis=(0,1))*intensityMultiplier
			if debug:
				print("channel = %d (of %d)" % (i+1, len(freqHz)))
		else:
			spectrum = pixels # ia.getprofile was used
		f.write('%f %f%s\n' % (xaxis[i]*freqMultiplier, spectrum[i], channelNumber))
	f.close()
	print("Wrote %s" % (outfile))
	if (showplot or plotfile is not None):
#        Could switch to the following once it works:
#        ispecPlot(outfile, velocity, plotfile, restfreq, plotrange)
		pl.clf()
		desc = pl.subplot(111)
		xaxis = np.array(xaxis)
		if debug:
			print("mean of pixels: %f, max: %f, shape=%s, type=%s" % (np.mean(pixels), np.max(pixels), np.shape(pixels), type(pixels)))
		if (len(np.shape(pixels)) == 4):
			if (spectralAxis == 2):
				if debug: print("A")
#                yaxis = pixels[0,0,:,0]*intensityMultiplier  # original code only worked with single pixel as blc=trc
				yaxis = np.mean(pixels,axis=(0,1,3))*intensityMultiplier
			else:
				if debug: 
					print("intensityMultiplier=%f" % (intensityMultiplier))
					print("B, spectralAxis=%d, shape(pixels[0,0,0])=%s, type=%s, pixels[0,0,0]=%s" % (spectralAxis,np.shape(pixels[0,0,0]),type(pixels[0,0,0]),pixels[0,0,0]))
#                yaxis = pixels[0,0,0]*intensityMultiplier # original code only worked with single pixel as blc=trc
				yaxis = np.mean(pixels,axis=(0,1,2))*intensityMultiplier
		elif (len(np.shape(pixels)) == 3):
			if debug: print("C")
#            yaxis = pixels[0,0]*intensityMultiplier# original code only worked with single pixel as blc=trc
			yaxis = np.mean(pixels,axis=(0,1))*intensityMultiplier
		else:
			yaxis = pixels*intensityMultiplier # ia.getprofile was used
		if debug:
			print("mean of yaxis: %f, max: %f" % (np.mean(yaxis), np.max(yaxis)))
		pl.plot(xaxis, yaxis, '-', color=linecolor, drawstyle=drawstyle)
		pl.ylabel('Intensity (%s)' % bunit)
		if (velocity):
			pl.xlabel('Velocity (km/s)')
		else:
			pl.xlabel('Frequency (%s)'%frequnits)
		desc.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
		if (plotrange[:2] != [0,0]):
			pl.xlim(plotrange[:2])
		else:
			pl.xlim([np.min(xaxis),np.max(xaxis)])
		if (plotrange[2:] != [0,0]):
			pl.ylim(plotrange[2:])
		if (blc is None):
			if os.path.exists(region) or wholeImage:
				if len(region) > 0:
					pl.title('region: %s' % (os.path.basename(region)))
				pl.text(0.5, 1.08, os.path.basename(image), ha='center', transform=desc.transAxes)
			elif len(mask) > 0:
				pl.title('mask: %s' % (os.path.basename(mask)))
				pl.text(0.5, 1.08, os.path.basename(image), ha='center', transform=desc.transAxes)
			elif (not regionXCenterIsRADec):
				radec = findPixel(image,pixel=[regionXCenter, regionYCenter])
				if (title == ''):
					title = os.path.basename(image)
				else:
					pl.text(0.5, 1.08, os.path.basename(image), ha='center', transform=desc.transAxes)
				pl.title(title,size=fontsize)
				pl.text(0.5, 0.96, '[%.1f,%.1f] = %s' % (regionXCenter,regionYCenter,radec), transform=desc.transAxes, ha='center')
			else:
				if (title == ''):
					title = os.path.basename(image) + '%s %s' % (regionXCenter, regionYCenter)                    
				pl.title(title ,size=fontsize)
		else:
			radec = findPixel(image,pixel=blc)
			if title == '':
				title = os.path.basename(image)
			else:
				pl.text(0.5, 1.08, os.path.basename(image), ha='center', transform=desc.transAxes)
			pl.title(title,size=fontsize)
			pl.text(0.5, 0.96, '[%.1f,%.1f] = %s' % (blc[0],blc[1],radec), transform=desc.transAxes, ha='center')
		if labelCH3CN:
			lines = np.array(sorted([220.23503, 220.32363, 220.40390, 220.47581,
									 220.53932, 220.59442, 220.64108, 220.67929,
									 220.70902, 220.73026, 220.74301, 220.74726],
									reverse=True))
			for line in lines:
				freq = line * (1 - xoffsetKms/ckms)
				pl.plot([freq,freq],pl.ylim(),'k:')
			if fitCH3CN_K >= 0:
				if (fitCH3CN_K+1 >= len(lines)):
					print("Cannot fit above K=",len(lines)-2)
					return
				minFreq = np.mean([lines[fitCH3CN_K],lines[fitCH3CN_K-1]])
				maxFreq = np.mean([lines[fitCH3CN_K],lines[fitCH3CN_K+1]])
				background = np.median(yaxis)
				amp = np.max(yaxis)
				shift = lines[fitCH3CN_K]
				width = lines[fitCH3CN_K]*3/ckms
				print("This line fitting does not yet work!")
				print("Calling onedgaussfit with %d points" % len(xaxis))
				print("background=%f, amp=%f, shift=%f, width=%f" % (background,amp,shift,width))
				results = onedgaussfit(xaxis, yaxis, params=[background,amp,shift,width,0],
							 limitedmin=[False,False,True,False,False],
							 limitedmax=[False,False,True,False,False],
							 minpars=[0,0,minFreq,0,0],
							 maxpars=[0,0,maxFreq,0,0])
				print("Fit results = ", results)
		if labelCH313CN:
			lines = np.array(sorted([220.12794, 220.21618, 220.29612, 220.36773,
									 220.43099, 220.48586, 220.53233, 220.57037,
									 220.59998, 220.62114, 220.63383, 220.63807],
									reverse=True))
			for line in lines:
				freq = line * (1 - xoffsetKms/ckms)
				pl.plot([freq,freq],pl.ylim(),'r:')
		if (label != ''):
			pl.text(0.5,0.97,label,transform=desc.transAxes,size=fontsize)
		if grid:
			desc.xaxis.grid(True,which='major')
			desc.yaxis.grid(True,which='major')
		if overlayFindcont != '':
			# First line format: 4~111;185~347;431~1794;1906~2042 pngname 0.865335
			# Second line format: 256.685871902GHz~256.686360197GHz LSRK ...
			f = open(overlayFindcont,'r')
			lines = f.readlines()
			f.close()
			yrange = np.max(yaxis)-np.min(yaxis)
			if overlayFindcontMode == 'channel':
				token = lines[0].split()
				channelRanges = token[0].split(';')
				findcontChannels = []
				print("channel ranges: ", channelRanges)
				for channelRange in channelRanges:
					c0,c1 = [int(i) for i in channelRange.split('~')]
					findcontChannels += range(c0,c1+1)
				for channelRange in channelRanges:
					c0,c1 = [int(i) for i in channelRange.split('~')]
					f0 = xaxis[c0]
					f1 = xaxis[c1]
					if False:
						for level in [level0+0.01*yrange, level0+0.1*yrange]:
							pl.plot([f0,f1], [level,level], '-',
									color=findcontColor, lw=findcontLW)
					else:
						level = np.median(yaxis[c0:c1+1])
						pl.plot([f0,f1], [level,level], '-', 
								color=findcontColor, lw=findcontLW)
				pl.plot(pl.xlim(), [level0,level0],'--',color='b',lw=1)
			elif overlayFindcontMode == 'frequency':
				freqLine = 1
				if lines[freqLine][0] == '#':  # allow one comment line
					freqLine += 1
				if lines[freqLine].find('LSRK') > 0:
					print("Parsing output from fc.findContinuum")
					freqRanges = lines[freqLine].strip().split(' LSRK')
				else:
					print("Parsing output from tt.findContinuumHistogram")
					freqRanges = lines[freqLine].strip().split()[1].split(';')  # format is 'msname.ms 220~221GHz;222~223GHz;' etc.
				findcontFreqRanges = []
				for freqRange in freqRanges:
					if len(freqRange) > 0:
						c0,c1 = [float(i.replace('GHz','')) for i in freqRange.split('~')]
						findcontFreqRanges.append([c0,c1])
				findcontChannels = []
				for findcontFreqRange in findcontFreqRanges:
					f0 = findcontFreqRange[0]
					f1 = findcontFreqRange[1]
					c0 = np.argmin(np.abs(xaxis-f0))
					c1 = np.argmin(np.abs(xaxis-f1))
					c0,c1 = sorted([c0,c1])
#                    print("c0,c1 = ", c0,c1)
					findcontChannels += range(c0,c1+1)
				for findcontFreqRange in findcontFreqRanges:
					f0 = findcontFreqRange[0]
					f1 = findcontFreqRange[1]
					c0 = np.argmin(np.abs(xaxis-f0))
					c1 = np.argmin(np.abs(xaxis-f1))
					c0,c1 = sorted([c0,c1])
					if False:
						level0 = np.median(yaxis[c0:c1+1])
						for level in [level0+0.01*yrange, level0+0.1*yrange]:
							pl.plot([f0,f1], [level,level], '-',
									color=findcontColor, lw=findcontLW)
					else:
						level = np.median(yaxis[c0:c1+1])
						pl.plot([f0,f1], [level,level], '-',
								color=findcontColor, lw=findcontLW)
				level0 = np.median(yaxis[findcontChannels])
				pl.text(0.05, 0.95, 'dashed line = %f' % (level0), ha='left', color='b', transform=desc.transAxes)
				pl.plot(pl.xlim(), [level0,level0],'--',color='b',lw=1)
		if velocity:
			if minorTickInterval is None:
				minorTickInterval = 1
		if minorTickInterval is not None:
			minorLocator = MultipleLocator(minorTickInterval)
			desc.xaxis.set_minor_locator(minorLocator)
		pl.draw()
		if (plotfile is not None):
			if plotfile == True:
				print("Using automatic plotfile naming")
				if label != '':
					label = label.replace(' ','')+'.'
				if blc is not None:
					plotfile = image + '.%d.%d.%sispec.png' % (blc[0],blc[1],label)
				elif len(mask) > 0:
					plotfile = image + '.%s.%sispec.png' % (os.path.basename(mask),label)
				else:
					plotfile = image + '.%sispec.png' % (label)
			pl.savefig(plotfile)
			print("Wrote %s" % (plotfile))
	# endif showplot
	myrg.done()
	if (len(np.shape(pixels)) == 4):
		if (spectralAxis == 3):
			intensity = pixels[0,0,0]
		else:
			intensity = pixels[0,0,:,0]
	elif (len(np.shape(pixels)) == 3):
		intensity = pixels[0,0]
	else:
		intensity = pixels
	if (blc is None and (region != '' or wholeImage)):
		if os.path.exists(region) or wholeImage:
			return(np.array(xaxis), intensity, None)
		else:
			return(np.array(xaxis), intensity, [regionXCenter,regionYCenter])
	elif mask == '':
		return(np.array(xaxis), intensity, blc[:2])
	else:
		return(np.array(xaxis), intensity, None)

def nanmedian(a, axis=0):
	"""
	Takes the median of an array, ignoring the nan entries
	"""
	if (np.__version__ < '1.81'):
		try:
			return(scipy.stats.nanmedian(a,axis)) 
		except:
			return(np.nanmedian(a,axis))
	else:
		return(np.nanmedian(a,axis))
	
def planckIntensity(temperature, frequency):
	"""
	Computes the Planck law per unit frequency interval
	temperature: K
	frequency: GHz (can be a scalar or array)
	returns: in cgs
	"""
	b = 2*h*pow(frequency*1e9,3) / pow(C,2) / np.expm1(h*frequency*1e9/(k*temperature))
	return(b)

def roundMeasurementToString(a, b, digits, totalDigits=None):
	"""
	a: value
	b: uncertainty (to be shown in parenthesis)
	digits: after the decimal
	totalDigits
	converts 134,20,2  to '134 (20)'
	"""
	if totalDigits is not None:
		mystring = '%*.*f (%.*f)' % (totalDigits, digits, float(a), digits, float(b))
	elif b != 0:
		mystring = '%.*f (%.*f)' % (digits, float(a), digits, float(b))
	else:
		mystring = '%.*f' % (digits, float(a))
	return mystring

def roundMeasurementToAsymmetricString(a, b, digits, totalDigits=None, latex=True, uncertainty=-1):
	"""
	a: main value
	b: a tuple with the negative and positive sigmas
	latex: if True, it converts 134,(20,21),2  to '134^{+20}_{-21})'
		   if False, it converts to '134+20-21'
	uncertainty: if zero, then do not show the uncertainty
	"""
	if (type(b) != list and type(b) != tuple and type(b) != np.ndarray):
		return roundMeasurementToString(a, b, digits)
	if latex:
		if totalDigits is not None:
			if uncertainty == 0:
				mystring = '%*.*f' % (totalDigits, digits, float(a))
			else:
				mystring = '%*.*f$^{%+.*f}_{-%.*f}$' % (totalDigits, digits, float(a), digits, float(b[1]), digits, abs(float(b[0])))
		elif uncertainty == 0:
			mystring = '%.*f' % (digits, float(a))
		else:
			mystring = '%.*f$^{%+.*f}_{-%.*f}$' % (digits, float(a), digits, float(b[1]), digits, abs(float(b[0])))
	else:
		if totalDigits is not None:
			mystring = '%*.*f %+.*f-%.*f' % (totalDigits, digits, float(a), digits, float(b[1]), digits, abs(float(b[0])))
		else:
			mystring = '%.*f %+.*f-%.*f' % (digits, float(a), digits, float(b[1]), digits, abs(float(b[0])))
	return mystring

def stefanBoltzmannLuminosity(radius, temperature, distance=None, verbose=False):
	"""
	radius: in arcsec, single value, or a list of 2 FWHM (diameters), or
	   a string with units ('au' or 'rsun')
	distance: in kpc; or if None, then interpret radius as cm (unless rsun 
	   in radius)
	temperature: in K
	Returns:
	luminosity in Lsun
	"""
	rsunSpecified = False
	if (type(radius) in [list,np.ndarray,tuple]):
		radius = 0.5*(radius[0]*radius[1])**0.5
	elif type(radius) == str:
		R = radius.lower()
		if R.find('au')>0:
			R = float(R.replace('au',''))*AU
		elif R.find('rsun')>0:
			rsunSpecified = True
			R = float(R.replace('rsun',''))*RSUN
		else:
			R = float(R.replace('au',''))*AU
		radius = R
		if verbose:
			print("Using R = %e cm" % (R))
	if distance is None or rsunSpecified:
		if verbose:
			print("Radius = %f AU = %f Rsun" % (radius/AU, radius/RSUN))
		R = radius
	else: # convert AU to cm
		R = radius*(distance*1000)*AU
	L = 4*stefan*np.pi*pow(R,2)*pow(temperature,4) / LSUN
	return(L)

def tauFromBrightness(Tbrightness, Tdust):
	"""
	Inverts the equation: Tbrightness = Tdust*(1-exp(-tau))
	and returns the opacity.  Works for single values or lists/arrays.
	"""
	return(-np.log(1-np.array(Tbrightness)/Tdust))

def extract_spectra():

	pathlib.Path(output_dir).mkdir(exist_ok=True)
	pathlib.Path(output_dir+'ispec_files/').mkdir(exist_ok=True)

	#Need one template image to pull values for the key
	test_image = cubes[0]

	#Create a file that maps the new pixel coordinates (X_Y) to the corresponding pixel in the original image
	pixel_file = output_dir+core+'_pixel_key.txt'
	ra_dec_pairs = dict()
	pixel_pairs = dict()
	with open(pixel_file, 'w') as file:
		for i in range(ra_pixels.size):
			for j in range(dec_pixels.size):
				ra_dec_pairs[str(i)+'_'+str(j)] = au.findPixel(test_image, pixel=[ra_pixels[i],dec_pixels[j]])
				pixel_pairs[str(i)+'_'+str(j)] = [ra_pixels[i],dec_pixels[j]]
				file.write(str(i)+'_'+str(j)+'\t'+str(ra_pixels[i])+','+str(dec_pixels[j])+'\n')

	#Create a file that matches the new pixel coordinates to the appropriate RA/Dec
	pos_file = output_dir+core+'_pos_key.txt'
	with open(pos_file, 'w') as file:
		for pos in ra_dec_pairs:
			file.write(pos+'\t'+ra_dec_pairs[pos]+'\n')

	#Make sure the ispec_files directory is clear before combining files
	os.system('for f in '+output_dir+'ispec_files/*.ispec; do rm "$f"; done')
	os.system('for f in '+output_dir+'ispec_files/*.txt; do rm "$f"; done')

	#Extract .ispec files for each of the spectral cubes
	pos_pixels = dict()
	for cube in cubes:
		for pos in pixel_pairs: 
			outfile = output_dir+'ispec_files/'+os.path.basename(cube)+'_'+pos+'.ispec'
			pixel = pixel_pairs[pos]
			ispec(image=cube,blc=pixel, showplot=False, planck=False, outfile=outfile)
			if pos not in pos_pixels:
				pos_pixels[pos] = dict()
			pos_pixels[pos][cube] = pixel

	#Combine the .ispec files for each pixel and convert the frequencies to MHz
	for pos in pixel_pairs:
		os.system(f'cat {output_dir}ispec_files/*_{pos}.ispec > {output_dir}ispec_files/{core}_spectra_{pos}_ghz.txt')
		ghz_file = output_dir+'ispec_files/'+core+'_spectra_'+pos+'_ghz.txt'
		lines = np.loadtxt(ghz_file)
		freq = []
		for i in range(np.shape(lines)[0]):
			freq.append(lines[i,0])
		freq = np.asarray(freq)
		freq *= 1e3
		for i in range(np.shape(lines)[0]):
			lines[i,0] = freq[i]
		np.savetxt(output_dir+core+'_spectra_'+pos+'.txt', lines)

	#Extract the continuum files and match to the correct pixel
	#Outputs a dictionary with the pixels and corresponding continuum value(s)
	cont_values = dict()
	cont_pixels = dict()
	for image in cont_images:
		for pos in pixel_pairs: 
			pixel = pixel_pairs[pos]
			stats=imstat(image,box=','.join([str(i) for i in pixel]*2))
			if pos not in cont_values:
				cont_values[pos] = list()
			if pos not in cont_pixels:
				cont_pixels[pos] = list()
			cont_values[pos].append(float(stats['max']))
			cont_pixels[pos].append(pixel)
	cont_file = output_dir+core+'_cont_values.txt'
	with open(cont_file, 'w') as file:
		json.dump(cont_values, file)

	#Record which pixels were used from each spectral cube / continuum image for the outputs of extract_spectra()
	pixel_key_outfile = output_dir+core+'_pixel_record.txt'
	with open(pixel_key_outfile, 'w') as file:
		for pos in pixel_pairs:
			file.write(pos+'\n')
			file.write('Spectra Extraction Pixels \n')
			for cube in cubes:
				file.write(cube+'\n')
				file.write(str(pos_pixels[pos][cube])+'\n')
			file.write('Continuum Pixels \n')
			file.write(str(cont_pixels[pos])+'\n')

	#os.system(f'tar cfvz {output_dir}{core}_spectra.tar.gz {output_dir}*{core}*.txt')

	print('Successfully extracted spectra from ' + str( len(ra_pixels)*len(dec_pixels) ) + ' pixels!')

	return
	
extract_spectra()

