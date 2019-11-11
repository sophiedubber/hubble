import numpy as np
from astropy.io import fits
import photutils.psf as pt
from astropy.table import Table
import matplotlib.pyplot as plt
from photutils.psf import (IterativelySubtractedPSFPhotometry,BasicPSFPhotometry,FittableImageModel)
from photutils.detection import IRAFStarFinder
from photutils.psf import IntegratedGaussianPRF, DAOGroup
from photutils.background import MMMBackground, MADStdBackgroundRMS
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm,sigma_clipped_stats
from photutils import find_peaks
from astropy.nddata import NDData
from photutils.psf import extract_stars
from photutils import EPSFBuilder
from astropy.visualization import simple_norm


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# PLOTTING FUNCTIONS
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def plot_extracted_stars(stars):

	nrows = int(len(stars)/5)
	ncols = 5
	fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20),
	                       squeeze=True)
	ax = ax.ravel()
	for i in range(nrows*ncols):
	    norm = simple_norm(stars[i], 'log', percent=99.)
	    ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')

	plt.savefig('stars.pdf')
	#plt.show()
	plt.close()

	return

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def plot_epsf(epsf):

	norm = simple_norm(epsf.data, 'log', percent=99.)
	plt.figure()
	plt.imshow(epsf.data, norm=norm, origin='lowerleft', cmap='viridis')
	plt.colorbar()
	plt.show()

	return


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# SIMPLE GAUSSIAN PSF MODEL AND ITERATIVELY FITTING MODEL
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def simple_psf_options():

	psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
	
	psf_model.x_0.fixed = True
	psf_model.y_0.fixed = True
	pos = Table(names=['x_0', 'y_0'], data=[xcent,ycent])
	
	photometry = BasicPSFPhotometry(group_maker=daogroup,bkg_estimator=mmm_bkg,psf_model=psf_model,fitter=LevMarLSQFitter(),fitshape=(11,11))
	
	photometry = IterativelySubtractedPSFPhotometry(finder=iraffind,group_maker=daogroup,bkg_estimator=mmm_bkg,psf_model=psf_model,fitter=LevMarLSQFitter(),niters=1, fitshape=(11,11))
	result_tab = photometry(image=imdata)
	residual_image = photometry.get_residual_image()
	
	plt.figure()
	plt.imshow(residual_image,vmin=-0.2,vmax=2.0,cmap='plasma')
	plt.show()

	return


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# INITIAL SETUP FUNCTION
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def init_setup():

	fitimage = fits.open('Serpens3/idxq28010_drz.fits')
	imdata = fitimage[1].data
	head = fitimage[0].header
	
	bkgrms = MADStdBackgroundRMS()
	std = bkgrms(imdata)
	mean = np.mean(imdata)
	sigma_psf = 2.0
	iraffind = IRAFStarFinder(threshold=3.5*std,fwhm=sigma_psf*gaussian_sigma_to_fwhm,minsep_fwhm=0.01, roundhi=5.0, roundlo=-5.0,sharplo=0.0, sharphi=2.0)
	
	daogroup = DAOGroup(2.0*sigma_psf*gaussian_sigma_to_fwhm)
	mmm_bkg = MMMBackground()

	return imdata, bkgrms, std, sigma_psf, iraffind, daogroup, mmm_bkg, mean

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# EPSF FUNCTIONS
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def construct_epsf(imdata,mean):

	peaks_tbl = find_peaks(imdata,threshold=20.0)
	peaks_tbl['peak_value'].info.format = '%.8g'
	for i in range(len(peaks_tbl)):
		print(i,peaks_tbl['x_peak'][i],peaks_tbl['y_peak'][i])

	#rem_indA = [0,1,3,4,5,6,8,9,11,12,13,16,84,85,86,87,91,92,99,102]
	#rem_indB = [0,1,3,4,5,6,8,9,11,12,13,15,16,17,24,25,61,62,73,78,84,85,86,87,91,92,99,102]
	#rem_indC = [0,1,2,3,4,5,6,8,9,10,11,12,13,15,16,17,22,23,24,25,61,62,71,73,77,78,84,85,86,87,91,92,94,96,99,102]
	#rem_indVIS = [2,3,4,5,6,7,8,9,10,12,13,14]
	#rem_indD = [3,5]
	#rem_indVIS2 = [0,1,2,7]
	#rem_indE = [22,23,24,25,27,28,29]
	#rem_indF = [11,12,13,15,16,18,19,20,21,22,23,24,25,26]
	#rem_indG = [9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
	#rem_indH=[4,5,6,7,8,9,10,11,12,13]
	#rem_indI = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
	#rem_indJ = [0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,22,25,27,28,29,30,33,34,36,37,38,39]
	#rem_indK = [0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,23,24,25,26,29,30,31,33,34,35]
	#rem_indL = [4,5,6,7,8,10,11,12,13]
	#rem_indM = [5,6,7,8,9,10,11,13,14,15,16,18,27,29]
	#rem_indN = [4,5,6,7,8,9,10,11,13,14,15,16,17,19,20,21]
	#peaks_tbl.remove_rows([rem_indM])

	plt.figure()
	plt.scatter(peaks_tbl['x_peak'],peaks_tbl['y_peak'],c='k')
	plt.imshow(imdata,vmin=-0.2,vmax=2.,origin='lowerleft')
	plt.show()

	stars_tbl = Table()
	stars_tbl['x'] = peaks_tbl['x_peak']
	stars_tbl['y'] = peaks_tbl['y_peak']

	mean_val, median_val, std_val = sigma_clipped_stats(imdata,sigma=2.0)
	imdata -= median_val

	nddata = NDData(data=imdata)
	stars = extract_stars(nddata,stars_tbl,size=20)

	epsf_builder = EPSFBuilder(oversampling=4,maxiters=3,progress_bar=False)
	epsf,fitted_stars = epsf_builder(stars)

	hdu = fits.PrimaryHDU(epsf.data)
	hdul = fits.HDUList([hdu])
	hdul.writeto('Serpens3/epsf.fits')

	return stars, epsf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# epsf subtraction using epsf cosntructed above using photutils
def do_epsf_subtraction(imdata,epsf,iraffind,daogroup,mmm_bkg):

	photometry_epsf = IterativelySubtractedPSFPhotometry(finder=iraffind,group_maker=daogroup,bkg_estimator=mmm_bkg,psf_model=epsf,fitter=LevMarLSQFitter(),niters=3,fitshape=(11,11),aperture_radius=14)
	result_tab = photometry_epsf(image=imdata)
	residual_image = photometry_epsf.get_residual_image()
	
	plt.figure()
	plt.imshow(residual_image,vmin=-0.2,vmax=2.0,cmap='plasma',origin='lowerleft')
	plt.show()

	hdu = fits.PrimaryHDU(residual_image)
	hdul = fits.HDUList([hdu])
	#hdul.writeto('Serpens2/residual_fits/espf_residual_image.fits')

	return

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# epsf subtraction using epsf constructed using tiny tim
def do_epsf_subtraction2(imdata,iraffind,daogroup,mmm_bkg):

	epsf_file = fits.open('Serpens3/serpens3s400_psf.fits')
	epsf_fits = epsf_file[0].data

	epsf = FittableImageModel(epsf_fits)

	photometry_epsf = IterativelySubtractedPSFPhotometry(finder=iraffind,group_maker=daogroup,bkg_estimator=mmm_bkg,psf_model=epsf,fitter=LevMarLSQFitter(),niters=3,fitshape=(11,11),aperture_radius=14)
	result_tab = photometry_epsf(image=imdata)
	residual_image = photometry_epsf.get_residual_image()
	
	plt.figure()
	plt.imshow(residual_image,vmin=-0.2,vmax=2.0,cmap='plasma',origin='lowerleft')
	plt.show()	

	return 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# MAIN MAIN MAIN MAIN
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def main():

	imdata, bkgrms, std, sigma_psf, iraffind, daogroup, mmm_bkg, mean = init_setup()
	stars, epsf = construct_epsf(imdata,mean)
	#plot_extracted_stars(stars)
	#plot_epsf(epsf)
	#do_epsf_subtraction(imdata,epsf,iraffind,daogroup,mmm_bkg)
	#do_epsf_subtraction2(imdata,iraffind,daogroup,mmm_bkg)

	return

main()
