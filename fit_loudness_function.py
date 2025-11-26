import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from loudness_function import loudness_function
from scipy.optimize import fmin
from scipy.optimize import fmin_cg
from scipy.optimize import fmin_slsqp
from scipy.optimize import fmin_tnc
from loudness_function_bh2002 import loudness_function_bh2002
from fminsearchConstrained import fminsearchConstrained

'''% fit = fit_loudness_function(measured_data, fit_mode)
 version: 0.92
 This work is based on the publication
 "Optimized loudness-function estimation for categorical loudness scaling
 data." (2014) by Dirk Oetting, Thomas Brand and Stephan D. Ewert
 DOI: 10.1016/j.heares.2014.07.003

 ------------------------------------------------------------------------------
  Adaptive Categorical Loudness Scaling Procedure for AFC for Mathwork's MATLAB
 
  Author(s): Stephan Ewert, Dirk Oetting
 
  Copyright (c) 2013-2014, Stephan Ewert, Dirk Oetting
  All rights reserved.
 
  This work is licensed under the 
  Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License (CC BY-NC-ND 4.0). 
  To view a copy of this license, visit
  http://creativecommons.org/licenses/by-nc-nd/4.0/ or send a letter to Creative
  Commons, 444 Castro Street, Suite 900, Mountain View, California, 94041, USA.
 ------------------------------------------------------------------------------'''

def fit_loudness_function(measured_data, fit_mode='BTUX', optAlg='fmin',defaultUpperSlope=1.53):
    """
    This function creates a new fit structure from loudness scaling data points.
    A fit structure can be used to transform sound levels to categorical units and vice versa.

    PARAMETERS:
        measured_data:
            The data points from the categorical loudness scaling.
            [Level1, Response1, Level2, Response2, ...]
        fit_mode:
            BY: Brand and Hohmann (2001) fitting function
            BX: same as BY but minimize error in x-direction (level direction)
            BTX: BX + threshold estimation, minimize error in x-direction
            BTUX: BTX + UCL estimation, minimize error in x-direction (recommended for hearing-aid fitting)
            BTPX: threshold estimation, UCL estimation using data from Pascoe (1998)
            minimize error in x-direction
            BTUY: threshold estimation, UCL estimation, minimize error in y-direction

    OUTPUT:
        fit:
            Fit structure describing the loudness function.
            fit = [Lcut, m_low, m_high]
    """
    fit_mode = fit_mode.upper()
    optAlg = optAlg.upper()

    if fit_mode == 'BY':
        Lcut, m_low, m_high = bh_fit_2001(measured_data,optAlg)
        HTL = Lcut - 22.5 * 1 / m_low
        fit = [m_low, HTL, m_high]
    elif fit_mode == 'BX':
        m_low, HTL, m_high = bezier_fit(measured_data, 'x_v11',optAlg)
        fit = [m_low, HTL, m_high]
    elif fit_mode == 'BTX':
        m_low, HTL, m_high = bezier_fit(measured_data, 'x_v3',optAlg,defaultUpperSlope)
        fit = [m_low, HTL, m_high]
    elif fit_mode == 'BTUX':
        m_low, HTL, m_high = bezier_fit(measured_data, 'x_v36',optAlg,defaultUpperSlope)
        fit = [m_low, HTL, m_high]
    elif fit_mode == 'BTUY':
        m_low, HTL, m_high = bezier_fit(measured_data, 'y_v36',optAlg)
        fit = [m_low, HTL, m_high]
    elif fit_mode == 'BTPX':
        m_low, HTL, m_high = bezier_fit(measured_data, 'x_v37',optAlg)
        fit = [m_low, HTL, m_high]
    else:
        print(f'The fit mode "{fit_mode}" is not available.')
        return None

    # Convert values according to the function given in Brand and Hohmann (2001)
    m_lo = fit[0]
    HTL = fit[1]
    m_hi = fit[2]
    Lcut = HTL + 22.5 / m_lo
    fit = [Lcut, m_lo, m_hi]

    return fit


def bh_fit_2001(daten,optAlg):
    """
    Calculate [Lcut, Mlow, Mhigh] using the "Original" procedure.

    Parameters:
    - daten: Data array [level1, cu1, level2, cu2, ...]

    Returns:
    - Lcut, Mlow, Mhigh
    """
    # Start values for optimization
    fitparams_start = [75, 0.35, 0.65]

    # Use fmin to minimize the cost function
    # fitparams = fmin(lambda fitparams: costfc2(fitparams, daten), fitparams_start, disp=False) #Original cost function 
    if optAlg=='NEL':
        fitparams = fmin(lambda fitparams: costfc2(fitparams, daten), fitparams_start, disp=False,
                     xtol=1e-4, ftol=1e-4, maxiter=200*len(fitparams_start), maxfun=200*len(fitparams_start),full_output=False)
    if optAlg=='CG':
        fitparams=fmin_cg(lambda fitparams: costfc2(fitparams, daten), fitparams_start, fprime=None, gtol=1e-05, epsilon=1.4901161193847656e-08, maxiter=None, full_output=0, disp=1, retall=0, callback=None)

    if optAlg=='SLS':
        fitparams=fmin_slsqp(lambda fitparams: costfc2(fitparams, daten), fitparams_start, eqcons=(), f_eqcons=None, ieqcons=(), f_ieqcons=None, bounds=(), fprime=None, fprime_eqcons=None, fprime_ieqcons=None, iter=100, acc=1e-06, iprint=1, disp=None, full_output=0, epsilon=1.4901161193847656e-08, callback=None)

    if optAlg=='TNC':
        fitparams=fmin_tnc(lambda fitparams: costfc2(fitparams, daten), fitparams_start, fprime=None, approx_grad=0, bounds=None, epsilon=1e-08, scale=None, offset=None, messages=15, maxCGit=-1, maxfun=None, eta=-1, stepmx=0, accuracy=0, fmin=0, ftol=-1, xtol=-1, pgtol=-1, rescale=-1, disp=None, callback=None)

    if optAlg=='TRU':
        fitparams=minimize(lambda fitparams: costfc2(fitparams, daten), fitparams_start, method='trust-constr', jac=None, hess=None, hessp=None, bounds=None, tol=None, callback=None, options=None)
        fitparams=fitparams.x

    Lcut = fitparams[0]
    Mlow = fitparams[1]
    Mhigh = fitparams[2]

    return Lcut, Mlow, Mhigh


def bezier_fit(daten, mode,optAlg,defaultUpperSlope):
    # Check if 'x_v37' mode is passed and handle it
    pascoe_fit = False
    if mode == 'x_v37':
        pascoe_fit = True
        mode = 'x_v3'

    cu = np.array(daten[1::2])#.flatten()
    cu_all = cu.copy()
    levels_dB = np.array(daten[0::2])#.flatten()

    # Initialize randomized timer
    # np.random.seed(int(np.sum(100 * np.random.rand(1))))
    np.random.MT19937(0)
    idx = np.where(np.array(cu) <= 25)[0]

    # Check if cu and level values are not equal
    if any(np.diff(cu[idx]) > 0) and any(levels_dB[idx] > 0) and len(idx) > 5:
        a1 = np.polyfit(cu[idx], levels_dB[idx], 1)
        m1 = 1 / a1[0]
        b1 = -1 / a1[0] * a1[1]

        a2 = np.polyfit(levels_dB[idx], cu[idx], 1)
        m2 = a2[0]
        b2 = a2[1]

        m_low = np.mean([m1, m2])
        htl = np.mean([-b1 / m1, -b2 / m2])
    else:
        m_low = 1
        htl = np.mean(levels_dB[idx])

    mins = [0.2, -20, 0.2]
    maxs = [5, 100, 5]

    if mode == 'x_v11':
        # no weighting
        optFn = lambda fitparams, daten: costfcn_bezier_x(fitparams, daten, False, True)
    elif mode == 'y_v11':
        # no weighting
        optFn = lambda fitparams, daten: costfcn_bezier_y(fitparams, daten)
    elif mode[1:4] == '_v3':
        # find a good estimate for the hearing threshold
        optFn = lambda fitparams, daten: costfc_psychometric(fitparams, daten)
        searchOptions = {'disp': False}
        fitparams_start = [0.5, 0]
        mins_htl = [0.4, -30]
        maxs_htl = [1, 100]

        # use data points up to responses of soft
        htl_levels = levels_dB
        htl_probability = cu != 0
        range_of_htl = 0
        if np.sum(htl_probability == 0) == 0:
            # if no responses "not heard" are given, select 5 dB below the
            # lowest response as a "not heard" response
            fitparams = [0.4, np.min(htl_levels) - 5]
            estimated_htl = fitparams[1]
            mins[1] = estimated_htl - range_of_htl
            maxs[1] = estimated_htl + range_of_htl
        else:
            # daten_htl = np.concatenate([htl_levels, htl_probability]).reshape(-1, 1)
            # daten_htl = np.reshape([htl_levels, htl_probability]).reshape(-1, 1)
            daten_htl=np.array([htl_levels, htl_probability]).flatten(order='F')
            # result = minimize(lambda x: optFn(x, daten_htl), fitparams_start,
            #                   bounds=list(zip(mins_htl, maxs_htl)), options=searchOptions)
            result,feval = fminsearchConstrained(optFn,fitparams_start,mins_htl,maxs_htl,optAlg,searchOptions,daten_htl)
            # fitparams[ik, :] = result[0]
            fitparams= result
            # error_x[ik] = feval
            error_x = feval
            # fitparams = result.x
            estimated_htl = fitparams[1]
            if min(htl_levels) < estimated_htl < max(htl_levels):
                mins[1] = estimated_htl - range_of_htl
                maxs[1] = estimated_htl + range_of_htl
            else:
                print('Fitting method BTUX: HTL estimation failed for this dataset. Threshold estimation was not applied.')

        # remove data points with CU = 0
        idx_remove = np.where(np.array(cu) == 0)[0]
        cu = np.delete(cu, idx_remove)
        levels_dB = np.delete(levels_dB, idx_remove)
        # daten = np.concatenate([levels_dB, cu]).reshape(-1, 1)
        daten=np.array([levels_dB, cu]).flatten(order='F')

        if mode[1:] == '_v36':
            # set upper slope if less than 4 data points are available above 35
            # CU
            if np.sum(cu_all >= 35) < 4:
                # assume median slope
                # mins[2] = 1.53
                mins[2] =defaultUpperSlope #<---------------------- Modified here
                maxs[2] = mins[2]
                print(f'Fitting method BTUX: not enough data points in the upper loudness range. m_high was fixed to {mins[2]} CU/dB')

            # if mode[0] == 'x':
            #     optFn = lambda fitparams, daten: costfcn_bezier_x(fitparams, daten, True, True)
            # elif mode[0] == 'y':
            #     optFn = lambda fitparams, daten: costfcn_bezier_y(fitparams, daten)
        if mode[0] == 'x':
            optFn = lambda fitparams, daten: costfcn_bezier_x(fitparams, daten, True, True)
        elif mode[0] == 'y':
            optFn = lambda fitparams, daten: costfcn_bezier_y(fitparams, daten)

    number_of_runs = 10
    # error_x = np.full((1, number_of_runs), np.nan)
    error_x = np.empty((number_of_runs))
    error_x[:]=np.nan
    fitparams = np.empty((number_of_runs, len(mins)))
    fitparams_start = []
    fitparams_start = np.empty([3])
    # fitparams_start = np.empty((3))

    for ik in range(number_of_runs):
        # slope of loudness function
        # fitparams_start[0] = m_low + (np.random.rand(1) - 0.5) * 0.05
        fitparams_start[0] = m_low + (0.1  - 0.5) * 0.05
        # level of htl
        if mode[1:] == '_v11':
            # fitparams_start[1] = htl + (np.random.rand(1) - 0.5) * 5
            # fitparams_start[2] = mins[2] + (maxs[2] - mins[2]) * np.random.rand(1)
            fitparams_start[1] = htl + (0.1 - 0.5) * 5
            fitparams_start[2] = mins[2] + (maxs[2] - mins[2]) * 0.1
            
        elif mode[1:] in ('_v3', '_v36'):
            # fitparams_start[1] = mins[1] + (maxs[1] - mins[1]) * np.random.rand(1)
            fitparams_start[1] = mins[1] + (maxs[1] - mins[1]) * 0.1
            if len(fitparams_start)==2:
                # fitparams_start.append(mins[2] + (maxs[2] - mins[2]) * np.random.rand(1))
                fitparams_start.append(mins[2] + (maxs[2] - mins[2]) * 0.1)
            elif len(fitparams_start)==3:
                # fitparams_start[2] = mins[2] + (maxs[2] - mins[2]) * np.random.rand(1)
                fitparams_start[2] = mins[2] + (maxs[2] - mins[2]) * 0.1

        searchOptions = {'disp': False}
        # result = minimize(lambda x, optFn(x, daten), fitparams_start,
        #                   bounds=list(zip(mins, maxs)), options=searchOptions)
        result,feval = fminsearchConstrained(optFn,fitparams_start,mins,maxs,optAlg,searchOptions,daten)
        fitparams[ik, :] = result
        error_x[ik] = feval

    idx = np.argmin(error_x)
    fit_params = fitparams[idx, :]

    if pascoe_fit:
        # v_37
        re_fit = False
        MaxUCL = 140
        if loudness_function(50, fit_params, True) > MaxUCL:
            UCL = MaxUCL
            re_fit = True
        if fit_params[2] < 0.25 or np.sum(cu_all >= 35) < 4:
            UCL = PascoeUCL(fit_params[1])
            re_fit = True
        if re_fit:
            mins[2] = UCL
            maxs[2] = mins[2]
            max_Lcut = UCL - 5 / 25
            mins[0] = 22.5 / (max_Lcut - mins[1])

            # result = minimize(lambda x: optFn(x, daten), fitparams_start,
            #                   bounds=list(zip(mins, maxs)), options=searchOptions)
            result,feval = fminsearchConstrained(optFn,fitparams_start,mins,maxs,optAlg,searchOptions,daten)
            
            fit_params = result
            print('Fitting method BTUX: re-fit using UCL estimation of Pascoe (1988)')
            # convert from UCL to m_high
            b = 2.5 - fit_params[0] * fit_params[1]
            Lcut = (25 - b) / fit_params[0]
            fit_params[2] = 25 / (fit_params[2] - Lcut)
            # limit slope to 5 CU/dB
            fit_params[2] = min(fit_params[2], 5)

    m_low = fit_params[0]
    HTL = fit_params[1]
    m_high = fit_params[2]

    return m_low, HTL, m_high


def costfc2(fitparams, daten):
    """
    Cost function to minimize the sum of squared errors of the fit model to the data.
    
    Parameters:
    - fitparams: [Lcut, Mlow, Mhigh]
    - daten: [level1, cu1, level2, cu2, ...]

    Returns:
    - x: Sum of squared errors
    """
    level = np.array(daten[0::2])
    cu = np.array(daten[1::2])

    cu_fit = loudness_function_bh2002(level, fitparams)

    # Linear extrapolation of cu values
    x2 = loudness_function_bh2002(50, fitparams, True)  # get level at CU = 50
    x0 = loudness_function_bh2002(0, fitparams, True)  # get level at CU = 0

    cu_fit[level < x0] = fitparams[1] * (level[level < x0] - x0) + 0
    cu_fit[level > x2] = fitparams[2] * (level[level > x2] - x2) + 50

    # Set points where the measured data is 50 and the fit function is bigger
    # than 50 to the constant value 50, so that the difference cu - cu_fit = 0
    cu_fit[(cu == 50) & (cu_fit > cu)] = 50

    # Set points where the measured data is 0 and the fit function is lower
    # than 0 to the constant value 0, so that the difference cu - cu_fit = 0
    cu_fit[(cu == 0) & (cu_fit < cu)] = 0

    # Calculate the distance between cu and cu_fit
    delta_x = cu_fit - cu
    x = np.sum(delta_x**2)

    return x


def psychometric_function(fitparams, x):
    """
    Psychometric function.

    Parameters:
    - fitparams: [slope, threshold]
    - x: Input values

    Returns:
    - p: Psychometric function output
    """
    slope = fitparams[0]
    threshold = fitparams[1]
    p = 1 / (1 + np.exp(-slope * (x - threshold)))
    return p


def costfc_psychometric(fitparams, x):
    """
    Calculate the error of the psychometric function.

    Parameters:
    - fitparams: [slope, threshold]
    - x: Data containing levels and probabilities

    Returns:
    - x: Sum of squared errors
    """
    # daten = x[0]
    daten = x
    level = daten[0::2]
    probability = daten[1::2]

    psychometric_fit = psychometric_function(fitparams, level)
    x = np.sum((psychometric_fit - probability)**2)

    return x


def costfcn_bezier_y(fitparams, x):
    """
    Calculate the error of the Bezier function in the y-direction.

    Parameters:
    - fitparams: [Lcut, Mlow, Mhigh]
    - x: Data containing levels and cu values

    Returns:
    - x: Sum of squared errors
    """
    # daten = x[0]
    daten = x

    level = daten[0::2]
    cu = daten[1::2]

    cu_fit = loudness_function(level, fitparams)

    # Linear extrapolation of cu values
    x2 = loudness_function(50, fitparams, True)  # get level at CU = 50
    x0 = loudness_function(0, fitparams, True)   # get level at CU = 0

    cu_fit[level < x0] = fitparams[1] * (level[level < x0] - x0) + 0
    cu_fit[level > x2] = fitparams[2] * (level[level > x2] - x2) + 50

    # Set points where the measured data is 50 and the fit function is bigger
    # than 50 to the constant value 50, so that the difference cu - cu_fit = 0
    cu_fit[(cu == 50) & (cu_fit > cu)] = 50

    # Set points where the measured data is 0 and the fit function is lower
    # than 0 to the constant value 0, so that the difference cu - cu_fit = 0
    cu_fit[(cu == 0) & (cu_fit < cu)] = 0

    # Calculate the distance between cu and cu_fit
    delta_x = cu_fit - cu
    x = np.sum(delta_x**2)

    return x,


def costfcn_bezier_x(fitparams, x, weighting=False, limit_50=True, outliner_range=40):
    """
    Calculate the error of the broken stick function in the x-direction.

    Parameters:
    - fitparams: [Lcut, Mlow, Mhigh]
    - x: Data containing levels and cu values
    - weighting: Flag to apply weighting (default is False)
    - limit_50: Flag to limit contribution for CU=50 (default is True)
    - outliner_range: Range for limiting contribution (default is 40)

    Returns:
    - x: Sum of squared errors
    """
    if len(x) < 5:
        outliner_range = 40
    if len(x) < 4:
        limit_50 = True
    if len(x) < 3:
        weighting = False

    # daten = x[0]
    daten = x

    level = np.array(daten[0::2])
    cu = np.array(daten[1::2])

    level_fit = loudness_function(cu, fitparams, True)
    delta_x = level_fit - level

    cus = loudness_function([0, 50], fitparams, True)
    cu0_level = cus[0]
    UCL = cus[1]

    # set contribution from error function to zeros for levels which are
    # outside the fit-function range.
    if limit_50:
        idx_cu = np.where(cu == 50)[0]
        idx_level = np.where(level > UCL)[0]
        idx = np.intersect1d(idx_cu, idx_level)
        delta_x[idx] = 0

    # limit contribution of data point below loudness 0 CU as 0
    idx_cu = np.where(cu == 0)[0]
    idx_level = np.where(level < cu0_level)[0]
    idx = np.intersect1d(idx_cu, idx_level)
    delta_x[idx] = 0

    # limit contribution to error of 40 dB
    delta_x[np.where(np.abs(delta_x) > outliner_range)[0]] = outliner_range

    if weighting:
        cus = np.arange(0, 51, 5)
        std_kls = np.zeros(11)
        
        for m in range(11):
            idx = np.where(cu == cus[m])[0]
            if not any(idx):
                continue
            variante = 2
            if variante == 1:
                std_kls[m] = np.std(level[idx])
                if std_kls[m] < 3:
                    std_kls[m] = 3
            else:
                std_mean = np.array([np.nan, 8.5, 11.0, 11.8, 9.1, 6.9, 5.2, 4.3, 3.9, 3.4, 3.6])
                std_kls[m] = std_mean[m]

            delta_x[idx] = delta_x[idx] / std_kls[m]

    x = np.sum(delta_x**2)

    return x


def PascoeUCL(HTL, mode='smoothed'):
    """
    Estimate the Uncomfortable Loudness Level (UCL) from the Hearing Threshold Level (HTL).

    Parameters:
    - HTL: Hearing Threshold Level
    - mode: 'smoothed' or 'other' (default is 'smoothed')

    Returns:
    - UCL: Estimated Uncomfortable Loudness Level
    """
    if mode == 'smoothed':
        UCLverHTL = np.array([
            [-100, 100],
            [40, 100],
            [120, 140]
        ])
    UCL = np.interp(HTL, UCLverHTL[:, 0], UCLverHTL[:, 1])
    return UCL
