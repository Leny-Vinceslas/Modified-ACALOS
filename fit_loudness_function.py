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

'''
fit = fit_loudness_function(measured_data, fit_mode)

This implementation fits categorical loudness scaling data using several
published variants (BY, BX, BTX, BTUX, BTUY, BTPX), with optional optimizer
selection and default upper-slope control for BTUX.

This work is based on the publication
"Optimized loudness-function estimation for categorical loudness scaling
data." (2014) by Dirk Oetting, Thomas Brand and Stephan D. Ewert
DOI: 10.1016/j.heares.2014.07.003

------------------------------------------------------------------------------
This is an experimental implementation of the above-mentioned publication.
It is not part of the original reference software and are not supported by the original authors.
------------------------------------------------------------------------------
'''

def fit_loudness_function(measured_data, fit_mode='BTUX', optAlg='fmin', defaultUpperSlope=1.53):
    fit_mode = fit_mode.upper()
    optAlg = optAlg.upper()

    if fit_mode == 'BY':
        Lcut, m_low, m_high = bh_fit_2001(measured_data, optAlg)
        HTL = Lcut - 22.5 * 1 / m_low
        fit = [m_low, HTL, m_high]
    elif fit_mode == 'BX':
        m_low, HTL, m_high = bezier_fit(measured_data, 'x_v11', optAlg, defaultUpperSlope)
        fit = [m_low, HTL, m_high]
    elif fit_mode == 'BTX':
        m_low, HTL, m_high = bezier_fit(measured_data, 'x_v3', optAlg, defaultUpperSlope)
        fit = [m_low, HTL, m_high]
    elif fit_mode == 'BTUX':
        m_low, HTL, m_high = bezier_fit(measured_data, 'x_v36', optAlg, defaultUpperSlope)
        fit = [m_low, HTL, m_high]
    elif fit_mode == 'BTUY':
        m_low, HTL, m_high = bezier_fit(measured_data, 'y_v36', optAlg, defaultUpperSlope)
        fit = [m_low, HTL, m_high]
    elif fit_mode == 'BTPX':
        m_low, HTL, m_high = bezier_fit(measured_data, 'x_v37', optAlg, defaultUpperSlope)
        fit = [m_low, HTL, m_high]
    else:
        print(f'The fit mode "{fit_mode}" is not available.')
        return None

    m_lo = fit[0]
    HTL = fit[1]
    m_hi = fit[2]
    Lcut = HTL + 22.5 / m_lo
    fit = [Lcut, m_lo, m_hi]

    return fit


def bh_fit_2001(daten, optAlg):
    fitparams_start = [75, 0.35, 0.65]

    if optAlg == 'NEL':
        fitparams = fmin(lambda p: costfc2(p, daten), fitparams_start, disp=False,
                         xtol=1e-4, ftol=1e-4, maxiter=200 * len(fitparams_start),
                         maxfun=200 * len(fitparams_start), full_output=False)
    elif optAlg == 'CG':
        fitparams = fmin_cg(lambda p: costfc2(p, daten), fitparams_start, fprime=None,
                            gtol=1e-05, epsilon=1.4901161193847656e-08, maxiter=None,
                            full_output=0, disp=1, retall=0, callback=None)
    elif optAlg == 'SLS':
        fitparams = fmin_slsqp(lambda p: costfc2(p, daten), fitparams_start, eqcons=(), f_eqcons=None,
                               ieqcons=(), f_ieqcons=None, bounds=(), fprime=None, fprime_eqcons=None,
                               fprime_ieqcons=None, iter=100, acc=1e-06, iprint=1, disp=None, full_output=0,
                               epsilon=1.4901161193847656e-08, callback=None)
    elif optAlg == 'TNC':
        fitparams = fmin_tnc(lambda p: costfc2(p, daten), fitparams_start, fprime=None, approx_grad=0,
                             bounds=None, epsilon=1e-08, scale=None, offset=None, messages=15,
                             maxCGit=-1, maxfun=None, eta=-1, stepmx=0, accuracy=0, fmin=0, ftol=-1,
                             xtol=-1, pgtol=-1, rescale=-1, disp=None, callback=None)
    elif optAlg == 'TRU':
        result = minimize(lambda p: costfc2(p, daten), fitparams_start, method='trust-constr')
        fitparams = result.x
    else:
        fitparams = fmin(lambda p: costfc2(p, daten), fitparams_start, disp=False)

    Lcut, Mlow, Mhigh = fitparams[0], fitparams[1], fitparams[2]
    return Lcut, Mlow, Mhigh


def bezier_fit(daten, mode, optAlg, defaultUpperSlope):
    pascoe_fit = False
    if mode == 'x_v37':
        pascoe_fit = True
        mode = 'x_v3'

    cu = np.array(daten[1::2])
    cu_all = cu.copy()
    levels_dB = np.array(daten[0::2])

    np.random.MT19937(0)
    idx = np.where(np.array(cu) <= 25)[0]

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
        optFn = lambda p, d: costfcn_bezier_x(p, d, False, True)
    elif mode == 'y_v11':
        optFn = lambda p, d: costfcn_bezier_y(p, d)
    elif mode[1:4] == '_v3':
        optFn = lambda p, d: costfc_psychometric(p, d)
        searchOptions = {'disp': False}
        fitparams_start = [0.5, 0]
        mins_htl = [0.4, -30]
        maxs_htl = [1, 100]

        htl_levels = levels_dB
        htl_probability = cu != 0
        range_of_htl = 0
        if np.sum(htl_probability == 0) == 0:
            fitparams = [0.4, np.min(htl_levels) - 5]
            estimated_htl = fitparams[1]
            mins[1] = estimated_htl - range_of_htl
            maxs[1] = estimated_htl + range_of_htl
        else:
            daten_htl = np.array([htl_levels, htl_probability]).flatten(order='F')
            result, feval = fminsearchConstrained(optFn, fitparams_start, mins_htl, maxs_htl, optAlg, searchOptions, daten_htl)
            fitparams = result
            estimated_htl = fitparams[1]
            if min(htl_levels) < estimated_htl < max(htl_levels):
                mins[1] = estimated_htl - range_of_htl
                maxs[1] = estimated_htl + range_of_htl
            else:
                print('Fitting method BTUX: HTL estimation failed for this dataset. Threshold estimation was not applied.')

        idx_remove = np.where(np.array(cu) == 0)[0]
        cu = np.delete(cu, idx_remove)
        levels_dB = np.delete(levels_dB, idx_remove)
        daten = np.array([levels_dB, cu]).flatten(order='F')

        if mode[1:] == '_v36':
            if np.sum(cu_all >= 35) < 4:
                mins[2] = defaultUpperSlope
                maxs[2] = mins[2]
                print(f'Fitting method BTUX: not enough data points in the upper loudness range. m_high was fixed to {mins[2]} CU/dB')

        if mode[0] == 'x':
            optFn = lambda p, d: costfcn_bezier_x(p, d, True, True)
        elif mode[0] == 'y':
            optFn = lambda p, d: costfcn_bezier_y(p, d)
    else:
        daten = np.array([levels_dB, cu]).flatten(order='F')

    number_of_runs = 10
    error_x = np.empty((number_of_runs))
    error_x[:] = np.nan
    fitparams = np.empty((number_of_runs, len(mins)))
    fitparams_start = np.empty([3])

    for ik in range(number_of_runs):
        fitparams_start[0] = m_low + (0.1 - 0.5) * 0.05
        if mode[1:] == '_v11':
            fitparams_start[1] = htl + (0.1 - 0.5) * 5
            fitparams_start[2] = mins[2] + (maxs[2] - mins[2]) * 0.1
        elif mode[1:] in ('_v3', '_v36'):
            fitparams_start[1] = mins[1] + (maxs[1] - mins[1]) * 0.1
            if len(fitparams_start) == 2:
                fitparams_start.append(mins[2] + (maxs[2] - mins[2]) * 0.1)
            elif len(fitparams_start) == 3:
                fitparams_start[2] = mins[2] + (maxs[2] - mins[2]) * 0.1

        searchOptions = {'disp': False}
        result, feval = fminsearchConstrained(optFn, fitparams_start, mins, maxs, optAlg, searchOptions, daten)
        fitparams[ik, :] = result
        error_x[ik] = feval

    idx_best = np.argmin(error_x)
    fit_params = fitparams[idx_best, :]

    if pascoe_fit:
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
            result, feval = fminsearchConstrained(optFn, fitparams_start, mins, maxs, optAlg, searchOptions, daten)
            fit_params = result
            print('Fitting method BTUX: re-fit using UCL estimation of Pascoe (1988)')
            b = 2.5 - fit_params[0] * fit_params[1]
            Lcut = (25 - b) / fit_params[0]
            fit_params[2] = 25 / (fit_params[2] - Lcut)
            fit_params[2] = min(fit_params[2], 5)

    m_low = fit_params[0]
    HTL = fit_params[1]
    m_high = fit_params[2]

    return m_low, HTL, m_high


def costfc2(fitparams, daten):
    level = np.array(daten[0::2])
    cu = np.array(daten[1::2])

    cu_fit = loudness_function_bh2002(level, fitparams)

    x2 = loudness_function_bh2002(50, fitparams, True)
    x0 = loudness_function_bh2002(0, fitparams, True)

    cu_fit[level < x0] = fitparams[1] * (level[level < x0] - x0)
    cu_fit[level > x2] = fitparams[2] * (level[level > x2] - x2) + 50

    cu_fit[(cu == 50) & (cu_fit > cu)] = 50
    cu_fit[(cu == 0) & (cu_fit < cu)] = 0

    delta_x = cu_fit - cu
    return np.sum(delta_x ** 2)


def psychometric_function(fitparams, x):
    slope = fitparams[0]
    threshold = fitparams[1]
    return 1 / (1 + np.exp(-slope * (x - threshold)))


def costfc_psychometric(fitparams, x):
    daten = x
    level = daten[0::2]
    probability = daten[1::2]
    psychometric_fit = psychometric_function(fitparams, level)
    return np.sum((psychometric_fit - probability) ** 2)


def costfcn_bezier_y(fitparams, x):
    daten = x
    level = daten[0::2]
    cu = daten[1::2]

    cu_fit = loudness_function(level, fitparams)

    x2 = loudness_function(50, fitparams, True)
    x0 = loudness_function(0, fitparams, True)

    cu_fit[level < x0] = fitparams[1] * (level[level < x0] - x0)
    cu_fit[level > x2] = fitparams[2] * (level[level > x2] - x2) + 50

    cu_fit[(cu == 50) & (cu_fit > cu)] = 50
    cu_fit[(cu == 0) & (cu_fit < cu)] = 0

    delta_x = cu_fit - cu
    return np.sum(delta_x ** 2),


def costfcn_bezier_x(fitparams, x, weighting=False, limit_50=True, outliner_range=40):
    if len(x) < 5:
        outliner_range = 40
    if len(x) < 4:
        limit_50 = True
    if len(x) < 3:
        weighting = False

    daten = x

    level = np.array(daten[0::2])
    cu = np.array(daten[1::2])

    level_fit = loudness_function(cu, fitparams, True)
    delta_x = level_fit - level

    cus = loudness_function([0, 50], fitparams, True)
    cu0_level = cus[0]
    UCL = cus[1]

    if limit_50:
        idx_cu = np.where(cu == 50)[0]
        idx_level = np.where(level > UCL)[0]
        idx = np.intersect1d(idx_cu, idx_level)
        delta_x[idx] = 0

    idx_cu = np.where(cu == 0)[0]
    idx_level = np.where(level < cu0_level)[0]
    idx = np.intersect1d(idx_cu, idx_level)
    delta_x[idx] = 0

    delta_x[np.where(np.abs(delta_x) > outliner_range)[0]] = outliner_range

    if weighting:
        cus = np.arange(0, 51, 5)
        std_kls = np.zeros(11)
        for m in range(11):
            idx = np.where(cu == cus[m])[0]
            if not any(idx):
                continue
            std_mean = np.array([np.nan, 8.5, 11.0, 11.8, 9.1, 6.9, 5.2, 4.3, 3.9, 3.4, 3.6])
            std_kls[m] = std_mean[m]
            delta_x[idx] = delta_x[idx] / std_kls[m]

    return np.sum(delta_x ** 2)


def PascoeUCL(HTL, mode='smoothed'):
    if mode == 'smoothed':
        UCLverHTL = np.array([[-100, 100], [40, 100], [120, 140]])
    return np.interp(HTL, UCLverHTL[:, 0], UCLverHTL[:, 1])
