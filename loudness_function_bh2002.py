import numpy as np

def loudness_function_bh2002(x, fitparams, inverse=False):
    '''
    fit = loudness_function_bh2002(x, fitparams, inverse)

    function calculates the loudness function according to fitparams

    PARAMETERS:
        x:
        either levels to calculate CU
        if inverse = true, x contains CU and calculates levels
        fitparams:
        fitparams =[lcut, mlow, mhigh]
        mlow and mhigh have to be positive values
        if negative values are given, the absolute value will be used.
        the smallest value for mlow and mhigh is set to 0.001 CU/dB
        inverse:
            activates the inverse loudness function

    OUTPUT:
        y:
            either CU (inverse=false) or levels (inverse=true)
    Authors: Dirk Oetting, SE 16.05.2013 16:27, DO/SE 24.07.2013

    includes DO fixes

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
    ------------------------------------------------------------------------------
    '''
    # x=np.array([x])
    # x=np.array(x)
    if type(x)==int:
        x=np.array([x])
    elif type(x)==list:
        x=np.array([x]).flatten()
    elif type(x)==np.ndarray:
        x=x.flatten()


    if len(fitparams) != 3:
        raise ValueError("fitparams should contain exactly three values.")
    
    Lcut, m_lo, m_hi = fitparams
    
    # Check values of given parameters
    if m_lo <= 0:
        if m_lo == 0:
            m_lo = 0.001
        elif m_lo < 0:
            m_lo = -m_lo
    
    if m_hi <= 0:
        if m_hi == 0:
            m_hi = 0.001
        elif m_hi < 0:
            m_hi = -m_hi
    
    # Calculate point where Bezier function should go through
    L15 = y2x_lin(15, Lcut, 25, m_lo)
    L35 = y2x_lin(35, Lcut, 25, m_hi)
    C = np.array([[L15, Lcut, L35], [15, 25, 35]])
    
    # failed flag
    failed = 0
    
    if not inverse:
        cu = np.ones_like(x) * np.nan
        
        if m_lo == m_hi:
            cu = x2y_lin(x, Lcut, 25, m_lo)
        else:
            # Calculate all values below Lcut
            idx = x <= Lcut
            cu[idx] = x2y_lin(x[idx], Lcut, 25, m_lo)
            
            # Find all values above Lcut
            idx = x > Lcut
            cu[idx] = x2y_lin(x[idx], Lcut, 25, m_hi)
            
            # Calculate transition range between 15 and 35 CU
            idx = (cu > 15) & (cu < 35)
            
            if np.any(idx):
                cu[idx], failed = bezier_x2y_for_3_control_points(x[idx], C)
        
        if failed:
            y = np.array([])
            return y
        
        # Limit to 0 to 50
        cu = np.clip(cu, 0, 50)
        y = cu
    else:
        # Limit x from 0 to 50
        x = np.clip(x, 0, 50)
        
        if m_lo == m_hi:
            y = y2x_lin(x, Lcut, 25, m_hi)
        else:
            levels = np.ones_like(x) * np.nan
            
            # Find all values below CU = 15 
            # idx = x <= 15 np.array(b, dtype=int)

            idx=np.where(x <= 15)[0]
            levels[idx] = y2x_lin(x[idx], Lcut, 25, m_lo)
            # if len(x[idx]) > 1:
            #     levels[idx] = y2x_lin(x[idx], Lcut, 25, m_lo)
            # elif len(x[idx]) == 1:
            #     levels = y2x_lin(x[idx], Lcut, 25, m_lo)
            
            # Find all values above CU = 35
            # idx = x >= 35
            idx=np.where(x >= 35)
            levels[idx] = y2x_lin(x[idx], Lcut, 25, m_hi)
            # if len(x[idx]) > 1:
            #     levels[idx] = y2x_lin(x[idx], Lcut, 25, m_hi)
            # elif len(x[idx]) == 1:
            #     levels= y2x_lin(x[idx], Lcut, 25, m_hi)
            
            # Calculate transition range between 15 and 35 CU
            idx = (x > 15) & (x < 35)
            
            if np.any(idx):
                levels[idx], failed = bezier_x2y_for_3_control_points(x[idx], C, True)
            
            if failed:
                y = np.array([])
                return y
            
            y = levels
    
    return y

def x2y_lin(x, x0, y0, m):
    return y0 + m * (x - x0)

def y2x_lin(y, x0, y0, m):
    return (y - y0) / m + x0

def bezier_x2y_for_3_control_points(x, C, inverse=False):
    failed = 0

    # Calculate t of x
    t = np.nan
    y = np.zeros_like(x)
    y0 = C[1, 0]
    y1 = 2 * C[1, 1] - 2 * C[1, 0]
    y2 = C[1, 0] - 2 * C[1, 1] + C[1, 2]

    x2 = C[0, 0] - 2 * C[0, 1] + C[0, 2]
    x1 = 2 * C[0, 1] - 2 * C[0, 0]
    x0 = C[0, 0]

    if not inverse:
        # m_low and m_high are not identical
        if x2 != 0:
            t1 = (-x1 / (2 * x2) + 0.5 * np.sqrt((x1 / x2) ** 2 - 4 * (x0 - x) / x2))
            t2 = (-x1 / (2 * x2) - 0.5 * np.sqrt((x1 / x2) ** 2 - 4 * (x0 - x) / x2))
            
            # Check if t values are between zero and one
            if np.any(t1) and np.all(np.imag(t1) == 0) and np.min(t1) >= 0 and np.max(t1) <= 1:
                t = t1
            elif np.any(t2) and np.all(np.imag(t2) == 0) and np.min(t2) >= 0 and np.max(t2) <= 1:
                t = t2
            else:
                print("Something strange happens in BezierX2YFor3ControlPoints")
                failed = 1
                return y, failed
        # L_cut ~= L15 should always be the case
        elif x1 != 0:
            t = (x - x0) / x1
        
    # If inverse
    else:
        t = x / y1 - y0 / y1
    
    # Calculate y of t
    idx = (t >= 0) & (t <= 1)
    if not inverse:
        y[idx] = y2 * t[idx] ** 2 + y1 * t[idx] + y0
    else:
        y[idx] = x2 * ((t + x1 / (2 * x2)) ** 2) - (x1 ** 2) / (4 * x2) + x0
    
    return y, failed

# Example usage:
# x_values = np.linspace(0, 50, 100)
# fit_params = [25, 0.01, 0.02]
# result = loudness_function_bh2002(x_values, fit_params)
# print(result)
