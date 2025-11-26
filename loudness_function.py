from loudness_function_bh2002 import loudness_function_bh2002 


def loudness_function(x, fitparams, inverse=False):
    """
    Calculate the loudness function according to fitparams.

    Parameters:
    - x: Levels to calculate CU or CU values to calculate levels
    - fitparams: [m_low, HTL, m_high, (UCL)]
      UCL is optional and will override m_high
      Values describing the loudness function.
    - inverse: Activates the inverse loudness function (default is False)

    Returns:
    - y: Either CU (inverse=False) or levels (inverse=True)
    """
    if len(fitparams) < 3:
        raise ValueError("fitparams should have at least 3 values: [m_low, HTL, m_high, (UCL)]")

    CP = 25
    m_lo = fitparams[0]
    HTL = fitparams[1]
    b = 2.5 - m_lo * HTL
    Lcut = (CP - b) / m_lo

    if len(fitparams) == 4:
        UCL = fitparams[3]
        if Lcut >= UCL:
            print('Warning: Wrong setting for UCL, function converted to linear')
            UCL = (50 - b) / m_lo
        m_hi = (50 - CP) / (UCL - Lcut)
    else:
        m_hi = fitparams[2]

    fitparams = [Lcut, m_lo, m_hi]

    return loudness_function_bh2002(x, fitparams, inverse)


def loudness_function_Lcut(x, fitparams, inverse=False):
    """
    Variant that accepts [Lcut, m_low, m_high] directly.

    Parameters:
    - x: Levels to calculate CU or CU values to calculate levels
    - fitparams: [Lcut, m_low, m_high]
      Values describing the loudness function.
    - inverse: Activates the inverse loudness function (default is False)

    Returns:
    - y: Either CU (inverse=False) or levels (inverse=True)
    """
    if len(fitparams) != 3:
        raise ValueError("fitparams must be [Lcut, m_low, m_high]")

    return loudness_function_bh2002(x, fitparams, inverse)


