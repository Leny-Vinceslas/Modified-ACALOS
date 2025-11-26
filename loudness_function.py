from loudness_function_bh2002 import loudness_function_bh2002

def loudness_function(x, fitparams, inverse=False):
    """
    Wrapper allowing [m_low, HTL, m_high, (optional UCL)] parametrization.
    """
    if len(fitparams) < 3:
        raise ValueError("fitparams must be [m_low, HTL, m_high, (UCL)]")

    CP = 25.0
    m_low, HTL = fitparams[0], fitparams[1]
    b = 2.5 - m_low * HTL
    Lcut = (CP - b) / m_low

    if len(fitparams) == 4:
        UCL = fitparams[3]
        if Lcut >= UCL:
            UCL = (50.0 - b) / m_low
        m_high = (50.0 - CP) / (UCL - Lcut)
    else:
        m_high = fitparams[2]

    return loudness_function_bh2002(x, [Lcut, m_low, m_high], inverse)


def loudness_function_Lcut(x, fitparams, inverse=False):
    """
    Variant that accepts [Lcut, m_low, m_high] directly.
    """
    if len(fitparams) != 3:
        raise ValueError("fitparams must be [Lcut, m_low, m_high]")
    return loudness_function_bh2002(x, fitparams, inverse)
