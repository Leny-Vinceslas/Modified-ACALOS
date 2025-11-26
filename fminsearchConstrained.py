import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fmin
from scipy.optimize import fmin_powell
from scipy.optimize import fmin_cg
from scipy.optimize import fmin_slsqp
from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_tnc

def fminsearchConstrained(fun, x0, lb, ub,optAlg, options=None, *args):
    '''Constrained function minimization

 Based on fminsearchbnd.m (BSD license)
 Copyright (c) 2006, John D'Errico
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are
 met:
 
     * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
     * Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in
       the documentation and/or other materials provided with the distribution
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.

 v0.1 Stephan Ewert, 2014'''
 
    bounded = [any([np.isfinite(lb[i]), np.isfinite(ub[i])]) for i in range(len(ub))]

    def sinetransform(x, lb, ub, bounded):
        out = x.copy()
        for i in range(len(x)):
            if bounded[i] == 1:
            # if bounded[i] == True:
                out[i] = (np.sin(x[i]) + 1) / 2  # periodically ranges from 0 to 1
                out[i] = out[i] * (ub[i] - lb[i]) + lb[i]  # ranges from lower bound to upper bound
        return out

    # Transform starting values using the inverse of sinetransform
    x0t = x0.copy()
    for i in range(len(x0)):
        if bounded[i]:
            if x0[i] < lb[i]:
                x0t[i] = -np.pi / 2
            elif x0[i] > ub[i]:
                x0t[i] = np.pi / 2
            else:
                if (ub[i] - lb[i])==0:
                    x0t[i]=np.nan
                else:
                    x0t[i] = 2 * (x0[i] - lb[i]) / (ub[i] - lb[i]) - 1
                
                # x0t[i] = 2 * (x0[i] - lb[i]) / (ub[i] - lb[i]) - 1
                
                x0t[i] = 2 * np.pi + np.arcsin(max(-1,min(1,x0t[i])))
                # x0t[i] = 2 * np.pi + np.arcsin(np.clip(x0t[i], -1, 1))

    # Internal constraint function defined inline
    def constraintfun(x, *args):
        return fun(sinetransform(x, lb, ub, bounded), *args)

    # Call fmin with constraints
    feval=0

    if optAlg=='NEL':
        result,feval,_,_,_ = fmin(constraintfun, x0t, args=args, xtol=1e-4, ftol=1e-4, maxiter=200*len(x0t), maxfun=200*len(x0t),full_output=True, disp=False)
    
    if optAlg=='CG':
        result=fmin_cg(constraintfun, x0t, fprime=None, args=args, gtol=1e-05, epsilon=1.4901161193847656e-08, maxiter=None, full_output=0, disp=False)

    if optAlg=='SLS':
        result=fmin_slsqp(constraintfun, x0t, eqcons=(), f_eqcons=None, ieqcons=(), f_ieqcons=None, bounds=(), fprime=None, fprime_eqcons=None, fprime_ieqcons=None, args=args, iter=100, acc=1e-06, iprint=1, disp=False, full_output=0, epsilon=1.4901161193847656e-08, callback=None)

    if optAlg=='TNC':
        result=fmin_tnc(constraintfun, x0t, fprime=None, args=args, approx_grad=0)

    if optAlg=='TRU':
        result=minimize(constraintfun, x0t, args=args, method='trust-constr', jac=None, hess=None, hessp=None, bounds=None, tol=None, callback=None)
        result=result.x
    
    result = sinetransform(result, lb, ub, bounded)

    return result,feval