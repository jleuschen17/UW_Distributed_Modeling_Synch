import numpy as np
from scipy.optimize import nnls
from .utils import dBc_to_S2, ppb_day_to_D_frac, D_frac_to_ppb_day


def build_pn_matrix(f_off):
    pn_mat = np.column_stack([f_off**-4, f_off**-3, f_off**-2, f_off**-1, np.ones_like(f_off)])
    return pn_mat


def build_adev_matrix_Sphi(taus, fgrid, f0): 
    taus = np.asarray(taus)
    f = np.asarray(fgrid)

    basis = 2.0 * np.column_stack([f**-4, f**-3, f**-2, f**-1, np.ones_like(f)])
    H = np.empty((taus.size, 5), dtype=float)
    
    for i, tau in enumerate(taus): 
        scale = 8.0 / ((2*np.pi * f0 * tau)**2)
        kernel = np.sin(np.pi * f * tau)**4
        integrand = basis * kernel[:, None]
        col_ints = np.trapezoid(integrand, f, axis=0)
        H[i, :] = scale * col_ints
    return H

def build_adev_matrix_Sy(taus, fgrid): 
    taus = np.asarray(taus)
    f = np.asarray(fgrid)

    basis = np.column_stack([f**-2, f**-1, np.ones_like(f), f**1, f**2])
    H = np.empty((taus.size, 5), dtype=float)
    
    for i, tau in enumerate(taus): 
        kernel = ((np.sin(np.pi * f * tau))**4) / ((np.pi * f * tau)**2)
        integrand = basis * kernel[:, None]
        col_ints = np.trapezoid(integrand, f, axis=0)
        H[i, :] = 2.0 * col_ints
    return H

def add_drift_col(A_adev, taus): 
    taus = np.asarray(taus, float)
    H_drift = 0.5 * (taus**2)[:, None]
    return np.hstack([A_adev, H_drift])

def fit_oscillator_coeffs_PSD(f_off, L_dBc, weight=True):
    f = np.asarray(f_off, float)
    A = build_pn_matrix(f)
    b = dBc_to_S2(L_dBc)
    if weight:
        w = 1.0 / b
        A, b = A * w[:, None], b * w

    x, _ = nnls(A, b)
    return {"a4": x[0], "a3": x[1], "a2": x[2], "a1": x[3], "a0": x[4]}

def fit_oscillator_coeffs_adev(taus, sigma_y, fgrid, f0, weight=True):
    A = build_adev_matrix_Sphi(taus, fgrid, f0)
    b = np.asarray(sigma_y, float)**2

    if weight:
        w = 1.0 / np.sqrt(b)
        A, b = A * w[:, None], b * w

    x, _ = nnls(A, b)

    return {"a4": x[0], "a3": x[1], "a2": x[2], "a1": x[3], "a0": x[4]}

def fit_oscillator_coeffs_adev_drift(taus, sigma_y, fgrid, f0, weight=True): 
    A = build_adev_matrix_Sphi(taus, fgrid, f0)
    A = add_drift_col(A, taus)
    b = np.asarray(sigma_y, float)**2

    if weight:
        w = 1.0 / np.sqrt(b)
        A, b = A * w[:, None], b * w

    x, _ = nnls(A, b)
    coeffs = {"a4": x[0], "a3": x[1], "a2": x[2], "a1": x[3], "a0": x[4]}

    
    ppb_day = D_frac_to_ppb_day(np.sqrt(x[5]))

    return coeffs, ppb_day


def fit_oscillator_coeffs_joint(f_off, L_dBc, taus, sigma_y, fgrid, f0, pn_weight=True, adev_weight=True):
    f_off = np.asarray(f_off, float)
    taus = np.asarray(taus, float)
    sigma_y = np.asarray(sigma_y, float)

    A_pn = build_pn_matrix(f_off)
    b_pn = dBc_to_S2(L_dBc)

    if pn_weight:
        w_pn = 1.0 / np.maximum(b_pn, 1e-300)
        A_pn, b_pn = A_pn * w_pn[:, None], b_pn * w_pn


    A_adev = build_adev_matrix_Sphi(taus, fgrid, f0)
    b_adev = sigma_y**2

    if adev_weight: 
        w_adev = 1.0 / np.maximum(b_adev, 1e-300)
        A_adev, b_adev = A_adev * w_adev[:, None], b_adev * w_adev

    A = np.vstack([A_pn, A_adev])
    b = np.concatenate([b_pn, b_adev])

    x, _ = nnls(A, b)
    
    return {"a4": x[0], "a3": x[1], "a2": x[2], "a1": x[3], "a0": x[4]}

def fit_oscillator_coeffs_joint_drift(f_off, L_dBc, taus, sigma_y, fgrid, f0, pn_weight=True, adev_weight=True, lambda_pn=True):
    f_off = np.asarray(f_off, float)
    taus = np.asarray(taus, float)
    sigma_y = np.asarray(sigma_y, float)

    A_pn = build_pn_matrix(f_off)
    b_pn = dBc_to_S2(L_dBc)

    if pn_weight:
        w_pn = 1.0 / np.maximum(b_pn, 1e-300)
        A_pn, b_pn = A_pn * w_pn[:, None], b_pn * w_pn

    A_adev = build_adev_matrix_Sphi(taus, fgrid, f0)
    A_adev = add_drift_col(A_adev, taus)
    b_adev = sigma_y**2

    if adev_weight:
        w_adev = 1.0 / np.maximum(b_adev, 1e-300)
        A_adev, b_adev = A_adev * w_adev[:, None], b_adev * w_adev


    A_pn_pad = np.hstack([A_pn, np.zeros((A_pn.shape[0], 1))])

    A = np.vstack([A_pn_pad, A_adev])
    b = np.concatenate([b_pn, b_adev])

    x, _ = nnls(A, b)

    coeffs = {"a4": x[0], "a3": x[1], "a2": x[2], "a1": x[3], "a0": x[4]}
    
    ppb_day = D_frac_to_ppb_day(np.sqrt(x[5]))

    return coeffs, ppb_day

def fit_oscillator_coeffs_joint_2(f_off, L_dBc, taus, sigma_y, fgrid, f0): 
    a_pn = fit_oscillator_coeffs_PSD(f_off, L_dBc)
    a_freq = fit_oscillator_coeffs_adev(taus, sigma_y, fgrid, f0)
    return {'a4': a_freq['a4'], 'a3': a_freq['a3'], 'a2': a_freq['a2'], 'a1': a_pn['a1'], 'a0': a_pn['a0']}


def fit_oscillator_coeffs_joint_drift_2(f_off, L_dBc, taus, sigma_y, fgrid, f0):
    a_pn = fit_oscillator_coeffs_PSD(f_off, L_dBc)
    a_freq, ppb_day = fit_oscillator_coeffs_adev_drift(taus, sigma_y, fgrid, f0)
    oscillator_coeffs_lin =  {'a4': a_freq['a4'], 'a3': a_freq['a3'], 'a2': a_freq['a2'], 'a1': a_pn['a1'], 'a0': a_pn['a0']}
    return oscillator_coeffs_lin, ppb_day