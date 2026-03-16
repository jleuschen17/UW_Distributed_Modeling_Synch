import numpy as np

eps=1e-300


def db_to_lin(x):
    return 10.0 ** (x / 10.0)

def lin_to_db(x):
    return 10.0 * np.log10(np.maximum(x, eps))

def dBc_to_S1(L_dBc):
    return 2.0 * 10.0**(L_dBc / 10.0)

def dBc_to_S2(L_dBc):
    return 10.0**(L_dBc / 10.0)


def S1_to_dBrad2Hz(S1): 
    return 10.0 * np.log10(np.maximum(S1, eps))

def S1_to_dBcHz(S1):
    return 10.0 * np.log10(np.maximum(S1 / 2.0, eps))


def Sphi_to_Sy(f, Sphi, fosc):
    Sy = ((f**2) / ((fosc)**2)) * Sphi
    return Sy

def Sy_to_Sphi(f, Sy, fosc):
    m = f > 0
    Sphi = np.zeros_like(f, dtype=float)
    Sphi[m] = ((fosc**2) / (f[m]**2)) * Sy[m]
    return Sphi

def ppb_day_to_D_frac(ppb_day):
    return (ppb_day * 1e-9) / 86400.0


def D_frac_to_ppb_day(D_frac):
    return float(D_frac) * 86400.0 * 1e9

def dfdt_to_ppb_day(df_dt, f0):
    return (float(df_dt) / float(f0)) * 86400.0 * 1e9
    