import numpy as np
from .utils import db_to_lin, lin_to_db, ppb_day_to_D_frac



class Oscillator: 
    def __init__(self, fosc=10e6, fc=1e9, fs=2.0, T=60.0,
                 n_periods=10, n_captures=1, start_period=None,
                 start_phi_lo_0=False, start_phi_out_0=False,
                 ppb_day=0.0, include_drift=False,
                 coeffs_lin=None, coeffs_dB=None, seed=None):

        if coeffs_lin is not None:
            self.a = dict(coeffs_lin)
            self.a_dB = {k: lin_to_db(v) for k, v in self.a.items()}
        elif coeffs_dB is not None:
            self.a_dB = dict(coeffs_dB)
            self.a = {k: db_to_lin(v) for k, v in self.a_dB.items()}
        else:
            raise ValueError("Must provide either coeffs_lin or coeffs_dB")

        self.a4 = self.a["a4"]
        self.a3 = self.a["a3"]
        self.a2 = self.a["a2"]
        self.a1 = self.a["a1"]
        self.a0 = self.a["a0"]

        self.fosc = fosc
        self.fc = fc
        self.rf_scale = np.sqrt(2) * (fc / fosc)

        self.include_drift = include_drift
        self.ppb_day = ppb_day
        self.D_frac = ppb_day_to_D_frac(ppb_day) if include_drift else 0.0

        self.h = {
            "h-2": self.a4 / fosc**2,
            "h-1": self.a3 / fosc**2,
            "h0":  self.a2 / fosc**2,
            "h1":  self.a1 / fosc**2,
            "h2":  self.a0 / fosc**2,
        }

        self.fs = fs
        self.T = T
        self.n_periods = n_periods
        self.n_captures = n_captures
        self.start_period = int(start_period if start_period is not None else n_periods // 2)
        self.start_phi_lo_0 = start_phi_lo_0
        self.start_phi_out_0 = start_phi_out_0

        self.build_grid()

        self.rng = np.random.default_rng(seed)

        self.phi_lo = None
        self.phi_out = None
        



    def build_grid(self):
        T_total = self.T * self.n_periods
        self.N_total = int(np.ceil(T_total * self.fs))
        if self.N_total % 2:
            self.N_total += 1

        self.t = np.arange(self.N_total) / self.fs
        self.f = np.fft.rfftfreq(self.N_total, d=1.0 / self.fs)
        self.df = self.fs / self.N_total

        _, self.S1_vals = self.S1(self.f)
        _, self.Sy_vals = self.Sy(self.f)

        self.fmin_eff = self.f[1]
        self.tau_min = 1.0 / (2.0 * self.f[-1])
        self.tau_max = 1.0 / (2.0 * self.fmin_eff)
        self.taus = np.geomspace(self.tau_min, self.tau_max, 100)

        self.start_idx = int(round(self.start_period * self.T * self.fs))
        self.seg_len = int(round(self.n_captures * self.T * self.fs))
        self.end_idx = self.start_idx + self.seg_len
        self.t_out = self.t[self.start_idx:self.end_idx]

    def update_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def S1(self, f=None, fmin=-2, fmax=4, num=2000):
        if f is None:
            f = np.logspace(fmin, fmax, num)
        f = np.asarray(f, dtype=float)

        with np.errstate(divide='ignore', invalid='ignore'):
            S_ts = (self.a4 / f**4) + (self.a3 / f**3) + (self.a2 / f**2) + (self.a1 / f) + self.a0

        infs = ~np.isfinite(S_ts)
        S_ts[infs] = 0.0

        S1 = np.zeros_like(f, dtype=float)
        S1[f > 0] = 2.0 * S_ts[f > 0]


        return f, S1
    
    def Sy(self, f=None):
        if f is None: 
            f = self.f
        with np.errstate(divide='ignore', invalid='ignore'):
            Sy_vals = ((self.h['h-2']*(f**-2)) + (self.h['h-1']*(f**-1)) + (self.h['h0']*(f**0)) + (self.h['h1']*(f**1)) + (self.h['h2']*(f**2))) * 2.0
            Sy_vals[~np.isfinite(Sy_vals)] = 0.0
        
        return f, Sy_vals
    
    def realize_phase_error(self): 
        psi = self.rng.uniform(0, 2*np.pi, size=self.f.shape)

        Z = np.zeros_like(self.f, dtype=np.complex128)
        Z[1:-1] = (self.N_total * np.sqrt(0.5 * self.S1_vals[1:-1] * self.df)) * np.exp(1j * psi[1:-1])
        Z[0] = 0.0
        Z[-1] = (self.N_total * np.sqrt(self.S1_vals[-1] * self.df)) * np.cos(psi[-1])

        phi_lo = np.fft.irfft(Z, n=self.N_total)

        if self.include_drift and self.D_frac != 0.0:
            phi_lo = phi_lo + (np.pi * self.fosc * self.D_frac) * (self.t**2)
            # sigma_D = self.ppb_day_to_D_frac(abs(self.ppb_day))
            # D_real = self.rng.normal(0.0, sigma_D) 
            # phi_lo = phi_lo + (np.pi * self.fosc * D_real) * (self.t**2)
        
        if self.start_phi_lo_0:
            phi_lo = phi_lo - phi_lo[0]

        self.phi_lo = phi_lo
        self.phi_out = phi_lo[self.start_idx:self.end_idx]
        if self.start_phi_out_0:
            self.phi_out = self.phi_out - self.phi_out[0] 
    
    

    def time_fluctuation(self, phi=None):
        return phi / (2 * np.pi * self.fosc)

    def fractional_frequency(self, phi=None):
        Ts = 1.0 / self.fs
        return np.diff(phi) / (2 * np.pi * self.fosc * Ts)

    def phase_to_rf(self, phi=None):
        return phi * self.rf_scale
    

    def allan_dev_from_phase_realization(self, phi_type='lo', cutoff=None):
        tau0 = 1.0/self.fs
        m_all = np.maximum(1, np.round(self.taus / tau0).astype(int))
        m_all = np.unique(m_all)

        if phi_type == 'lo':
            x = self.phi_lo
        elif phi_type == 'out': 
            x = self.phi_out


        N = len(x)
        m_valid = m_all[m_all*2 < N]
        if cutoff is not None: 
            m_valid = m_valid[:-cutoff]
        taus_valid = m_valid * tau0
        adev = np.empty_like(taus_valid, dtype=float)

        for k, m in enumerate(m_valid): 
            d2 = x[2*m:] - 2.0*x[m:-m] + x[:-2*m]
            denom = 2.0 * (2.0 * np.pi * self.fosc * m * tau0)**2 * (N - 2*m)
            adev[k] = np.sqrt(np.sum(d2*d2) / denom)
        return taus_valid, adev

    def allan_dev_from_phase_psd(self, f=None, S1_vals=None, taus=None):
        if taus is None: 
            taus = self.taus
        if f is None: 
            f = self.f
        if S1_vals is None:  
            S1_vals = self.S1_vals
        taus = np.asarray(taus)

        vals = []
        for tau in taus:
            scaler = 8.0 / ((2*np.pi * self.fosc * tau)**2)
            kernel = np.sin(np.pi * f * tau)**4
            integrand = S1_vals * kernel
            integral = np.trapezoid(integrand, f)
            sigma_y2 = scaler * integral
            if self.include_drift and self.D_frac != 0.0: 
                sigma_y2 += 0.5 * (self.D_frac * tau)**2
            vals.append(np.sqrt(sigma_y2))
        return np.asarray(taus), np.asarray(vals)
    

    def allan_dev_from_freq_psd(self, f=None, Sy_vals=None, taus=None):
        if f is None: 
            f = self.f
        if Sy_vals is None: 
            Sy_vals = self.Sy_vals
        if taus is None: 
            taus = self.taus

        vals = []
        for t in taus:
            m = f > 0
            kernel = np.zeros_like(f, dtype=float)
            kernel[m] = ((np.sin(np.pi * f[m] * t))**4) / ((np.pi * f[m] * t)**2)
            integrand = Sy_vals * kernel
            integral = np.trapezoid(integrand, f)
            sigma_y2 = 2.0 * integral
            if self.include_drift and self.D_frac != 0.0: 
                sigma_y2 += 0.5 * (self.D_frac * t)**2
            vals.append(np.sqrt(sigma_y2))
        return np.asarray(taus), np.asarray(vals)
    
    

    
    def variance_check(self):
        var_pred = np.trapezoid(self.S1_vals, self.f)
        var_emp = np.var(self.phi_lo)
        rel_error = abs(var_emp - var_pred) / max(var_pred, 1e-300) * 100
        return {"expected": var_pred, "realized": var_emp, "rel_error_pct": rel_error}
