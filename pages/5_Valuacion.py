import streamlit as st
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
import plotly.graph_objects as go

st.set_page_config(page_title="Valuación del Swaption", layout="wide")
st.title("💼 Valuación del Swaption (8M × 10A)")

# Copiamos de nuevo las funciones necesarias (simplificado)
DC = 28.0/360.0
NODES = [3, 6, 9, 13, 26, 39, 52, 65, 91, 130, 195, 260, 390]
PAR = [6.55, 6.60, 6.67, 6.87, 7.42, 7.75, 7.94, 8.08, 8.28, 8.48, 8.69, 8.78, 8.75]
PAR = [p/100.0 for p in PAR]

from scipy.optimize import brentq
def bootstrap_ois(nodes, par, dc=DC):
    Nmax = nodes[-1]
    lnP = np.zeros(Nmax + 1)
    knot = {0: 0.0}
    prev = 0
    for nk, S in zip(nodes, par):
        ann_known = dc*np.sum(np.exp(lnP[1:prev+1])) if prev >= 1 else 0.0
        lp = knot[prev]
        def eq(x, lp=lp, prev=prev, nk=nk, S=S, ann_known=ann_known):
            j = np.arange(prev+1, nk+1)
            seg = lp + (j-prev)/(nk-prev)*(x-lp)
            ann = ann_known + dc*np.sum(np.exp(seg))
            return S*ann - (1.0 - np.exp(x))
        x = brentq(eq, -5.0, 1e-12, xtol=1e-14)
        j = np.arange(prev+1, nk+1)
        lnP[prev+1:nk+1] = lp + (j-prev)/(nk-prev)*(x-lp)
        knot[nk] = x
        prev = nk
    return np.exp(lnP)

@st.cache_data
def get_DF():
    return bootstrap_ois(NODES, PAR)

DF = get_DF()

def fwd_swap_and_annuity(expiry_y, tenor_y):
    def coupons_from_years(t): return int(round(t*360.0/28.0))
    a = coupons_from_years(expiry_y)
    b = min(a + coupons_from_years(tenor_y), NODES[-1])
    A = DC*np.sum(DF[a+1:b+1])
    return (DF[a]-DF[b])/A, A, a, b

def sabr_lognormal_vol(K, f, T, alpha, beta, rho, nu):
    K = np.asarray(K, float)
    Fb = (f*K)**((1.0-beta)/2.0)
    L = np.log(f/K)
    z = (nu/alpha)*Fb*L
    sq = np.sqrt(1.0 - 2.0*rho*z + z*z)
    chi = np.log((sq + z - rho)/(1.0 - rho))
    chi_safe = np.where(np.abs(z) < 1e-7, 1.0, chi)
    ratio = np.where(np.abs(z) < 1e-7, 1.0 - 0.5*rho*z, z/chi_safe)
    den = Fb*(1.0 + ((1.0-beta)**2/24.0)*L**2 + ((1.0-beta)**4/1920.0)*L**4)
    B = (1.0 + (((1.0-beta)**2/24.0)*alpha**2/(f*K)**(1.0-beta)
                + 0.25*rho*beta*nu*alpha/Fb
                + (2.0-3.0*rho**2)/24.0*nu**2)*T)
    return alpha/den*ratio*B

def alpha_from_atm(sig_atm, f, T, beta, rho, nu):
    def g(a): return float(sabr_lognormal_vol(f, f, T, a, beta, rho, nu)) - sig_atm
    return brentq(g, 1e-6, 5.0, xtol=1e-12)

def black76_swaption(f, K, T, sigma, annuity, notional=1.0, payer=True):
    d1 = (np.log(f/K) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if payer:
        core = f*norm.cdf(d1) - K*norm.cdf(d2)
    else:
        core = K*norm.cdf(-d2) - f*norm.cdf(-d1)
    return notional*annuity*core

# Parámetros fijos del swaption
EXPIRY_FIX = 8/12
TENOR_FIX = 10.0
SIGMA_N_ATM_BBG = 156.02/1e4

f_sw, A_sw, _, _ = fwd_swap_and_annuity(EXPIRY_FIX, TENOR_FIX)
sig_atm_LN = SIGMA_N_ATM_BBG / f_sw

# Parámetros SABR aproximados (tomados de la calibración del notebook)
rho_fix = -0.002   # interpolado en 8M
nu_fix = 1.278
alpha_fix = alpha_from_atm(sig_atm_LN, f_sw, EXPIRY_FIX, 1.0, rho_fix, nu_fix)

# Interfaz de usuario
st.sidebar.header("Parámetros del swaption")
notional = st.sidebar.number_input("Nocional (MXN)", value=10_000_000, step=1_000_000)
strike_pct = st.sidebar.number_input("Strike (%)", value=round(f_sw*100,4), step=0.01, format="%.4f")
direction = st.sidebar.radio("Tipo", ["Receptor", "Pagador"])

strike = strike_pct / 100
sigma_K = sabr_lognormal_vol(strike, f_sw, EXPIRY_FIX, alpha_fix, 1.0, rho_fix, nu_fix)
premium = black76_swaption(f_sw, strike, EXPIRY_FIX, sigma_K, A_sw, notional, payer=(direction=="Pagador"))

col1, col2 = st.columns(2)
col1.metric("Forward swap rate", f"{f_sw*100:.4f}%")
col2.metric("Anualidad (PVBP)", f"{A_sw:.4f}")
col1.metric("Vol ATM log-normal", f"{sig_atm_LN*100:.2f}%")
col2.metric("Vol SABR (K)", f"{sigma_K*100:.3f}%")
st.metric("Prima", f"${premium:,.2f} MXN", delta=f"{premium/notional*10000:.1f} pb del nocional")

# Gráfico prima vs strike
strikes = np.linspace(f_sw*0.6, f_sw*1.4, 80)
vols = sabr_lognormal_vol(strikes, f_sw, EXPIRY_FIX, alpha_fix, 1.0, rho_fix, nu_fix)
premiums_pay = [black76_swaption(f_sw, k, EXPIRY_FIX, v, A_sw, notional, True) for k,v in zip(strikes,vols)]
premiums_rec = [black76_swaption(f_sw, k, EXPIRY_FIX, v, A_sw, notional, False) for k,v in zip(strikes,vols)]
fig = go.Figure()
fig.add_trace(go.Scatter(x=strikes*100, y=premiums_pay, mode='lines', name='Pagador'))
fig.add_trace(go.Scatter(x=strikes*100, y=premiums_rec, mode='lines', name='Receptor'))
fig.add_vline(x=strike*100, line_dash="dash", line_color="black", annotation_text="Strike")
fig.add_vline(x=f_sw*100, line_dash="dot", line_color="gray", annotation_text="ATM")
fig.update_layout(xaxis_title="Strike (%)", yaxis_title="Prima (MXN)", height=500)
st.plotly_chart(fig, use_container_width=True)