import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import plotly.graph_objects as go

st.set_page_config(page_title="Superficie SABR", layout="wide")
st.title("🎛️ Superficie de Volatilidad SABR (β=1)")

# ========== DATOS Y FUNCIONES ==========
# Parámetros del cubo VCUB (normal, pb)
EXPIRY_LBL = ["1Mo","3Mo","6Mo","9Mo","1Yr","2Yr","3Yr","4Yr","5Yr","6Yr",
              "7Yr","8Yr","9Yr","10Yr","12Yr","15Yr","20Yr","25Yr","30Yr"]
EXPIRY_Y   = [1/12,3/12,6/12,9/12,1,2,3,4,5,6,7,8,9,10,12,15,20,25,30]
STRIKE_ABS = [0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.05,0.06,0.07,0.08,0.09,0.10]
VOL_N = np.array([
 [470.89,451.57,430.25,407.43,383.33,358.08,331.70,275.50,214.09,146.43,102.12,170.03,242.52],
 [415.75,399.20,380.89,361.27,340.57,318.90,296.31,248.47,196.97,141.58,107.05,166.79,230.60],
 [333.09,320.70,306.91,292.10,276.49,260.19,243.28,207.96,171.28,134.22,114.29,161.76,212.55],
 [250.44,242.21,232.93,222.95,212.42,201.49,190.26,167.46,145.58,126.84,121.52,156.70,194.47],
 [140.23,137.55,134.30,130.74,127.00,123.23,119.57,113.46,111.32,117.00,131.14,149.96,170.37],
 [144.61,141.73,138.28,134.49,130.50,126.42,122.39,115.18,111.16,114.36,127.00,145.66,166.47],
 [147.07,144.16,140.68,136.85,132.80,128.65,124.53,116.97,112.19,114.04,125.41,143.47,164.12],
 [147.96,145.03,141.52,137.65,133.56,129.36,125.16,117.35,112.07,113.03,123.51,141.15,161.66],
 [148.44,145.48,141.94,138.04,133.91,129.65,125.38,117.31,111.49,111.47,120.93,138.06,158.39],
 [148.74,145.76,142.20,138.27,134.11,129.82,125.50,117.28,111.12,110.45,119.21,135.98,156.19],
 [150.00,147.01,143.43,139.35,135.10,130.97,126.60,118.22,111.73,110.41,118.35,134.65,154.71],
 [151.05,148.05,144.45,140.40,136.27,131.91,127.50,119.00,112.26,110.44,117.73,133.64,153.59],
 [150.36,147.33,143.72,139.55,135.13,131.12,126.67,118.05,111.06,108.76,115.50,131.12,150.96],
 [149.47,146.41,142.78,138.78,134.53,130.12,125.64,116.89,109.62,106.75,112.79,128.01,147.71],
 [149.87,146.94,143.38,138.83,135.20,130.82,126.34,117.55,110.08,106.71,111.94,126.60,146.10],
 [150.22,147.41,143.92,140.02,135.83,131.46,126.99,118.16,110.53,106.72,111.20,125.32,144.61],
 [150.41,147.68,144.24,140.36,135.87,131.84,127.38,118.53,110.81,106.75,110.79,124.56,143.73],
 [150.24,147.44,143.96,140.05,135.87,131.50,127.04,118.20,110.56,106.72,111.16,125.23,144.51],
 [150.14,147.30,143.79,139.88,135.68,131.31,126.84,118.02,110.42,106.71,111.37,125.62,144.96],
]) / 1e4
ATM_N = np.array([99.93,106.03,113.82,121.25,131.00,131.05,131.48,130.74,129.41,128.49,
                  128.50,128.49,126.72,124.59,124.59,124.59,124.59,124.59,124.59]) / 1e4
TENOR_VCUB = 1.0

# Funciones
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

def normal_to_lognormal(sigN, f, K):
    K = np.asarray(K, float)
    return np.where(np.abs(f-K) < 1e-8, sigN/f, sigN*np.log(f/K)/(f-K))

def calibrate_sabr(strikes, vols, f, T, beta=1.0):
    strikes = np.asarray(strikes, float); vols = np.asarray(vols, float)
    atm0 = vols[np.argmin(np.abs(strikes - f))]
    x0 = [max(atm0, 1e-3), 0.0, 0.5]
    def res(x):
        return sabr_lognormal_vol(strikes, f, T, x[0], beta, x[1], x[2]) - vols
    sol = least_squares(res, x0, bounds=([1e-5,-0.999,1e-4],[5.0,0.999,5.0]),
                         max_nfev=20000, xtol=1e-12, ftol=1e-12)
    a, r, n = sol.x
    return dict(alpha=a, beta=beta, rho=r, nu=n, rmse=float(np.sqrt(np.mean(sol.fun**2))))

# Calibración (cache)
@st.cache_data
def calibrate_all():
    # Necesitamos f para cada vencimiento. Para simplificar, usamos f aproximada (sacada de la curva OIS)
    # En tu notebook ya tenías fwd_swap_and_annuity; aquí usamos valores típicos (podríamos importar de la curva)
    # Dado que es demo, usamos f aproximados del notebook
    f_approx = [0.0698,0.0720,0.0752,0.0785,0.0801,0.0849,0.0862,0.0876,0.0895,0.0896,
                0.0920,0.0922,0.0922,0.0952,0.0955,0.0953,0.0873,0.0849,0.0849]
    rows = []
    calib_dict = {}
    for i, lbl in enumerate(EXPIRY_LBL):
        T = EXPIRY_Y[i]
        f = f_approx[i]
        row = VOL_N[i]
        atmn = ATM_N[i]
        Ks = np.array(STRIKE_ABS)
        sel = np.abs(Ks/f - 1.0) <= 0.55
        Kall = np.append(Ks[sel], f)
        LNall = np.append(normal_to_lognormal(row[sel], f, Ks[sel]), atmn/f)
        c = calibrate_sabr(Kall, LNall, f, T, beta=1.0)
        calib_dict[lbl] = c
        rows.append([lbl, f*100, atmn/f*100, c["alpha"], c["rho"], c["nu"], c["rmse"]*1e4])
    return pd.DataFrame(rows, columns=["Tenor","fwd (%)","ATM LN (%)","alpha","rho","nu","RMSE (pb)"]), calib_dict

calib_df, calib_dict = calibrate_all()
st.dataframe(calib_df, use_container_width=True)

# Superficie 3D
mny_grid = np.linspace(-0.5, 0.5, 41)
Z = np.zeros((len(EXPIRY_Y), len(mny_grid)))
for i, lbl in enumerate(EXPIRY_LBL):
    c = calib_dict[lbl]
    f = calib_df.loc[i, "fwd (%)"]/100
    Z[i,:] = sabr_lognormal_vol(f*(1+mny_grid), f, EXPIRY_Y[i], c["alpha"], 1.0, c["rho"], c["nu"])*100

fig = go.Figure(data=[go.Surface(x=mny_grid, y=EXPIRY_Y, z=Z, colorscale="Viridis")])
fig.update_layout(title="Superficie SABR (β=1)", scene=dict(
    xaxis_title="Moneyness (K/F-1)", yaxis_title="Vencimiento (años)", zaxis_title="Vol LN (%)"),
    height=600)
st.plotly_chart(fig, use_container_width=True)