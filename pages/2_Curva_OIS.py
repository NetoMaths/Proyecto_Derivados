import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import brentq
import plotly.graph_objects as go

st.set_page_config(page_title="Curva OIS", layout="wide")
st.title("📈 Curva OIS de TIIE de Fondeo")

# ========== DATOS Y FUNCIONES (copiadas del notebook) ==========
DC = 28.0/360.0
NODES = [3, 6, 9, 13, 26, 39, 52, 65, 91, 130, 195, 260, 390]
PAR = [6.55, 6.60, 6.67, 6.87, 7.42, 7.75, 7.94, 8.08, 8.28, 8.48, 8.69, 8.78, 8.75]
PAR = [p/100.0 for p in PAR]

@st.cache_data
def bootstrap_curve():
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

    DF = bootstrap_ois(NODES, PAR)
    idx = np.arange(0, NODES[-1]+1)
    years = idx*28/360.0
    zero = np.where(years>0, -np.log(DF)/years, np.nan)
    fwd28 = np.full_like(DF, np.nan)
    fwd28[1:] = (DF[:-1]/DF[1:] - 1.0)/DC

    def par_rate_node(n):
        return (1.0 - DF[n])/(DC*np.sum(DF[1:n+1]))

    nodos = pd.DataFrame({
        "Nodo": [f"{n}F1" for n in NODES],
        "Tenor": ["3M","6M","9M","1A","2A","3A","4A","5A","7A","10A","15A","20A","30A"],
        "Tasa par mercado (%)": [s*100 for s in PAR],
        "Tasa par modelo (%)": [par_rate_node(n)*100 for n in NODES],
        "DF(T)": [DF[n] for n in NODES]
    })
    curva = pd.DataFrame({
        "Periodo (28d)": idx[1:],
        "Plazo (días)": idx[1:]*28,
        "Plazo (años)": years[1:],
        "DF": DF[1:],
        "Tasa cero cont. (%)": zero[1:]*100,
        "Fwd 28d (%)": fwd28[1:]*100
    })
    return DF, years, zero, fwd28, nodos, curva

DF, years, zero, fwd28, nodos, curva = bootstrap_curve()
# ===============================================================

st.subheader("Verificación del bootstrapping")
st.dataframe(nodos, use_container_width=True)

st.subheader("Curva completa (primeras y últimas filas)")
st.dataframe(pd.concat([curva.head(8), curva.tail(5)]), use_container_width=True)

# Gráficos
fig = go.Figure()
fig.add_trace(go.Scatter(x=years[1:], y=DF[1:], mode='lines', name='DF(t)'))
fig.update_layout(title="Factor de descuento", xaxis_title="años", yaxis_title="DF")
st.plotly_chart(fig, use_container_width=True)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=years[1:], y=zero[1:]*100, mode='lines', name='Tasa cero'))
for n, lbl in zip(NODES, nodos["Tenor"]):
    fig2.add_trace(go.Scatter(x=[years[n]], y=[zero[n]*100], mode='markers', name=lbl))
fig2.update_layout(title="Tasa cero continua (%)", xaxis_title="años")
st.plotly_chart(fig2, use_container_width=True)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=years[1:], y=fwd28[1:]*100, mode='lines', name='Forward 28d'))
fig3.update_layout(title="Forward 28d implícita (%)", xaxis_title="años")
st.plotly_chart(fig3, use_container_width=True)