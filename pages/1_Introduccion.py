import streamlit as st

st.set_page_config(page_title="Introducción", layout="wide")

st.title("📘 Valuación de Swaption Europeo sobre TIIE de Fondeo (F‑TIIE)")

st.markdown("""
**Fecha de valuación:** 20 de mayo de 2026  
**Mercado:** MXN – TIIE de Fondeo (tasa overnight libre de riesgo)

### Instrumento objetivo (Bloomberg SWPM)
- Swaption europeo **receptor**
- Vencimiento de la opción: **8 meses**
- Swap subyacente: **10 años**
- Nocional: **10,000,000 MXN**
- Forward ATM: **8.7241%**
- Volatilidad normal ATM: **156.02 pb**

### Fuentes de datos
- Tasas par de swaps F‑TIIE: Boletín MexDer (20‑may‑2026) – **13 nodos representativos**
- Matriz de volatilidades: Bloomberg VCUB *MXN TIIE‑F RFR BVOL Cube* (vols normales, tenor 1A)

### Metodología (resumen)
1. **Curva OIS**: bootstrapping con interpolación log‑lineal sobre factores de descuento (estándar OIS).  
2. **Volatilidad**: modelo SABR con β=1 (log‑normal), calibrado por vencimiento.  
3. **Valuación**: Black‑76 sobre la anualidad (PVBP) del swap subyacente.  

Para más detalles, consulte la pestaña **Conclusiones**.
""")

# Opcional: mostrar tabla de nodos representativos
st.subheader("Nodos representativos (tasas par)")
st.dataframe({
    "Periodos (28d)": [3,6,9,13,26,39,52,65,91,130,195,260,390],
    "Tenor": ["3M","6M","9M","1A","2A","3A","4A","5A","7A","10A","15A","20A","30A"],
    "Tasa par (%)": [6.55,6.60,6.67,6.87,7.42,7.75,7.94,8.08,8.28,8.48,8.69,8.78,8.75]
}, use_container_width=True)