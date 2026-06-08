import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Conclusiones", layout="wide")
st.title("📄 Conclusiones y Referencias")

st.markdown("""
## Resumen del modelo
- **Curva OIS** construida correctamente con bootstrapping e interpolación log‑lineal, reproduciendo las tasas par de los 13 nodos representativos de MexDer.
- **Calibración SABR** con β=1 (log‑normal) captura el smile de volatilidad de los swaptions F‑TIIE con buen ajuste (RMSE típico < 30 pb).
- **Valuación del swaption** 8M×10A con Black‑76, usando la volatilidad SABR y la anualidad del swap subyacente.

## Limitaciones y supuestos
- Se ignora el desfase de liquidación t+1/t+2 (DF(0)=1 en la fecha de valuación).
- La matriz de volatilidades VCUB corresponde a tenor 1A; para el swaption a 10A se ancla la volatilidad ATM al dato Bloomberg SWPM.
- El modelo asume curva única OIS (misma curva para proyección y descuento).

## Bibliografía clave
- MexDer (2014) – *Swap Contract New Valuation Methodology*.
- Healy (2020) – *Equivalence between forward rate interpolations...* (arXiv:2005.13890).
- Hagan et al. (2002) – *Managing Smile Risk* (Wilmott).
- Del Castillo Spíndola (2016) – *Superficies de Volatilidad para TIIE* (presentación).
- Bloomberg VCUB y SWPM.

---
**Desarrollado como parte del proyecto de derivados sobre TIIE de Fondeo.**  
Última actualización: {}
""".format(datetime.today().strftime("%d-%m-%Y")))

if st.button("📥 Exportar reporte (CSV de la curva)"):
    # Aquí podrías generar un CSV de la curva OIS
    st.info("Función de exportación por implementar.")