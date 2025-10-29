import os
import math
from typing import Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ---------------------------------------------
# Configuración general de la app
# ---------------------------------------------
st.set_page_config(page_title="Elasticidades por SKU", layout="wide")


# ---------------------------------------------
# Utilidades de carga y cache
# ---------------------------------------------
@st.cache_data(show_spinner=False)
def _read_csv_cached(path: str, mtime: float) -> pd.DataFrame:
    # La clave del cache incluye el mtime para invalidar cuando cambie el archivo
    return pd.read_csv(path)


def _safe_read_csv(path: str) -> Tuple[pd.DataFrame, bool]:
    """Lee CSV si existe. Retorna (df, exists)."""
    if os.path.exists(path):
        try:
            mtime = os.path.getmtime(path)
            return _read_csv_cached(path, mtime), True
        except Exception:
            # fallback sin cache si algo raro pasa
            return pd.read_csv(path), True
    return pd.DataFrame(), False


def _ensure_logs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Columnas esperadas en el notebook
    if "precio_promedio" not in df.columns or "kilos" not in df.columns:
        raise KeyError("Faltan columnas requeridas: 'precio_promedio' y/o 'kilos'")
    s_p = pd.to_numeric(df["precio_promedio"], errors="coerce")
    s_q = pd.to_numeric(df["kilos"], errors="coerce")
    df["ln_precio_promedio"] = np.where(s_p > 0, np.log(s_p), np.nan)
    df["ln_kilos"] = np.where(s_q > 0, np.log(s_q), np.nan)
    return df


@st.cache_data(show_spinner=False)
def compute_elasticidades(df: pd.DataFrame) -> pd.DataFrame:
    required = ["sku", "ln_precio_promedio", "ln_kilos"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Falta la columna '{c}' en el DataFrame")

    work = df[required].replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        return pd.DataFrame(columns=["sku", "n_puntos", "beta", "alpha", "r2"])  # vacío

    def fit_group(g: pd.DataFrame) -> pd.Series:
        x = g["ln_precio_promedio"].to_numpy(float)
        y = g["ln_kilos"].to_numpy(float)
        n = x.size
        if n < 2 or float(np.var(x)) == 0.0:
            return pd.Series({"alpha": np.nan, "beta": np.nan, "n_puntos": int(n), "r2": np.nan})
        # polyfit devuelve [pendiente, intercepto]
        beta, alpha = np.polyfit(x, y, 1)
        r = np.corrcoef(x, y)[0, 1]
        return pd.Series({
            "alpha": float(alpha),
            "beta": float(beta),
            "n_puntos": int(n),
            "r2": float(r ** 2)
        })

    res = work.groupby("sku", as_index=False, sort=False).apply(fit_group).reset_index(drop=True)

    # Adjunta nombre de producto si existe
    if "nombre_producto" in df.columns:
        res = res.merge(
            df[["sku", "nombre_producto"]].drop_duplicates("sku"),
            on="sku",
            how="left",
        )

    # Ordena columnas
    cols: List[str] = ["sku"]
    if "nombre_producto" in res.columns:
        cols.append("nombre_producto")
    cols += ["n_puntos", "beta", "alpha", "r2"]
    res = res[cols].drop_duplicates()
    return res


def get_labels_and_keys(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    skus = sorted(df["sku"].astype(str).unique().tolist())
    labels: List[str] = []
    for s in skus:
        if "nombre_producto" in df.columns:
            nm = df.loc[df["sku"].astype(str) == s, "nombre_producto"]
            lab = f"{s} - {nm.iloc[0]}" if not nm.empty else s
        else:
            lab = s
        labels.append(lab)
    return labels, skus


def parse_sku_from_label(texto: str, labels: List[str], skus: List[str]) -> str | None:
    if not texto:
        return None
    t = str(texto).strip()
    if t in skus:
        return t
    if " - " in t and t.split(" - ", 1)[0] in skus:
        return t.split(" - ", 1)[0]
    tt = t.lower()
    for lab in labels:
        if tt and tt in lab.lower():
            return lab.split(" - ", 1)[0]
    return None


def last_10_by_date(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    if "semana" in d.columns:
        with pd.option_context("mode.chained_assignment", None):
            try:
                d["semana"] = pd.to_datetime(d["semana"], errors="coerce")
            except Exception:
                pass
        d = d.sort_values("semana").tail(10)
    return d


def series_stats(vals: pd.Series) -> Tuple[float, float, float]:
    vals = pd.to_numeric(vals, errors="coerce")
    mean = float(vals.mean()) if not math.isnan(float(vals.mean())) else np.nan
    std = float(vals.std(ddof=1)) if not math.isnan(float(vals.std(ddof=1))) else np.nan
    cv = float(std / mean) if (mean not in (0, np.nan) and not math.isnan(mean) and not math.isnan(std)) else np.nan
    return mean, std, cv


# ---------------------------------------------
# Carga de datos
# ---------------------------------------------
DATA_FILE = "output_price.csv"
ELAS_FILE = "elasticidades_loglog_por_sku.csv"

df_raw, df_exists = _safe_read_csv(DATA_FILE)
if not df_exists:
    st.error(f"No se encontró el archivo '{DATA_FILE}'. Colócalo en la carpeta de la app.")
    st.stop()

df = _ensure_logs(df_raw)

# Carga elasticidades precalculadas si existen, si no las computa
elas_df_file, elas_exists = _safe_read_csv(ELAS_FILE)
if elas_exists and not elas_df_file.empty and set(["sku", "beta", "alpha"]).issubset(elas_df_file.columns):
    # Asegura tipos/columnas
    res = elas_df_file.copy()
    if "nombre_producto" not in res.columns and "nombre_producto" in df.columns:
        res = res.merge(df[["sku", "nombre_producto"]].drop_duplicates("sku"), on="sku", how="left")
else:
    with st.spinner("Calculando elasticidades por SKU..."):
        res = compute_elasticidades(df)


# ---------------------------------------------
# UI principal
# ---------------------------------------------
st.title("Basic Elasticities")
st.caption("Visualización de curvas log-log por SKU y cálculo de precio para un volumen objetivo.")

tab_curvas, tab_precio = st.tabs(["Curvas por SKU", "Precio objetivo y tablas"])


# ---------------------------------------------
# Tab 1: Curvas por SKU
# ---------------------------------------------
with tab_curvas:
    st.subheader("Curva ln(Q) vs ln(P)")

    labels, skus = get_labels_and_keys(df)
    sel_label = st.selectbox("SKU", options=labels, index=0 if labels else None, placeholder="Escribe para buscar...")
    sku = parse_sku_from_label(sel_label, labels, skus)

    if not sku:
        st.info("Selecciona un SKU para visualizar.")
    else:
        datos = (
            df[df["sku"].astype(str) == sku][["ln_precio_promedio", "ln_kilos"]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        fila = res[res["sku"].astype(str) == sku]
        alpha = float(fila["alpha"].iloc[0]) if not fila.empty and pd.notna(fila["alpha"].iloc[0]) else np.nan
        beta = float(fila["beta"].iloc[0]) if not fila.empty and pd.notna(fila["beta"].iloc[0]) else np.nan
        r2 = float(fila["r2"].iloc[0]) if (not fila.empty and "r2" in fila.columns and pd.notna(fila["r2"].iloc[0])) else np.nan

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Puntos", int(datos.shape[0]))
        c2.metric("Alpha", f"{alpha:.3f}" if not np.isnan(alpha) else "—")
        c3.metric("Beta", f"{beta:.3f}" if not np.isnan(beta) else "—")
        c4.metric("R²", f"{r2:.3f}" if not np.isnan(r2) else "—")

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        if not datos.empty:
            x = datos["ln_precio_promedio"].to_numpy()
            y = datos["ln_kilos"].to_numpy()
            ax.scatter(x, y, alpha=0.75, label="Datos (ln P vs ln Q)")
            if x.size > 0 and not (np.isnan(alpha) or np.isnan(beta)):
                xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                yy = alpha + beta * xx
                ax.plot(xx, yy, color="red", label=f"y = {alpha:.2f} + {beta:.2f} x")
        ax.set_xlabel("ln(P) precio_promedio")
        ax.set_ylabel("ln(Q) kilos")
        ax.grid(True, alpha=0.25)
        ax.legend()
        st.pyplot(fig, clear_figure=True, use_container_width=False)


# ---------------------------------------------
# Tab 2: Precio objetivo y tablas
# ---------------------------------------------
with tab_precio:
    st.subheader("Precio sugerido para un volumen objetivo")

    labels2, skus2 = get_labels_and_keys(df)
    sel_label2 = st.selectbox(
        "SKU", options=labels2, index=0 if labels2 else None, key="sku_tab2", placeholder="Escribe para buscar..."
    )
    sku2 = parse_sku_from_label(sel_label2, labels2, skus2)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        vol = st.number_input("Volumen objetivo Q (kilos)", min_value=0.0, value=10.0, step=1.0)

    # Panel de resultados
    if not sku2:
        st.info("Selecciona un SKU para calcular y ver tablas.")
        st.stop()

    # Historial (últimos 10 por fecha)
    dsku = df[df["sku"].astype(str) == sku2][[c for c in ["semana", "precio_promedio", "kilos", "venta", "costo"] if c in df.columns]]
    d10 = last_10_by_date(dsku)

    with col_left:
        st.markdown("Historial (últimos 10 por fecha)")
        st.dataframe(d10.reset_index(drop=True), use_container_width=True)

    # Parámetros + estadísticos
    with col_right:
        st.markdown("Parámetros y estadísticos")
        fila2 = res[res["sku"].astype(str) == sku2]
        if fila2.empty:
            st.warning("No hay parámetros estimados para este SKU.")
            base = pd.DataFrame([{"sku": sku2, "n_puntos": 0, "alpha": np.nan, "beta": np.nan, "r2": np.nan}])
        else:
            base = fila2[[c for c in ["sku", "n_puntos", "alpha", "beta", "r2", "nombre_producto"] if c in fila2.columns]].copy()
        pm, ps, pcv = series_stats(dsku["precio_promedio"]) if "precio_promedio" in dsku.columns else (np.nan, np.nan, np.nan)
        qm, qs, qcv = series_stats(dsku["kilos"]) if "kilos" in dsku.columns else (np.nan, np.nan, np.nan)
        base["precio_mean"] = pm
        base["precio_std"] = ps
        base["precio_cv"] = pcv
        base["kilos_mean"] = qm
        base["kilos_std"] = qs
        base["kilos_cv"] = qcv
        st.dataframe(base.reset_index(drop=True), use_container_width=True)

    st.markdown("Precio para Q objetivo")
    if fila2.empty or pd.isna(base["beta"].iloc[0]):
        st.info("No es posible calcular precio: beta no disponible.")
    else:
        beta2 = float(base["beta"].iloc[0])
        alpha2 = float(base["alpha"].iloc[0])
        if vol is None or vol <= 0 or beta2 == 0 or math.isnan(beta2) or math.isnan(alpha2):
            st.info("No es posible calcular precio (beta nula o volumen inválido).")
        else:
            lnP = (math.log(vol) - alpha2) / beta2
            precio = float(math.exp(lnP))
            out = pd.DataFrame([
                {"sku": sku2, "Q_objetivo": vol, "precio_sugerido": precio}
            ])
            st.dataframe(out, use_container_width=True)


# ---------------------------------------------
# Barra lateral: controles y ayuda
# ---------------------------------------------
with st.sidebar:
    st.header("Datos")
    st.caption(f"Archivo datos: {DATA_FILE}")
    st.caption(f"Archivo elasticidades: {ELAS_FILE}{' (precalculado)' if elas_exists else ''}")

    if st.button("Recalcular elasticidades desde datos"):
        with st.spinner("Recalculando elasticidades desde 'output_price.csv'..."):
            res_new = compute_elasticidades(df)
            st.session_state["_elas_df_override"] = res_new
            st.success("Elasticidades recalculadas. Recarga la página si no se ven reflejadas.")

