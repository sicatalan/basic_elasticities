import os
import math
from time import perf_counter
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
try:
    import altair as alt
    _ALT_OK = True
    _ALT_ERR = ""
except Exception as _e:
    alt = None  # type: ignore
    _ALT_OK = False
    _ALT_ERR = str(_e)

# Seaborn como alternativa prioritaria frente a Matplotlib puro
try:
    import seaborn as sns
    _SNS_OK = True
except Exception:
    sns = None  # type: ignore
    _SNS_OK = False


# ----------------------------
# Page
# ----------------------------
st.set_page_config(page_title="Elasticidades por SKU", layout="wide")
st.title("Basic Elasticities")
st.caption("Curvas log-log por SKU y precio objetivo con tablas.")


# ----------------------------
# IO helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def _read_csv_cached(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def _safe_read_csv(path: str) -> Tuple[pd.DataFrame, bool]:
    if os.path.exists(path):
        try:
            mtime = os.path.getmtime(path)
            return _read_csv_cached(path, mtime), True
        except Exception:
            return pd.read_csv(path), True
    return pd.DataFrame(), False


def _ensure_logs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "kilos" not in df.columns:
        raise KeyError("Falta la columna 'kilos' en el DataFrame")
    s_q = pd.to_numeric(df["kilos"], errors="coerce")
    s_v = pd.to_numeric(df.get("venta"), errors="coerce") if "venta" in df.columns else pd.Series([np.nan] * len(df))
    if "precio_promedio" in df.columns:
        s_p = pd.to_numeric(df["precio_promedio"], errors="coerce")
    else:
        s_p = pd.Series([np.nan] * len(df))
    precio_calc = np.where((s_v > 0) & (s_q > 0), s_v / s_q, s_p)
    df["precio_promedio_calc"] = precio_calc
    if "costo" in df.columns:
        s_c = pd.to_numeric(df["costo"], errors="coerce")
        df["costo_por_kilo"] = np.where(s_q > 0, s_c / s_q, np.nan)
    df["ln_precio_promedio"] = np.where(precio_calc > 0, np.log(precio_calc), np.nan)
    df["ln_kilos"] = np.where(s_q > 0, np.log(s_q), np.nan)
    return df


def style_table(df: pd.DataFrame, money_cols: List[str], number_cols: Optional[List[str]] = None, money_decimals: int = 0, num_decimals: int = 2):
    number_cols = number_cols or []
    fmt_map = {}
    for c in money_cols:
        if c in df.columns:
            fmt_map[c] = (lambda x: "" if pd.isna(x) else (f"${x:,.{money_decimals}f}"))
    for c in number_cols:
        if c in df.columns and c not in fmt_map:
            fmt_map[c] = (lambda x: "" if pd.isna(x) else (f"{x:,.{num_decimals}f}"))
    try:
        return df.style.format(fmt_map)
    except Exception:
        d2 = df.copy()
        for c, f in fmt_map.items():
            d2[c] = d2[c].apply(lambda x: "" if pd.isna(x) else (f(x)))
        return d2


# ----------------------------
# Elasticidades (global cache por archivo)
# ----------------------------
def compute_elasticidades_core(df: pd.DataFrame) -> pd.DataFrame:
    required = ["sku", "ln_precio_promedio", "ln_kilos"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Falta la columna '{c}' en el DataFrame")
    work = df[required].replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        return pd.DataFrame(columns=["sku", "n_puntos", "beta", "alpha", "r2"])  # vacio
    def fit_group(g: pd.DataFrame) -> pd.Series:
        x = g["ln_precio_promedio"].to_numpy(float)
        y = g["ln_kilos"].to_numpy(float)
        n = x.size
        if n < 2 or float(np.var(x)) == 0.0:
            return pd.Series({"alpha": np.nan, "beta": np.nan, "n_puntos": int(n), "r2": np.nan})
        beta, alpha = np.polyfit(x, y, 1)
        r = np.corrcoef(x, y)[0, 1]
        return pd.Series({"alpha": float(alpha), "beta": float(beta), "n_puntos": int(n), "r2": float(r**2)})
    res = work.groupby("sku", as_index=False, sort=False).apply(fit_group).reset_index(drop=True)
    if "nombre_producto" in df.columns:
        res = res.merge(df[["sku", "nombre_producto"]].drop_duplicates("sku"), on="sku", how="left")
    cols = ["sku"] + (["nombre_producto"] if "nombre_producto" in res.columns else []) + ["n_puntos", "beta", "alpha", "r2"]
    return res[cols].drop_duplicates()


@st.cache_data(show_spinner=False)
def compute_elasticidades(data_file: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(data_file)
    df = _ensure_logs(df)
    return compute_elasticidades_core(df)


# ----------------------------
# Label helpers
# ----------------------------
def get_labels_and_keys(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    if "nombre_producto" in df.columns:
        mp = df[["sku", "nombre_producto"]].astype({"sku": str}).drop_duplicates("sku")
        skus = mp["sku"].tolist()
        names = mp["nombre_producto"].fillna("").astype(str).tolist()
        labels = [f"{s} - {n}" if n else s for s, n in zip(skus, names)]
        return labels, skus
    skus = sorted(df["sku"].astype(str).unique().tolist())
    return skus, skus


def parse_sku_from_label(texto: str, labels: List[str], skus: List[str]) -> Optional[str]:
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


# ----------------------------
# Aggregation and local fit
# ----------------------------
def points_by_week_for_sku(df: pd.DataFrame, sku: str) -> pd.DataFrame:
    d = df[df["sku"].astype(str) == str(sku)].copy()
    if d.empty:
        return d
    if "semana" not in d.columns:
        return _ensure_logs(d[[c for c in ["precio_promedio", "kilos", "venta", "costo"] if c in d.columns]])
    try:
        d["semana"] = pd.to_datetime(d["semana"], errors="coerce")
    except Exception:
        pass
    d["_kilos"] = pd.to_numeric(d.get("kilos"), errors="coerce")
    d["_precio"] = pd.to_numeric(d.get("precio_promedio"), errors="coerce")
    if "venta" in d.columns:
        d["_venta"] = pd.to_numeric(d.get("venta"), errors="coerce")
    if "costo" in d.columns:
        d["_costo"] = pd.to_numeric(d.get("costo"), errors="coerce")
    d["_pxq"] = d["_precio"] * d["_kilos"]
    grp = d.groupby("semana", as_index=False)
    kilos_sum = grp["_kilos"].sum().rename(columns={"_kilos": "kilos"})
    pxq_sum = grp["_pxq"].sum().rename(columns={"_pxq": "_pxq"})
    precio_mean = grp["_precio"].mean().rename(columns={"_precio": "_precio_mean"})
    out = kilos_sum.merge(pxq_sum, on="semana").merge(precio_mean, on="semana")
    if "_venta" in d.columns:
        venta_sum = grp["_venta"].sum().rename(columns={"_venta": "venta"})
        out = out.merge(venta_sum, on="semana", how="left")
        out["precio_promedio"] = np.where(out["kilos"] > 0, out["venta"] / out["kilos"], out["_precio_mean"])
    else:
        out["precio_promedio"] = np.where(out["kilos"] > 0, out["_pxq"] / out["kilos"], out["_precio_mean"])
    if "_costo" in d.columns:
        costo_sum = grp["_costo"].sum().rename(columns={"_costo": "costo"})
        out = out.merge(costo_sum, on="semana", how="left")
        out["costo_por_kilo"] = np.where(out["kilos"] > 0, out["costo"] / out["kilos"], np.nan)
    order_cols = ["semana", "precio_promedio", "kilos"]
    for c in ["venta", "costo", "costo_por_kilo"]:
        if c in out.columns:
            order_cols.append(c)
    out = out[order_cols]
    out = _ensure_logs(out)
    return out.sort_values("semana")


def fit_from_points(d_points: pd.DataFrame) -> Tuple[float, float, float, int]:
    datos = d_points[["ln_precio_promedio", "ln_kilos"]].replace([np.inf, -np.inf], np.nan).dropna()
    n = int(datos.shape[0])
    if n < 2:
        return (np.nan, np.nan, np.nan, n)
    x = datos["ln_precio_promedio"].to_numpy(float)
    y = datos["ln_kilos"].to_numpy(float)
    if float(np.var(x)) == 0.0:
        return (np.nan, np.nan, np.nan, n)
    beta, alpha = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    return (float(alpha), float(beta), float(r ** 2), n)


def last_10_by_date(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    if "semana" in d.columns:
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


def section_header(text: str):
    st.divider()
    st.markdown(
        f"""
        <div style='margin-top:2px; padding-top:6px; font-weight:700; font-size:1.1rem;'>
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def zone_summary_for_sku(df: pd.DataFrame, sku: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Resumen por zona_ventas para un SKU en un rango de fechas.
    Retorna columnas: zona_ventas, kilos, venta, costo, precio_por_kilo, costo_por_kilo, contribucion_por_kilo.
    """
    if "zona_ventas" not in df.columns or "semana" not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    try:
        d["semana"] = pd.to_datetime(d["semana"], errors="coerce")
    except Exception:
        pass
    d = d[(d["sku"].astype(str) == str(sku)) & (d["semana"].between(start, end))]
    if d.empty:
        return pd.DataFrame()
    # Asegura numéricos
    d["_kilos"] = pd.to_numeric(d.get("kilos"), errors="coerce")
    d["_venta"] = pd.to_numeric(d.get("venta"), errors="coerce") if "venta" in d.columns else np.nan
    d["_costo"] = pd.to_numeric(d.get("costo"), errors="coerce") if "costo" in d.columns else np.nan

    grp = d.groupby("zona_ventas", as_index=False)
    k = grp["_kilos"].sum().rename(columns={"_kilos": "kilos"})
    out = k
    if "venta" in d.columns:
        v = grp["_venta"].sum().rename(columns={"_venta": "venta"})
        out = out.merge(v, on="zona_ventas", how="left")
    if "costo" in d.columns:
        c = grp["_costo"].sum().rename(columns={"_costo": "costo"})
        out = out.merge(c, on="zona_ventas", how="left")

    # Derivados por kilo
    if "venta" in out.columns:
        out["precio_por_kilo"] = np.where(out["kilos"] > 0, out.get("venta", np.nan) / out["kilos"], np.nan)
    if "costo" in out.columns:
        out["costo_por_kilo"] = np.where(out["kilos"] > 0, out.get("costo", np.nan) / out["kilos"], np.nan)
    if "precio_por_kilo" in out.columns and "costo_por_kilo" in out.columns:
        out["contribucion_por_kilo"] = out["precio_por_kilo"] - out["costo_por_kilo"]
    return out.sort_values("kilos", ascending=False)


# ----------------------------
# Load data
# ----------------------------
DATA_FILE = "output_price.csv"
ELAS_FILE = "elasticidades_loglog_por_sku.csv"

with st.spinner("Cargando datos..."):
    df_raw, df_exists = _safe_read_csv(DATA_FILE)
if not df_exists:
    st.error(f"No se encontro el archivo '{DATA_FILE}'. Colocalo en la carpeta de la app.")
    st.stop()

df = _ensure_logs(df_raw)

with st.spinner("Calculando elasticidades por SKU..."):
    t0 = perf_counter()
    mtime = os.path.getmtime(DATA_FILE)
    res = compute_elasticidades(DATA_FILE, mtime)
    _elapsed = perf_counter() - t0

try:
    n_rows = int(df.shape[0])
    n_skus = int(df["sku"].nunique()) if "sku" in df.columns else 0
    st.info(f"Filas cargadas: {n_rows:,} | SKUs: {n_skus:,}")
    try:
        n_fit = int(res["beta"].notna().sum()) if "beta" in res.columns else 0
        st.caption(f"Elasticidades calculadas en {_elapsed:.2f} s | SKUs con parametros: {n_fit:,}")
    except Exception:
        pass
except Exception:
    pass


# ----------------------------
# Tabs
# ----------------------------
tab_curvas, tab_precio = st.tabs(["Curvas por SKU", "Precio objetivo y tablas"])


# ----------------------------
# Tab: Curvas
# ----------------------------
with tab_curvas:
    st.subheader("Curva ln(Q) vs ln(P)")
    labels, skus = get_labels_and_keys(df)
    if not labels:
        st.info("No hay SKUs disponibles en los datos cargados.")
        sku = None
    else:
        st.caption(f"SKUs disponibles: {len(labels):,}")
        sel_label = st.selectbox("SKU", options=labels, index=0)
        sku = parse_sku_from_label(sel_label, labels, skus)

    gen_curve = st.button("Generar curva", disabled=not bool(sku), key="btn_gen_curve")
    if gen_curve and sku:
        st.session_state["curve_do"] = True
        st.session_state["curve_sku"] = sku

    if not st.session_state.get("curve_do"):
        st.info("Elige un SKU y presiona 'Generar curva'.")
    else:
        sku_to_plot = st.session_state.get("curve_sku")
        if not sku_to_plot:
            st.warning("Selecciona un SKU valido.")
        else:
            with st.spinner("Generando grafica..."):
                d_points = points_by_week_for_sku(df, sku_to_plot)
                datos = d_points[["ln_precio_promedio", "ln_kilos"]].replace([np.inf, -np.inf], np.nan).dropna()
                alpha, beta, r2, n_pts = fit_from_points(d_points)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Puntos", int(n_pts))
                c2.metric("Alpha", f"{alpha:.3f}" if not np.isnan(alpha) else "-")
                c3.metric("Beta", f"{beta:.3f}" if not np.isnan(beta) else "-")
                c4.metric("R2", f"{r2:.3f}" if not np.isnan(r2) else "-")

                fig, ax = plt.subplots(figsize=(6.5, 4.5))
                if not datos.empty:
                    x = datos["ln_precio_promedio"].to_numpy()
                    y = datos["ln_kilos"].to_numpy()
                    ax.scatter(x, y, alpha=0.75, label="Datos (ln P vs ln Q)")
                    if x.size > 0 and not (np.isnan(alpha) or np.isnan(beta)):
                        xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                        yy = alpha + beta * xx
                        ax.plot(xx, yy, color="red", label=f"y = {alpha:.2f} + {beta:.2f} x")
                ax.set_xlabel("ln(P) precio")
                ax.set_ylabel("ln(Q) kilos")
                ax.grid(True, alpha=0.25)
                ax.legend()
                st.pyplot(fig, clear_figure=True, use_container_width=False)

                section_header("Datos que alimentan la gráfica")
                d_all = d_points.copy()
                cols_base = [c for c in ["semana", "precio_promedio", "kilos", "venta", "costo", "costo_por_kilo"] if c in d_all.columns]
                d_view = d_all[cols_base]
                copt1, copt2 = st.columns([1, 1])
                mode = copt1.radio("Muestra a mostrar", options=["Ultimos por fecha", "Muestra aleatoria"], horizontal=True, key="curve_tbl_mode")
                max_view = int(min(500, max(10, len(d_view))))
                n_show = copt2.slider("Filas", min_value=10, max_value=max_view, value=min(100, max_view), step=10, key="curve_tbl_n")
                if mode == "Ultimos por fecha" and "semana" in d_view.columns:
                    try:
                        d_view["semana"] = pd.to_datetime(d_view["semana"], errors="coerce")
                    except Exception:
                        pass
                    d_view = d_view.sort_values("semana").tail(n_show)
                else:
                    if len(d_view) > n_show:
                        d_view = d_view.sample(n_show, random_state=42)
                money_cols = [c for c in ["precio_promedio", "venta", "costo", "costo_por_kilo"] if c in d_view.columns]
                number_cols = [c for c in ["kilos"] if c in d_view.columns]
                st.dataframe(style_table(d_view.reset_index(drop=True), money_cols, number_cols), use_container_width=True)
                try:
                    csv_bytes = d_view.to_csv(index=False).encode("utf-8")
                    st.download_button("Descargar muestra CSV", data=csv_bytes, file_name=f"puntos_sku_{sku_to_plot}_muestra.csv", mime="text/csv")
                except Exception:
                    pass
            st.session_state["curve_do"] = False


# ----------------------------
# Tab: Precio objetivo
# ----------------------------
with tab_precio:
    st.subheader("Precio sugerido para un volumen objetivo")
    labels2, skus2 = get_labels_and_keys(df)
    if not labels2:
        st.info("No hay SKUs disponibles en los datos cargados.")
        sku2 = None
    else:
        st.caption(f"SKUs disponibles: {len(labels2):,}")
        sel_label2 = st.selectbox("SKU", options=labels2, index=0, key="sku_tab2")
        sku2 = parse_sku_from_label(sel_label2, labels2, skus2)

    col_left, col_right = st.columns([1, 1])
    with col_left:
        vol = st.number_input("Volumen objetivo Q (kilos)", min_value=0.0, value=10.0, step=1.0)
        calc_btn = st.button("Calcular", key="btn_calc", disabled=not bool(sku2) or vol is None or vol <= 0)
        if calc_btn and sku2 and vol and vol > 0:
            st.session_state["calc_do"] = True
            st.session_state["calc_sku"] = sku2
            st.session_state["calc_vol"] = float(vol)

    if not st.session_state.get("calc_do"):
        st.info("Elige SKU y volumen, luego presiona 'Calcular'.")
    else:
        st.session_state["calc_do"] = False
        sku_run = st.session_state.get("calc_sku")
        vol_run = st.session_state.get("calc_vol")
        if not sku_run or not vol_run or vol_run <= 0:
            st.warning("Parametros invalidos para calcular.")
        else:
            with st.spinner("Calculando tablas y precio..."):
                d_points2 = points_by_week_for_sku(df, sku_run)
                cols_hist = [c for c in ["semana", "precio_promedio", "kilos", "venta", "costo", "costo_por_kilo"] if c in d_points2.columns]
                d10 = last_10_by_date(d_points2[cols_hist])

                with col_left:
                    st.markdown("Historial (ultimos 10 por fecha)")
                    money_cols_hist = [c for c in ["precio_promedio", "venta", "costo", "costo_por_kilo"] if c in d10.columns]
                    number_cols_hist = [c for c in ["kilos"] if c in d10.columns]
                    st.dataframe(style_table(d10.reset_index(drop=True), money_cols_hist, number_cols_hist), use_container_width=True)

                with col_right:
                    st.markdown("Parametros y estadisticos")
                    a2, b2, r22, n2 = fit_from_points(d_points2)
                    base = pd.DataFrame([{ "sku": sku_run, "n_puntos": n2, "alpha": a2, "beta": b2, "r2": r22 }])
                    pm, ps, pcv = series_stats(d_points2["precio_promedio"]) if "precio_promedio" in d_points2.columns else (np.nan, np.nan, np.nan)
                    qm, qs, qcv = series_stats(d_points2["kilos"]) if "kilos" in d_points2.columns else (np.nan, np.nan, np.nan)
                    ckm, cks, ckcv = series_stats(d_points2["costo_por_kilo"]) if "costo_por_kilo" in d_points2.columns else (np.nan, np.nan, np.nan)
                    if ("precio_promedio" in d_points2.columns) and ("costo_por_kilo" in d_points2.columns):
                        contrib_series = pd.to_numeric(d_points2["precio_promedio"], errors="coerce") - pd.to_numeric(d_points2["costo_por_kilo"], errors="coerce")
                        cm, cs, ccv = series_stats(contrib_series)
                    else:
                        cm, cs, ccv = (np.nan, np.nan, np.nan)
                    base["precio_mean"] = pm
                    base["precio_std"] = ps
                    base["precio_cv"] = pcv
                    base["kilos_mean"] = qm
                    base["kilos_std"] = qs
                    base["kilos_cv"] = qcv
                    base["costo_kilo_mean"] = ckm
                    base["costo_kilo_std"] = cks
                    base["costo_kilo_cv"] = ckcv
                    base["contrib_kilo_mean"] = cm
                    base["contrib_kilo_std"] = cs
                    base["contrib_kilo_cv"] = ccv
                    money_cols_stats = [c for c in ["precio_mean","precio_std","costo_kilo_mean","costo_kilo_std","contrib_kilo_mean","contrib_kilo_std"] if c in base.columns]
                    number_cols_stats = [c for c in ["n_puntos","kilos_mean","kilos_std","kilos_cv","precio_cv","costo_kilo_cv","contrib_kilo_cv","r2"] if c in base.columns]
                    st.dataframe(style_table(base.reset_index(drop=True), money_cols_stats, number_cols_stats), use_container_width=True)

                section_header("Precio para Q objetivo")
                if pd.isna(base["beta"].iloc[0]):
                    st.info("No es posible calcular precio: beta no disponible.")
                else:
                    beta2 = float(base["beta"].iloc[0])
                    alpha2 = float(base["alpha"].iloc[0])
                    if beta2 == 0 or math.isnan(beta2) or math.isnan(alpha2):
                        st.info("No es posible calcular precio (beta nula).")
                    else:
                        lnP = (math.log(vol_run) - alpha2) / beta2
                        precio = float(math.exp(lnP))
                        if "costo" in d_points2.columns:
                            tot_k = float(d_points2["kilos"].sum()) if not pd.isna(d_points2["kilos"].sum()) else np.nan
                            tot_c = float(d_points2["costo"].sum()) if not pd.isna(d_points2["costo"].sum()) else np.nan
                            cpk = (tot_c / tot_k) if (tot_k and tot_k > 0) else np.nan
                        else:
                            cpk = np.nan
                        contrib = precio - cpk if (not math.isnan(precio) and not math.isnan(cpk)) else np.nan
                        out = pd.DataFrame([{ "sku": sku_run, "Q_objetivo": vol_run, "precio_sugerido": precio, "costo_por_kilo": cpk, "contribucion_por_kilo": contrib }])
                        st.dataframe(style_table(out, ["precio_sugerido", "costo_por_kilo", "contribucion_por_kilo"], ["Q_objetivo"], money_decimals=0, num_decimals=0), use_container_width=True)

                # ----------------------------
                # Contexto por zona de venta (último mes o rango elegido)
                # ----------------------------
                section_header("Contexto por zona de venta")
                if "zona_ventas" not in df.columns or "semana" not in df.columns:
                    st.info("No hay columnas 'zona_ventas' y/o 'semana' para generar el resumen.")
                else:
                    # Rango por defecto: último mes presente en datos del SKU
                    try:
                        all_dates = pd.to_datetime(df.loc[df["sku"].astype(str) == str(sku_run), "semana"], errors="coerce")
                        max_date = pd.to_datetime(all_dates.max()) if not all_dates.empty else pd.Timestamp.today()
                        min_date = pd.to_datetime(all_dates.min()) if not all_dates.empty else max_date - pd.Timedelta(days=30)
                    except Exception:
                        max_date = pd.Timestamp.today()
                        min_date = max_date - pd.Timedelta(days=30)

                    default_start = max_date - pd.Timedelta(days=30)
                    start_end = st.date_input("Rango de fechas", (default_start.date(), max_date.date()))
                    if isinstance(start_end, tuple) and len(start_end) == 2:
                        start_dt = pd.to_datetime(start_end[0])
                        end_dt = pd.to_datetime(start_end[1])
                    else:
                        start_dt = default_start
                        end_dt = max_date

                    zsum = zone_summary_for_sku(df, sku_run, start_dt, end_dt)
                    if zsum.empty:
                        st.info("Sin datos para ese rango de fechas.")
                    else:
                        money_cols_zone = [c for c in ["precio_por_kilo", "costo_por_kilo", "contribucion_por_kilo", "venta", "costo"] if c in zsum.columns]
                        number_cols_zone = [c for c in ["kilos"] if c in zsum.columns]
                        st.dataframe(style_table(zsum, money_cols_zone, number_cols_zone), use_container_width=True)

                        # Visualización: cuadrados por zona (tamaño=kilos, color=precio x kilo)
                        vis_df = zsum.copy()
                        vis_df["kilos"] = pd.to_numeric(vis_df["kilos"], errors="coerce")
                        vis_df["precio_por_kilo"] = pd.to_numeric(vis_df.get("precio_por_kilo"), errors="coerce")
                        if _ALT_OK and alt is not None and not vis_df.empty and "precio_por_kilo" in vis_df.columns:
                            color_scale = alt.Scale(domain=[float(vis_df["precio_por_kilo"].min()), float(vis_df["precio_por_kilo"].max())], range=["red", "darkgreen"]) 
                            chart = (
                                alt.Chart(vis_df)
                                .mark_square()
                                .encode(
                                    x=alt.X("zona_ventas:N", sort="-y", title="Zona"),
                                    y=alt.value(0),
                                    size=alt.Size("kilos:Q", title="Kilos", scale=alt.Scale(type="sqrt", range=[10, 1600])),
                                    color=alt.Color("precio_por_kilo:Q", title="Precio/kilo", scale=color_scale),
                                    tooltip=["zona_ventas", alt.Tooltip("kilos:Q", format=",.0f"), alt.Tooltip("precio_por_kilo:Q", format="$,.0f"), alt.Tooltip("costo_por_kilo:Q", format="$,.0f"), alt.Tooltip("contribucion_por_kilo:Q", format="$,.0f")],
                                )
                                .properties(height=200, use_container_width=True)
                            )
                            st.altair_chart(chart, use_container_width=True)
                        elif _SNS_OK and sns is not None and not vis_df.empty and "precio_por_kilo" in vis_df.columns:
                            try:
                                import matplotlib.cm as cm
                                import matplotlib.colors as mcolors
                                from matplotlib.ticker import FuncFormatter
                                # Orden por kilos
                                vis_df2 = vis_df.sort_values("kilos", ascending=False).reset_index(drop=True)
                                pmin, pmax = float(vis_df2["precio_por_kilo"].min()), float(vis_df2["precio_por_kilo"].max())
                                norm = mcolors.Normalize(vmin=pmin, vmax=pmax)
                                cmap = cm.get_cmap('RdYlGn')
                                fig2, ax2 = plt.subplots(figsize=(max(8, min(14, 0.5 * len(vis_df2) + 4)), 5))
                                sns.scatterplot(
                                    data=vis_df2,
                                    x="zona_ventas",
                                    y="precio_por_kilo",
                                    size="kilos",
                                    sizes=(50, 1800),
                                    hue="precio_por_kilo",
                                    palette=cmap,
                                    hue_norm=norm,
                                    marker="s",
                                    edgecolor=None,
                                    alpha=0.85,
                                    legend=False,
                                    ax=ax2,
                                )
                                ax2.set_xlabel("Zona")
                                ax2.set_ylabel("Precio/kilo")
                                ax2.tick_params(axis='x', rotation=45)
                                ax2.grid(True, axis='y', alpha=0.2)
                                ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"${x:,.0f}"))
                                # Colorbar manual
                                sm = cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
                                cbar = fig2.colorbar(sm, ax=ax2, pad=0.01)
                                cbar.set_label("Precio/kilo")
                                st.pyplot(fig2, clear_figure=True, use_container_width=True)
                                if not _ALT_OK:
                                    st.caption(f"Nota: Altair no disponible ({_ALT_ERR}). Usando Seaborn.")
                            except Exception:
                                st.info("No se pudo renderizar con Seaborn. Usando fallback simple.")
                                # Fallback Matplotlib: burbuja horizontal por zona
                                try:
                                    import matplotlib.cm as cm
                                    import matplotlib.colors as mcolors
                                    fig3, ax3 = plt.subplots(figsize=(8, 3 + 0.2 * len(vis_df)))
                                    zones = vis_df["zona_ventas"].astype(str).tolist()
                                    y = np.arange(len(zones))
                                    sizes = np.sqrt(np.maximum(vis_df["kilos"].fillna(0), 0)) * 10.0
                                    pmin, pmax = float(vis_df["precio_por_kilo"].min()), float(vis_df["precio_por_kilo"].max())
                                    norm = mcolors.Normalize(vmin=pmin, vmax=pmax)
                                    cmap = cm.get_cmap('RdYlGn')
                                    colors = cmap(norm(vis_df["precio_por_kilo"].fillna(pmin)))
                                    ax3.scatter(vis_df["precio_por_kilo"], y, s=sizes, c=colors, alpha=0.8)
                                    ax3.set_yticks(y)
                                    ax3.set_yticklabels(zones)
                                    ax3.set_xlabel("Precio/kilo")
                                    ax3.set_title("Precio por kilo y volumen por zona (fallback)")
                                    ax3.grid(True, axis='x', alpha=0.2)
                                    st.pyplot(fig3, clear_figure=True, use_container_width=True)
                                except Exception:
                                    st.info("No se pudo renderizar la visualización alternativa.")
                        else:
                            # Fallback Matplotlib si no hay Altair ni Seaborn
                            try:
                                import matplotlib.cm as cm
                                import matplotlib.colors as mcolors
                                fig2, ax2 = plt.subplots(figsize=(8, 3 + 0.2 * len(vis_df)))
                                zones = vis_df["zona_ventas"].astype(str).tolist()
                                y = np.arange(len(zones))
                                sizes = np.sqrt(np.maximum(vis_df["kilos"].fillna(0), 0)) * 10.0
                                pmin, pmax = float(vis_df["precio_por_kilo"].min()), float(vis_df["precio_por_kilo"].max())
                                norm = mcolors.Normalize(vmin=pmin, vmax=pmax)
                                cmap = cm.get_cmap('RdYlGn')
                                colors = cmap(norm(vis_df["precio_por_kilo"].fillna(pmin)))
                                ax2.scatter(vis_df["precio_por_kilo"], y, s=sizes, c=colors, alpha=0.8)
                                ax2.set_yticks(y)
                                ax2.set_yticklabels(zones)
                                ax2.set_xlabel("Precio/kilo")
                                ax2.set_title("Precio por kilo y volumen por zona (fallback)")
                                ax2.grid(True, axis='x', alpha=0.2)
                                st.pyplot(fig2, clear_figure=True, use_container_width=True)
                                if not _ALT_OK:
                                    st.caption(f"Nota: Altair no disponible ({_ALT_ERR}). Mostrando gráfico alternativo.")
                            except Exception:
                                st.info("No se pudo renderizar la visualización alternativa.")


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Datos")
    st.caption(f"Archivo datos: {DATA_FILE}")
    st.caption("Elasticidades: calculadas en memoria (no se usa CSV precalculado)")
    if st.button("Recalcular elasticidades"):
        compute_elasticidades.clear()
        st.toast("Recalculando elasticidades…")
        st.rerun()
    if st.button("Exportar elasticidades a CSV"):
        try:
            res.to_csv(ELAS_FILE, index=False)
            st.success(f"Exportado a '{ELAS_FILE}'")
        except Exception as e:
            st.error(f"No se pudo exportar: {e}")
