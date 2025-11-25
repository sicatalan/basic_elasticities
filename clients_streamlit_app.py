import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple


st.set_page_config(page_title="Curva teorica por cliente", layout="wide", initial_sidebar_state="expanded")

DATA_PATH = "clients_data.csv"


# -------------------------------------------------------------------
# Carga y preparación
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_clients_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def get_aggregated(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate_by_client(df)


def add_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["precio_por_kilo"] = np.where(
        work.get("kilos", 0) > 0,
        work.get("venta", np.nan) / work.get("kilos", np.nan),
        work.get("precio_promedio", np.nan),
    )
    if "costo_por_kilo" not in work.columns:
        work["costo_por_kilo"] = np.where(
            work.get("kilos", 0) > 0, work.get("costo", np.nan) / work.get("kilos", np.nan), np.nan
        )
    return work


def aggregate_by_client(df: pd.DataFrame) -> pd.DataFrame:
    base = add_price_columns(df)
    group_cols = ["jefe_categoria", "categoria", "familia", "sku", "nombre_producto", "nombre_cliente", "zona_ventas"]
    agg = (
        base.groupby(group_cols, as_index=False)
        .agg(
            kilos_total=("kilos", "sum"),
            venta_total=("venta", "sum"),
            costo_total=("costo", "sum"),
            precio_promedio_medio=("precio_promedio", "mean"),
            precio_por_kilo_medio=("precio_por_kilo", "mean"),
            costo_por_kilo_medio=("costo_por_kilo", "mean"),
            n_movimientos=("kilos", "count"),
        )
    )
    agg["precio_real"] = np.where(agg["kilos_total"] > 0, agg["venta_total"] / agg["kilos_total"], np.nan)
    agg["costo_kilo"] = np.where(agg["kilos_total"] > 0, agg["costo_total"] / agg["kilos_total"], agg["costo_por_kilo_medio"])
    return agg


# -------------------------------------------------------------------
# Limpieza y curva monotónica
# -------------------------------------------------------------------
def winsorize_and_filter(df: pd.DataFrame, price_col: str = "precio_real") -> Tuple[pd.DataFrame, Tuple[float, float]]:
    prices = df[price_col].replace([np.inf, -np.inf], np.nan).dropna()
    if prices.empty or prices.shape[0] < 3:
        return df.assign(precio_limpio=df[price_col]), (np.nan, np.nan)
    p5, p95 = np.percentile(prices, [5, 95])
    filtered = df[(df[price_col] >= p5) & (df[price_col] <= p95)].copy()
    if filtered.empty:
        filtered = df.copy()
    filtered["precio_limpio"] = filtered[price_col].clip(p5, p95)
    return filtered, (float(p5), float(p95))


def _pav_isotonic(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    # Pool Adjacent Violators para forzar tendencia no decreciente.
    means = y.astype(float).tolist()
    weights = w.astype(float).tolist()
    i = 0
    while i < len(means) - 1:
        if means[i] > means[i + 1]:
            new_mean = (means[i] * weights[i] + means[i + 1] * weights[i + 1]) / (weights[i] + weights[i + 1])
            new_weight = weights[i] + weights[i + 1]
            means[i] = new_mean
            weights[i] = new_weight
            del means[i + 1]
            del weights[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1
    # Expandir nuevamente por cantidad de elementos por bloque.
    expanded = []
    for mean_val, weight_val in zip(means, weights):
        expanded.extend([mean_val] * int(weight_val))
    # Si se usaron pesos distintos a 1, reasignar respetando longitudes originales.
    if len(expanded) < len(y):
        expanded = np.repeat(means, np.ceil(weights).astype(int))
    return np.array(expanded[: len(y)], dtype=float)


def monotonic_curve(volumes: np.ndarray, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid = ~np.isnan(volumes) & ~np.isnan(prices) & (volumes > 0)
    vols = volumes[valid]
    vals = prices[valid]
    if vols.size == 0:
        return np.array([]), np.array([])
    order = np.argsort(vols)
    v_sorted = vols[order]
    p_sorted = vals[order]
    y_iso = _pav_isotonic(-p_sorted, np.ones_like(p_sorted)) * -1  # Negar para forzar curva decreciente.
    return v_sorted, y_iso


def predict_theoretical_price(vol_sorted: np.ndarray, price_sorted: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    if vol_sorted.size == 0:
        return np.full_like(volumes, np.nan, dtype=float)
    return np.interp(volumes, vol_sorted, price_sorted, left=price_sorted[0], right=price_sorted[-1])


# -------------------------------------------------------------------
# Ajustes y resúmenes
# -------------------------------------------------------------------
def compute_adjusted_curve(df_filtered: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, float, float]:
    if df_filtered.empty:
        return df_filtered, np.array([]), np.array([]), np.nan, np.nan
    # Trabajar sobre copia mínima
    work = df_filtered.copy()
    clean_df, (p5, p95) = winsorize_and_filter(work, price_col="precio_real")
    vol_sorted, price_sorted = monotonic_curve(
        clean_df["kilos_total"].to_numpy(float), clean_df["precio_limpio"].to_numpy(float)
    )
    work["precio_teorico"] = predict_theoretical_price(
        vol_sorted, price_sorted, work["kilos_total"].to_numpy(float)
    )
    work["nuevo_precio"] = np.where(
        work["precio_real"] < work["precio_teorico"], work["precio_teorico"], work["precio_real"]
    )
    work["nuevo_revenue"] = work["nuevo_precio"] * work["kilos_total"]
    work["delta_ingreso"] = work["nuevo_revenue"] - work["venta_total"]
    work["margen_actual"] = work["venta_total"] - work["costo_total"]
    work["nuevo_margen"] = work["nuevo_revenue"] - work["costo_total"]
    work["delta_margen"] = work["nuevo_margen"] - work["margen_actual"]
    work["margen_pct_actual"] = np.where(
        work["costo_total"] > 0,
        100 * (work["venta_total"] - work["costo_total"]) / work["costo_total"],
        np.nan,
    )
    work["margen_pct_nuevo"] = np.where(
        work["costo_total"] > 0,
        100 * (work["nuevo_revenue"] - work["costo_total"]) / work["costo_total"],
        np.nan,
    )
    work["delta_margen_pp"] = work["margen_pct_nuevo"] - work["margen_pct_actual"]
    return work, vol_sorted, price_sorted, p5, p95


def summarize_by(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()
    grp = df.groupby(group_col, as_index=False).agg(
        ingreso_total=("venta_total", "sum"),
        costo_total=("costo_total", "sum"),
        nuevo_ingreso=("nuevo_revenue", "sum"),
        delta_ingreso=("delta_ingreso", "sum"),
    )
    grp["margen_pct_actual"] = np.where(
        grp["costo_total"] > 0, 100 * (grp["ingreso_total"] - grp["costo_total"]) / grp["costo_total"], np.nan
    )
    grp["margen_pct_nuevo"] = np.where(
        grp["costo_total"] > 0, 100 * (grp["nuevo_ingreso"] - grp["costo_total"]) / grp["costo_total"], np.nan
    )
    grp["delta_margen_pp"] = grp["margen_pct_nuevo"] - grp["margen_pct_actual"]
    return grp


@st.cache_data(show_spinner=False)
def compute_adjusted_all_cached(base_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el ajuste por SKU (no mezclado) para reutilizar en resúmenes y top 50."""
    if base_df.empty or "sku" not in base_df.columns:
        return base_df
    adjusted_parts = []
    for _, grp in base_df.groupby("sku", dropna=False):
        adjusted_grp, *_ = compute_adjusted_curve(grp)
        if not adjusted_grp.empty:
            adjusted_parts.append(adjusted_grp)
    if adjusted_parts:
        return pd.concat(adjusted_parts, ignore_index=True)
    return base_df


@st.cache_data(show_spinner=False)
def build_cat_fam_summary(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cat_summary = summarize_by(df, "categoria")
    fam_summary = summarize_by(df, "familia")
    money_cols_cat = ["ingreso_total", "costo_total", "nuevo_ingreso", "delta_ingreso"]
    pct_cols_cat = ["margen_pct_actual", "margen_pct_nuevo", "delta_margen_pp"]
    money_cols_fam = ["ingreso_total", "costo_total", "nuevo_ingreso", "delta_ingreso"]
    pct_cols_fam = ["margen_pct_actual", "margen_pct_nuevo", "delta_margen_pp"]
    cat_with_total = add_total_row(cat_summary, "categoria", money_cols_cat, pct_cols_cat)
    fam_with_total = add_total_row(fam_summary, "familia", money_cols_fam, pct_cols_fam)
    return cat_with_total, fam_with_total


@st.cache_data(show_spinner=False)
def build_top50_tables(adjusted_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if adjusted_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    sku_summary = adjusted_df.groupby(["sku", "nombre_producto"], as_index=False).agg(
        kilos_total=("kilos_total", "sum"),
        ingreso_total=("venta_total", "sum"),
        costo_total=("costo_total", "sum"),
        nuevo_ingreso=("nuevo_revenue", "sum"),
        delta_ingreso=("delta_ingreso", "sum"),
    )
    sku_summary["margen_pct_actual"] = np.where(
        sku_summary["costo_total"] > 0,
        100 * (sku_summary["ingreso_total"] - sku_summary["costo_total"]) / sku_summary["costo_total"],
        np.nan,
    )
    sku_summary["margen_pct_nuevo"] = np.where(
        sku_summary["costo_total"] > 0,
        100 * (sku_summary["nuevo_ingreso"] - sku_summary["costo_total"]) / sku_summary["costo_total"],
        np.nan,
    )
    sku_summary["delta_margen_pp"] = sku_summary["margen_pct_nuevo"] - sku_summary["margen_pct_actual"]
    sku_top50 = sku_summary.sort_values(["delta_ingreso", "kilos_total"], ascending=False).head(50)
    sku_set = set(sku_top50["sku"].astype(str))
    detalle_top = adjusted_df[adjusted_df["sku"].astype(str).isin(sku_set)]
    return sku_top50, detalle_top


@st.cache_data(show_spinner=False)
def to_xlsx_bytes(detail_df: pd.DataFrame, cat_df: pd.DataFrame, fam_df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        detail_df.to_excel(writer, sheet_name="Detalle", index=False)
        if not cat_df.empty:
            cat_df.to_excel(writer, sheet_name="Resumen_categoria", index=False)
        if not fam_df.empty:
            fam_df.to_excel(writer, sheet_name="Resumen_familia", index=False)
    buffer.seek(0)
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def to_xlsx_top(detail_df: pd.DataFrame, resumen_df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        resumen_df.to_excel(writer, sheet_name="Top50_SKU", index=False)
        detail_df.to_excel(writer, sheet_name="Detalle_Top50", index=False)
    buffer.seek(0)
    return buffer.getvalue()

# -------------------------------------------------------------------
# Formateo de tablas
# -------------------------------------------------------------------
def format_display_table(df: pd.DataFrame, money_cols: List[str], pct_cols: List[str], money_decimals: int = 0, pct_decimals: int = 1) -> pd.DataFrame:
    out = df.copy()
    for col in money_cols:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: f"${x:,.{money_decimals}f}" if pd.notna(x) else "")
    for col in pct_cols:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: f"{x:,.{pct_decimals}f}%" if pd.notna(x) else "")
    return out


def add_total_row(df: pd.DataFrame, label_col: str, money_cols: List[str], pct_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    totals = {}
    for col in df.columns:
        if col == label_col:
            totals[col] = "TOTAL"
        elif col in money_cols:
            totals[col] = df[col].sum()
        elif col in pct_cols:
            totals[col] = np.nan
        else:
            totals[col] = np.nan
    # Recalcular porcentajes de margen a partir de sumas si existen las columnas necesarias
    if {"ingreso_total", "costo_total", "nuevo_ingreso"}.issubset(df.columns):
        ingreso_total = df["ingreso_total"].sum()
        costo_total = df["costo_total"].sum()
        nuevo_total = df["nuevo_ingreso"].sum()
        if costo_total > 0:
            totals["margen_pct_actual"] = 100 * (ingreso_total - costo_total) / costo_total
            totals["margen_pct_nuevo"] = 100 * (nuevo_total - costo_total) / costo_total
            totals["delta_margen_pp"] = totals["margen_pct_nuevo"] - totals["margen_pct_actual"]
    return pd.concat([df, pd.DataFrame([totals])], ignore_index=True)


# -------------------------------------------------------------------
# UI helpers
# -------------------------------------------------------------------
def build_filters(df: pd.DataFrame, key_prefix: str = "") -> Tuple[pd.DataFrame, Optional[str]]:
    filtered = df
    if "jefe_categoria" in df.columns:
        chiefs = ["Todos"] + sorted(df["jefe_categoria"].dropna().unique().tolist())
        sel_chief = st.selectbox("Jefe de categoria", options=chiefs, key=f"{key_prefix}_chief")
        filtered = filtered if sel_chief == "Todos" else filtered[filtered["jefe_categoria"] == sel_chief]

    cats = ["Todas"] + sorted(filtered["categoria"].dropna().unique().tolist())
    sel_cat = st.selectbox("Categoría", options=cats, key=f"{key_prefix}_cat")
    filtered = filtered if sel_cat == "Todas" else filtered[filtered["categoria"] == sel_cat]

    fams = ["Todas"] + sorted(filtered["familia"].dropna().unique().tolist())
    sel_fam = st.selectbox("Familia", options=fams, key=f"{key_prefix}_fam")
    filtered = filtered if sel_fam == "Todas" else filtered[filtered["familia"] == sel_fam]

    sku_options = filtered[["sku", "nombre_producto"]].drop_duplicates("sku")
    labels = ["Todos"] + [
        f"{row.sku} - {row.nombre_producto}" if pd.notna(row.nombre_producto) else str(row.sku)
        for _, row in sku_options.iterrows()
    ]
    sel_label = st.selectbox("SKU (busca escribiendo sku o nombre)", options=labels, key=f"{key_prefix}_sku")
    sel_sku = None if sel_label == "Todos" else sel_label.split(" - ", 1)[0]
    filtered = filtered if sel_sku is None else filtered[filtered["sku"] == sel_sku]

    return filtered, sel_sku


def render_scatter(df_plot: pd.DataFrame, x_col: str, y_col: str, title: str, x_label: str, y_label: str) -> None:
    hover_fields = {
        "nombre_cliente": True,
        "zona_ventas": True,
        "sku": True,
        "nombre_producto": True,
        x_col: ":,.0f",
        y_col: ":,.1f",
    }
    if "venta_total" in df_plot.columns:
        hover_fields["venta_total"] = ":,.0f"
    fig = px.scatter(
        df_plot,
        x=x_col,
        y=y_col,
        color="zona_ventas",
        hover_data=hover_fields,
        labels={x_col: x_label, y_col: y_label, "zona_ventas": "Zona de venta"},
        title=title,
    )
    fig.update_traces(marker=dict(size=12, opacity=0.85, line=dict(width=0.6, color="#e2e8f0")))
    fig.update_layout(
        height=650,
        legend_title="Zona de venta",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font_color="#0f172a",
        margin=dict(l=40, r=20, t=60, b=60),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)


# -------------------------------------------------------------------
# Layout principal
# -------------------------------------------------------------------
st.title("Curva teorica y ajuste por cliente")
st.caption("Carga clients_data.csv, limpia outliers (p5-p95), fuerza curva monotona decreciente y calcula deltas.")

df_raw = load_clients_data()
agg_base = get_aggregated(df_raw)
st.session_state.setdefault("adjusted_all", None)

with st.container():
    if "fecha_inicio" in df_raw.columns and "fecha_fin" in df_raw.columns:
        try:
            f_ini = pd.to_datetime(df_raw["fecha_inicio"], errors="coerce")
            f_fin = pd.to_datetime(df_raw["fecha_fin"], errors="coerce")
            start_date = f_ini.min()
            end_date = f_fin.max()
            if pd.notna(start_date) and pd.notna(end_date):
                st.caption(f"Rango de datos: {start_date.date()} a {end_date.date()}")
        except Exception:
            pass

tab_data, tab_curve, tab_summary = st.tabs(["Datos por cliente", "Curva teorica y ajustes", "Resumenes"])

with tab_data:
    st.subheader("Explora ventas por categoria / familia / SKU")
    filtered_data, selected_sku_data = build_filters(agg_base, key_prefix="t1")
    agg_data = filtered_data
    table_cols = [
        "jefe_categoria",
        "categoria",
        "familia",
        "sku",
        "nombre_producto",
        "nombre_cliente",
        "zona_ventas",
        "kilos_total",
        "venta_total",
        "costo_total",
        "precio_real",
        "costo_kilo",
        "n_movimientos",
    ]
    money_cols_view = ["venta_total", "costo_total", "precio_real", "costo_kilo"]
    pct_cols_view: List[str] = []
    st.dataframe(format_display_table(agg_data[table_cols], money_cols_view, pct_cols_view), use_container_width=True)

    if st.button("Graficar precio vs volumen", key="plot_t1"):
        if agg_data.empty:
            st.info("Sin datos para graficar con los filtros actuales.")
        else:
            render_scatter(
                agg_data.rename(columns={"precio_real": "precio_plot", "kilos_total": "kilos_plot"}),
                x_col="kilos_plot",
                y_col="precio_plot",
                title="Kilos vs precio / kg (color por zona de ventas)",
                x_label="Kilos por cliente",
                y_label="Precio / kg",
            )

with tab_curve:
    st.subheader("Curva teorica por SKU y ajuste de clientes")
    filtered_curve, selected_sku = build_filters(agg_base, key_prefix="t2")

    if not selected_sku:
        st.info("Selecciona un SKU para construir la curva teórica.")
    else:
        agg_curve, vol_sorted, price_sorted, p5, p95 = compute_adjusted_curve(filtered_curve)
        if agg_curve.empty:
            st.info("No hay datos para el SKU seleccionado.")
        else:

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Clientes", f"{agg_curve.shape[0]}")
            col2.metric("P5 - P95 precio", f"{p5:,.0f} - {p95:,.0f}" if not np.isnan(p5) else "N/A")
            col3.metric("Δ Revenue SKU", f"${agg_curve['delta_ingreso'].sum():,.0f}")
            col4.metric("Δ Margen SKU", f"${agg_curve['delta_margen'].sum():,.0f}")

            # Gráfico con curva y marcadores especiales para clientes bajo la curva.
            chart_df = agg_curve.rename(columns={"kilos_total": "kilos_plot", "precio_real": "precio_plot"})
            hover_fields_curve = {
                "nombre_cliente": True,
                "zona_ventas": True,
                "sku": True,
                "nombre_producto": True,
                "precio_plot": ":,.0f",
                "kilos_plot": ":,.1f",
                "precio_teorico": ":,.0f",
                "delta_ingreso": ":,.0f",
            }
            fig_curve = px.scatter(
                chart_df,
                x="kilos_plot",
                y="precio_plot",
                color="zona_ventas",
                hover_data=hover_fields_curve,
                labels={
                    "kilos_plot": "Kilos por cliente",
                    "precio_plot": "Precio / kg (actual)",
                    "zona_ventas": "Zona de venta",
                },
                title="Curva teorica y clientes (cruces: bajo curva)",
            )
            fig_curve.update_traces(marker=dict(size=11, opacity=0.8, line=dict(width=0.6, color="#e2e8f0")))

            # Halo para clientes con precio actual por debajo de la curva
            d_cross = chart_df[(chart_df["precio_plot"] + 1e-6) < chart_df["precio_teorico"]]
            if not d_cross.empty:
                fig_curve.add_trace(
                    go.Scatter(
                        x=d_cross["kilos_plot"],
                        y=d_cross["precio_plot"],
                        mode="markers",
                        name="Bajo curva (actual)",
                        marker=dict(
                            symbol="circle",
                            size=13,
                            line=dict(width=2.2, color="black"),
                            color="rgba(255,255,255,0.1)",
                        ),
                        hoverinfo="skip",
                        showlegend=True,
                    )
                )

            # Curva teorica
            if vol_sorted.size > 0:
                fig_curve.add_trace(
                    go.Scatter(
                        x=vol_sorted,
                        y=price_sorted,
                        mode="lines",
                        name="Curva teorica",
                        line=dict(color="#0f172a", dash="dash", width=2),
                        hoverinfo="skip",
                    )
                )

            fig_curve.update_layout(
                height=650,
                legend_title="Zona de venta",
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                font_color="#0f172a",
                margin=dict(l=40, r=20, t=60, b=60),
                xaxis_title="Kilos por cliente",
                yaxis_title="Precio / kg",
            )
            st.plotly_chart(fig_curve, use_container_width=True, theme=None)

            st.markdown("**Detalle por cliente**")
            detail_cols = [
                "jefe_categoria",
                "nombre_cliente",
                "zona_ventas",
                "categoria",
                "familia",
                "sku",
                "nombre_producto",
                "kilos_total",
                "venta_total",
                "costo_total",
                "precio_real",
                "precio_teorico",
                "nuevo_precio",
                "nuevo_revenue",
                "delta_ingreso",
                "margen_actual",
                "nuevo_margen",
                "delta_margen",
                "margen_pct_actual",
                "margen_pct_nuevo",
                "delta_margen_pp",
            ]
            money_cols_detail = [
                "venta_total",
                "costo_total",
                "precio_real",
                "precio_teorico",
                "nuevo_precio",
                "nuevo_revenue",
                "delta_ingreso",
                "margen_actual",
                "nuevo_margen",
                "delta_margen",
            ]
            pct_cols_detail = ["margen_pct_actual", "margen_pct_nuevo", "delta_margen_pp"]
            st.dataframe(format_display_table(agg_curve[detail_cols], money_cols_detail, pct_cols_detail), use_container_width=True)

with tab_summary:
    st.subheader("Resumenes por categoria y familia")
    adjusted_all_state = st.session_state.get("adjusted_all")
    if not isinstance(adjusted_all_state, pd.DataFrame) or adjusted_all_state.empty:
        # Se calcula una sola vez y se reutiliza para evitar recomputos al generar XLSX.
        adjusted_all_state = compute_adjusted_all_cached(agg_base)
        st.session_state["adjusted_all"] = adjusted_all_state

    adjusted_all = adjusted_all_state
    filtered_sum, _ = build_filters(adjusted_all, key_prefix="t3")

    if st.button("Calcular resúmenes", key="btn_summary"):
        with st.spinner("Calculando resúmenes..."):
            if adjusted_all.empty:
                st.session_state["summary_pkg"] = None
            else:
                cat_with_total, fam_with_total = build_cat_fam_summary(filtered_sum)
                sku_top50, detalle_top = build_top50_tables(adjusted_all)

                summary_pkg = {
                    "cat": cat_with_total,
                    "fam": fam_with_total,
                    "sku_top": sku_top50,
                    "xlsx": to_xlsx_bytes(filtered_sum if not filtered_sum.empty else adjusted_all, cat_with_total, fam_with_total),
                    "xlsx_top": to_xlsx_top(detalle_top, sku_top50),
                }
                st.session_state["summary_pkg"] = summary_pkg

    pkg = st.session_state.get("summary_pkg")
    if pkg is None:
        st.info("Presiona 'Calcular resúmenes' para generar la vista.")
    else:
        cat_with_total = pkg.get("cat", pd.DataFrame())
        fam_with_total = pkg.get("fam", pd.DataFrame())
        sku_top50 = pkg.get("sku_top", pd.DataFrame())

        if not cat_with_total.empty:
            money_cols_cat = ["ingreso_total", "costo_total", "nuevo_ingreso", "delta_ingreso"]
            pct_cols_cat = ["margen_pct_actual", "margen_pct_nuevo", "delta_margen_pp"]
            st.markdown("**Resumen por categoría**")
            st.dataframe(format_display_table(cat_with_total, money_cols_cat, pct_cols_cat), use_container_width=True)

        if not fam_with_total.empty:
            money_cols_fam = ["ingreso_total", "costo_total", "nuevo_ingreso", "delta_ingreso"]
            pct_cols_fam = ["margen_pct_actual", "margen_pct_nuevo", "delta_margen_pp"]
            st.markdown("**Resumen por familia**")
            st.dataframe(format_display_table(fam_with_total, money_cols_fam, pct_cols_fam), use_container_width=True)

        if isinstance(pkg.get("xlsx"), (bytes, bytearray)) and len(pkg["xlsx"]) > 0:
            st.download_button(
                "Descargar detalle (XLSX)",
                data=pkg["xlsx"],
                file_name="detalle_ajustes.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        if not sku_top50.empty:
            money_cols_sku = ["ingreso_total", "costo_total", "nuevo_ingreso", "delta_ingreso"]
            pct_cols_sku = ["margen_pct_actual", "margen_pct_nuevo", "delta_margen_pp"]
            st.markdown("**Top 50 SKUs por impacto (sin filtros, para seguimiento)**")
            st.dataframe(format_display_table(sku_top50, money_cols_sku, pct_cols_sku), use_container_width=True)
            if isinstance(pkg.get("xlsx_top"), (bytes, bytearray)) and len(pkg["xlsx_top"]) > 0:
                st.download_button(
                    "Descargar Top 50 (XLSX)",
                    data=pkg["xlsx_top"],
                    file_name="top50_sku_impacto.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
