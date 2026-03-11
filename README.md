# Curva Monótona y Plan de Experimentos (clients_streamlit_app.py)

Este documento explica, en base a `clients_streamlit_app.py`, cómo se construye la curva teórica monótona decreciente y cómo se sugieren/asignan grupos de experimento por cliente.

## 1) Cómo se genera la curva monótona decreciente

La app trabaja sobre `clients_data.csv` y agrega a nivel cliente-SKU:

- `kilos_total = sum(kilos)`
- `venta_total = sum(venta)`
- `costo_total = sum(costo)`
- `precio_real = venta_total / kilos_total`

Luego, para construir la curva teórica por SKU:

1. Limpieza de outliers de precio con winsorización:
- Se calculan percentiles `p5` y `p95` de `precio_real`.
- Se filtra al rango `[p5, p95]`.
- Se usa `precio_limpio = clip(precio_real, p5, p95)`.

2. Orden por volumen:
- Se ordenan clientes por `kilos_total` ascendente.

3. Forzar monotonicidad:
- Se aplica isotonic regression tipo PAV sobre `-precio_limpio`.
- Luego se vuelve al signo original.
- Resultado: `precio_teorico` no creciente con el volumen (curva monótona decreciente).

4. Predicción por cliente:
- `precio_teorico` se obtiene por interpolación lineal (`np.interp`) sobre la curva.

5. Ajuste económico:
- Si `precio_real < precio_teorico`, se propone `nuevo_precio = precio_teorico`.
- Si no, se mantiene `nuevo_precio = precio_real`.
- Se calculan `delta_ingreso`, `delta_margen`, etc.

## 2) Cómo se sugieren SKUs y grupos de experimento

### 2.1 Sugerencia de SKUs (`select_experiment_skus`)

La preselección (máx. 6 SKUs) combina:

- 2 por mayor `kilos_total` (volumen),
- 2 por mayor desviación de `margen_pct_actual` (variabilidad),
- 2 por mayor `delta_margen` (potencial).

Filtros previos:

- mínimo `50` clientes por SKU,
- mínimo `delta_ingreso >= 200000`,
- exclusión de categorías:  
  `CAFE Y BEBIDAS CALIENTES`, `CARNES`, `HIGIENE Y LIMPIEZA`.

### 2.2 Asignación de grupos por cliente (`assign_experiment_groups`)

Para cada cliente:

- `gap_teorico = max(precio_teorico - precio_real, 0)`.

Se crean grupos según percentiles del gap positivo (`33`, `66`, `90`):

- `A_Control` (sin subida, factor `0.0`)
- `B_Subida_30` (factor `0.3`)
- `C_Subida_50` (factor `0.5`)
- `D_Subida_70` (factor `0.7`)
- `E_Normalizacion` (factor `1.0`)

Precio experimental:

- `precio_experimental = min(precio_real + factor * gap_teorico, precio_teorico)`.

Control adicional:

- Del universo con `gap_teorico > 0`, un `10%` aleatorio se mueve a `A_Control` para comparación.

## 3) Ejemplo: si eliges solo clientes "Punto a Punto"

Si en filtros eliges `Zona de ventas = Punto a Punto`:

- el universo de clientes para resumen/experimentos se restringe a ese segmento,
- la selección de SKUs sugeridos se recalcula solo con ese subset,
- la asignación de grupos A-E se hace solo sobre clientes Punto a Punto.

Importante:

- en el plan de experimentos la lógica es "curva global por SKU, clientes filtrados".  
  Es decir, la referencia de curva se calcula a nivel SKU y luego se aplica al subconjunto filtrado (por ejemplo, Punto a Punto).

## 4) Período de fechas usado en el detalle

Según `clients_data.csv` actual:

- `fecha_inicio`: `2025-10-27`
- `fecha_fin`: `2026-01-27`

Eso corresponde a `92` días, aproximadamente `3.02` meses.  
Por lo tanto, sí: el detalle está basado en ~3 meses de data.
