# -*- coding: utf-8 -*-
"""
信用組合 Análisis de datos de sociosアプリ
================================
Excelファイルをドラッグ＆ドロップして、非線形・交互作用の構造を探索するツール。

起動:
    streamlit run coop_analyzer.py

必須:  streamlit pandas numpy scikit-learn plotly openpyxl scipy
任意:  xgboost（勾配ブースティング）, lingam（因果探索）
"""

import io
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import lingam
    HAS_LINGAM = True
except Exception:
    HAS_LINGAM = False


# =====================================================================
# 設定
# =====================================================================
TEAL = "#15706B"
TEAL_L = "#4A9B96"
ACCENT = "#C1553B"
GRAY = "#8A8A8A"
SEQ = [TEAL, TEAL_L, "#8FBFBC", "#C9AE6B", ACCENT, "#6B7B8C", "#A8998A", "#5D8AA8"]
DIVERGE = ["#2C6E63", "#7FB0A6", "#E8E8E8", "#E0A08C", "#C1553B"]

POSITIVE_TOKENS = {"si", "sí", "s", "yes", "y", "true", "1", "はい", "有", "あり", "○", "1.0"}
NEGATIVE_TOKENS = {"no", "n", "false", "0", "いいえ", "無", "なし", "×", "0.0"}

ROLE_OPTIONS = ["ID", "カテゴリ", "数値", "日付", "二値フラグ", "除外"]
ZERO_OPTIONS = ["真のゼロ", "欠損と同義", "ゼロなし"]

st.set_page_config(page_title="Análisis de datos de socios", layout="wide", page_icon="📊")

st.markdown(f"""
<style>
    .stApp {{ font-family: 'Yu Gothic UI', 'BIZ UDPGothic', sans-serif; }}
    h1, h2, h3 {{ color: {TEAL}; }}
    div[data-testid="stMetricValue"] {{ font-size: 1.5rem; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 2px; }}
    .stTabs [data-baseweb="tab"] {{ border-radius: 0; padding: 8px 16px; }}
    .note {{ background:#F2F6F5; border-left:4px solid {TEAL}; padding:10px 14px;
             font-size:0.86rem; margin-bottom:12px; }}
    .warn {{ background:#FBF0EC; border-left:4px solid {ACCENT}; padding:10px 14px;
             font-size:0.86rem; margin-bottom:12px; }}
</style>
""", unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def warn(text):
    st.markdown(f'<div class="warn">{text}</div>', unsafe_allow_html=True)


def base_layout(fig, height=420, title=None):
    fig.update_layout(
        height=height, title=title, template="simple_white",
        font=dict(family="Yu Gothic UI, sans-serif", size=12),
        margin=dict(l=60, r=30, t=50 if title else 30, b=50),
        colorway=SEQ, hoverlabel=dict(font_size=12),
    )
    return fig


# =====================================================================
# コア: スキーマ推定
# =====================================================================
def read_any(uploaded, sheet_name=None, header_row=0):
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith((".csv", ".tsv", ".txt")):
        sep = "\t" if name.endswith(".tsv") else ","
        for enc in ("utf-8-sig", "cp932", "utf-8", "latin-1"):
            try:
                return pd.read_csv(io.BytesIO(data), sep=sep, encoding=enc, header=header_row)
            except Exception:
                continue
        raise ValueError("No se pudo detectar la codificación del archivo CSV.")
    return pd.read_excel(io.BytesIO(data), sheet_name=sheet_name, header=header_row)


def list_sheets(uploaded):
    if uploaded.name.lower().endswith((".csv", ".tsv", ".txt")):
        return None
    xl = pd.ExcelFile(io.BytesIO(uploaded.getvalue()))
    return xl.sheet_names


def _binary_positive(series):
    """二値Columnasの陽性ラベルを推定して返す。Evaluaciónできなければ None。"""
    vals = list(pd.Series(series).dropna().unique())
    if len(vals) != 2:
        return None
    low = [str(v).strip().lower() for v in vals]
    pos = [i for i, t in enumerate(low) if t in POSITIVE_TOKENS]
    neg = [i for i, t in enumerate(low) if t in NEGATIVE_TOKENS]
    if len(pos) == 1 and len(neg) == 1:
        return vals[pos[0]]
    try:
        nums = [float(v) for v in vals]
        if sorted(nums) == [0.0, 1.0]:
            return vals[nums.index(1.0)]
    except Exception:
        pass
    return None


def infer_schema(df):
    """各Columnasの役割・ゼロの扱い・構造的欠損を自動推定する。"""
    n = len(df)
    schema = {}
    for c in df.columns:
        s = df[c]
        nunique = s.nunique(dropna=True)
        na_rate = float(s.isna().mean())
        info = {"role": "カテゴリ", "zero": "ゼロなし", "structural_na": False,
                "positive": None, "na_rate": na_rate, "nunique": int(nunique)}

        if nunique <= 1 or na_rate > 0.95:
            info["role"] = "除外"
        elif pd.api.types.is_datetime64_any_dtype(s):
            info["role"] = "日付"
        elif nunique == n and n > 20 and (pd.api.types.is_integer_dtype(s) or
                                          pd.api.types.is_object_dtype(s) or
                                          pd.api.types.is_string_dtype(s)):
            info["role"] = "ID"
        elif nunique == 2:
            pos = _binary_positive(s)
            if pos is not None:
                info["role"] = "二値フラグ"
                info["positive"] = pos
            else:
                info["role"] = "カテゴリ"
        elif pd.api.types.is_numeric_dtype(s):
            info["role"] = "数値" if nunique > 6 else "カテゴリ"
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(s, errors="coerce")
            if parsed.notna().mean() > 0.9:
                info["role"] = "日付"
            else:
                info["role"] = "カテゴリ" if nunique <= 50 else "除外"

        if info["role"] == "数値" and pd.api.types.is_numeric_dtype(s):
            zero_rate = float((s == 0).mean())
            info["zero_rate"] = zero_rate
            info["zero"] = "真のゼロ" if zero_rate > 0.01 else "ゼロなし"
        if 0.05 < na_rate < 0.95:
            info["structural_na"] = True
        schema[c] = info
    return schema


def cols_by_role(schema, role):
    return [c for c, v in schema.items() if v["role"] == role]


def to_binary(series, positive):
    if positive is None:
        return pd.Series(np.nan, index=series.index)
    out = pd.Series(np.nan, index=series.index, dtype=float)
    mask = series.notna()
    out[mask] = (series[mask].astype(str).str.strip().str.lower()
                 == str(positive).strip().lower()).astype(float)
    return out


def build_working(df, schema, ratio_base, ref_date):
    """
    派生Variableを固定ルールで生成した作業用データフレームを返す。
    生成物: 二値フラグの0/1化、日付→経過年数、真のゼロColumnas→保有フラグ、比率Columnas
    """
    w = df.copy()
    meta = {"flags": [], "years": [], "ratios": [], "binaries": {}}

    for c, v in schema.items():
        if v["role"] == "除外":
            continue
        if v["role"] == "二値フラグ":
            newc = f"{c}[0/1]"
            w[newc] = to_binary(df[c], v["positive"])
            meta["binaries"][c] = newc
        elif v["role"] == "日付":
            d = pd.to_datetime(df[c], errors="coerce")
            newc = f"{c}_años_transcurridos"
            w[newc] = (pd.Timestamp(ref_date) - d).dt.days / 365.25
            meta["years"].append(newc)
        elif v["role"] == "数値":
            if v.get("zero") == "欠損と同義":
                w[c] = df[c].replace(0, np.nan)
            elif v.get("zero") == "真のゼロ" and v.get("zero_rate", 0) > 0.01:
                newc = f"{c}_tenencia"
                w[newc] = (pd.to_numeric(df[c], errors="coerce") > 0).astype(float)
                meta["flags"].append(newc)

    if ratio_base and ratio_base in w.columns:
        base = pd.to_numeric(w[ratio_base], errors="coerce").replace(0, np.nan)
        for c in cols_by_role(schema, "数値"):
            if c == ratio_base:
                continue
            newc = f"{c}÷{ratio_base}"
            w[newc] = pd.to_numeric(w[c], errors="coerce") / base
            meta["ratios"].append(newc)
    return w, meta


# =====================================================================
# コア: 指標
# =====================================================================
def gini(x):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    x = x[x >= 0]
    if len(x) < 2 or x.sum() == 0:
        return np.nan
    x = np.sort(x)
    n = len(x)
    return float((2 * np.sum((np.arange(1, n + 1)) * x)) / (n * x.sum()) - (n + 1) / n)


def lorenz_points(x):
    x = np.sort(np.asarray(pd.Series(x).dropna(), dtype=float))
    x = x[x >= 0]
    if len(x) == 0 or x.sum() == 0:
        return np.array([0, 1]), np.array([0, 1])
    cum = np.cumsum(x) / x.sum()
    return np.insert(np.arange(1, len(x) + 1) / len(x), 0, 0), np.insert(cum, 0, 0)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z ** 2 / n
    c = p + z ** 2 / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return (c - h) / d, (c + h) / d


def make_bins(x, nbins, method):
    x = pd.Series(x)
    try:
        if method == "分位ビン（等Recuento）":
            return pd.qcut(x, nbins, duplicates="drop")
        return pd.cut(x, nbins, duplicates="drop")
    except Exception:
        return pd.cut(x, min(nbins, 5), duplicates="drop")


def binned_rate(x, y, nbins=8, method="分位ビン（等Recuento）", binary=True):
    """ビンごとのイベント率（またはPromedio）とRecuento・信頼区間を返す。"""
    d = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"),
                      "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(d) < 20:
        return pd.DataFrame()
    d["bin"] = make_bins(d["x"], nbins, method)
    rows = []
    for b, g in d.groupby("bin", observed=True):
        if len(g) == 0:
            continue
        m = g["y"].mean()
        lo, hi = (wilson_ci(g["y"].sum(), len(g)) if binary
                  else (m - 1.96 * g["y"].std(ddof=1) / np.sqrt(len(g)),
                        m + 1.96 * g["y"].std(ddof=1) / np.sqrt(len(g))))
        rows.append({"bin": str(b), "center": g["x"].mean(), "n": len(g),
                     "rate": m, "lo": lo, "hi": hi})
    return pd.DataFrame(rows).sort_values("center").reset_index(drop=True)


def nonmonotonicity(x, y, nbins=8, binary=True):
    """
    非線形性スコア。ビン別イベント率の変動のうち、
    線形では説明できず、かつ標本誤差では説明できない分を返す。
    """
    t = binned_rate(x, y, nbins=nbins, binary=binary)
    if len(t) < 4:
        return None
    w = t["n"].values.astype(float)
    r = t["rate"].values.astype(float)
    c = t["center"].values.astype(float)
    wm = np.average(r, weights=w)
    var_total = np.average((r - wm) ** 2, weights=w)
    # 重み付き線形Regresión
    cm = np.average(c, weights=w)
    denom = np.average((c - cm) ** 2, weights=w)
    slope = np.average((c - cm) * (r - wm), weights=w) / denom if denom > 0 else 0.0
    pred = wm + slope * (c - cm)
    var_resid = np.average((r - pred) ** 2, weights=w)
    exp_noise = np.average(r * (1 - r) / w, weights=w) if binary else \
        np.average((r - pred) ** 2, weights=w) / max(len(t) - 2, 1)
    excess = max(var_resid - exp_noise, 0.0)
    d = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"),
                      "y": pd.to_numeric(y, errors="coerce")}).dropna()
    sp = stats.spearmanr(d["x"], d["y"]).statistic if len(d) > 10 else np.nan
    return {"非線形超過変動": excess,
            "非線形寄与率": excess / var_total if var_total > 0 else 0.0,
            "全変動": var_total, "Spearman": sp,
            "最小ビン率": r.min(), "最大ビン率": r.max(), "n": int(w.sum())}


def interaction_scan(df, cat_cols, target, min_cell=15, top=25, max_levels=15):
    """
    カテゴリColumnas2つの組合せについて、加法Modeloからの乖離を集計する。
    交互作用が強いペアほど上位に来る。
    """
    y = pd.to_numeric(df[target], errors="coerce")
    base = df.loc[y.notna()].copy()
    y = y.loc[y.notna()]
    overall = y.mean()
    out = []
    for a, b in combinations(cat_cols, 2):
        sub = pd.DataFrame({"a": base[a].astype(str), "b": base[b].astype(str), "y": y}).dropna()
        if not (2 <= sub["a"].nunique() <= max_levels) or not (2 <= sub["b"].nunique() <= max_levels):
            continue
        ra = sub.groupby("a")["y"].mean()
        rb = sub.groupby("b")["y"].mean()
        cell = sub.groupby(["a", "b"])["y"].agg(["mean", "size"]).reset_index()
        cell = cell[cell["size"] >= min_cell]
        if len(cell) < 3:
            continue
        cell["expected"] = (cell["a"].map(ra) + cell["b"].map(rb) - overall).clip(0.001, 0.999)
        cell["dev"] = cell["mean"] - cell["expected"]
        # 標本誤差で標準化し、自由度分を差し引くことで小セルの偶然を除く
        var = overall * (1 - overall)
        cell["z2"] = cell["size"] * cell["dev"] ** 2 / var
        dof = max((sub["a"].nunique() - 1) * (sub["b"].nunique() - 1), 1)
        chi2 = float(cell["z2"].sum())
        excess = max(chi2 - dof, 0.0)
        pval = float(1 - stats.chi2.cdf(chi2, dof)) if dof > 0 else np.nan
        worst = cell.loc[cell["z2"].idxmax()]
        out.append({"VariableA": a, "VariableB": b, "交互作用スコア": excess / len(sub),
                    "χ²": chi2, "自由度": dof, "p値": pval,
                    "最大乖離セル": f"{worst['a']} × {worst['b']}",
                    "実測": worst["mean"], "加法予測": worst["expected"],
                    "Recuento": int(worst["size"]), "有効セル数": len(cell)})
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_values("交互作用スコア", ascending=False).head(top).reset_index(drop=True)


def cell_risk_scan(df, cat_cols, target, min_cell=15, top=25, max_levels=15):
    """
    カテゴリ2Columnasの全組合せについて、イベント率が全体Promedioから乖離するセルをColumnas挙する。
    交互作用として識別できなくても、実務上の高リスク層はここに現れる。
    """
    y = pd.to_numeric(df[target], errors="coerce")
    base = df.loc[y.notna()]
    y = y.loc[y.notna()]
    overall = y.mean()
    rows = []
    for a, b in combinations(cat_cols, 2):
        sub = pd.DataFrame({"a": base[a].astype(str), "b": base[b].astype(str), "y": y}).dropna()
        if not (2 <= sub["a"].nunique() <= max_levels) or not (2 <= sub["b"].nunique() <= max_levels):
            continue
        g = sub.groupby(["a", "b"])["y"].agg(["mean", "size", "sum"]).reset_index()
        g = g[g["size"] >= min_cell]
        for _, r in g.iterrows():
            lo, hi = wilson_ci(r["sum"], r["size"])
            rows.append({"セグメント": f"{a}={r['a']} × {b}={r['b']}",
                         "Recuento": int(r["size"]), "率": r["mean"],
                         "リフト": r["mean"] / overall if overall else np.nan,
                         "95%下限": lo, "95%上限": hi})
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows).drop_duplicates("セグメント")
    return d.sort_values("95%下限", ascending=False).head(top).reset_index(drop=True)


# =====================================================================
# コア: 前処理パイプライン
# =====================================================================
def make_preprocessor(X, scale=True):
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ], remainder="drop")


def feature_names(pre, X):
    names = []
    for name, trans, cols in pre.transformers_:
        if name == "num":
            names += list(cols)
        elif name == "cat" and len(cols):
            oh = trans.named_steps["oh"]
            names += list(oh.get_feature_names_out(cols))
    return names


def onehot_mapping(pre, X):
    """One-Hot後のColumnas名から、元のColumnas名と水準名への対応を作る。"""
    m = {}
    for name, trans, cols in pre.transformers_:
        if name == "cat" and len(cols):
            oh = trans.named_steps["oh"]
            for col, cats_ in zip(cols, oh.categories_):
                for lv in cats_:
                    m[f"{col}_{lv}"] = (col, str(lv))
    return m


def get_models(task, balanced):
    cw = "balanced" if balanced else None
    if task == "Clasificación":
        m = {"Regresión logística": LogisticRegression(max_iter=2000, class_weight=cw),
             "Bosque aleatorio": RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                                         class_weight=cw, random_state=0, n_jobs=1),
             "SVM (RBF)": SVC(probability=True, class_weight=cw, random_state=0)}
        if HAS_XGB:
            m["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.06,
                                         subsample=0.9, colsample_bytree=0.9,
                                         eval_metric="logloss", random_state=0, n_jobs=1)
    else:
        m = {"Regresión Ridge": Ridge(alpha=1.0),
             "Bosque aleatorio": RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                                         random_state=0, n_jobs=1),
             "SVM (RBF)": SVR()}
        if HAS_XGB:
            m["XGBoost"] = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.06,
                                        subsample=0.9, colsample_bytree=0.9, random_state=0, n_jobs=1)
    return m


def tree_rules(tree, feat_names, X, y, task, onehot_map=None):
    """決定木の葉ノードを「条件 → Efecto」の表に変換する。"""
    onehot_map = onehot_map or {}
    t = tree.tree_
    leaf_id = tree.apply(X)
    rows = []

    def walk(node, conds):
        if t.children_left[node] == -1:
            mask = leaf_id == node
            n = int(mask.sum())
            if n == 0:
                return
            val = float(np.mean(y[mask]))
            rows.append({"ルール": " かつ ".join(conds) if conds else "（全体）",
                         "Recuento": n, "値": val})
            return
        f = feat_names[t.feature[node]]
        thr = t.threshold[node]
        if f in onehot_map:                      # One-HotColumnasは「Columnas=水準」で表記
            col, lvl = onehot_map[f]
            walk(t.children_left[node], conds + [f"{col} ≠ {lvl}"])
            walk(t.children_right[node], conds + [f"{col} = {lvl}"])
        else:
            walk(t.children_left[node], conds + [f"{f} ≤ {thr:,.2f}"])
            walk(t.children_right[node], conds + [f"{f} > {thr:,.2f}"])

    walk(0, [])
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    overall = float(np.mean(y))
    d["リフト"] = d["値"] / overall if overall != 0 else np.nan
    if task == "Clasificación":
        ci = [wilson_ci(r["値"] * r["Recuento"], r["Recuento"]) for _, r in d.iterrows()]
        d["下限"] = [c[0] for c in ci]
        d["上限"] = [c[1] for c in ci]
    return d.sort_values("リフト", ascending=False).reset_index(drop=True)


# =====================================================================
# セッション初期化
# =====================================================================
for k, v in [("df", None), ("schema", None), ("raw", None), ("fname", None),
             ("pops", {}), ("ratio_base", None), ("scores", {})]:
    if k not in st.session_state:
        st.session_state[k] = v


# =====================================================================
# サイドバー
# =====================================================================
st.sidebar.markdown(f"### <span style='color:{TEAL}'>Análisis de datos de socios</span>",
                    unsafe_allow_html=True)

if st.session_state.df is None:
    st.sidebar.info("Cargue un archivo en la pestaña ①.")
    work = None
    schema = None
    view_ratio = False
    view_center = "Promedio"
else:
    df = st.session_state.df
    schema = st.session_state.schema
    work, meta = build_working(df, schema, st.session_state.ratio_base,
                               st.session_state.get("ref_date", pd.Timestamp.today()))

    view_ratio = False
    view_center = "Promedio"


def numeric_pool(w, schema, ratio):
    """表示モードに応じた数値Columnasのリストを返す。"""
    base = cols_by_role(schema, "数値") + [c for c in w.columns if c.endswith("_años_transcurridos")]
    if ratio:
        rb = st.session_state.ratio_base
        out = []
        for c in base:
            r = f"{c}÷{rb}"
            out.append(r if r in w.columns else c)
        return [c for c in dict.fromkeys(out) if c in w.columns]
    return [c for c in dict.fromkeys(base) if c in w.columns]


def binary_pool(w, schema):
    return ([v for v in [f"{c}[0/1]" for c in cols_by_role(schema, "二値フラグ")] if v in w.columns]
            + [c for c in w.columns if c.endswith("_tenencia")])


def category_pool(w, schema):
    return [c for c in cols_by_role(schema, "カテゴリ") if c in w.columns]


# =====================================================================
# タブ
# =====================================================================
T = st.tabs(["① Carga de datos", "② Distribución", "③ Tabla cruzada", "④ Árbol de decisión (estructura de datos)",
             "⑤ Predicción con aprendizaje automático", "⑥ Exploración causal (LiNGAM)", "⑦ Detección de anomalías"])

# ---------------------------------------------------------------------
# ① Carga de datos
# ---------------------------------------------------------------------
with T[0]:
    st.header("① Carga de datos")
    note("Arrastre y suelte un archivo Excel o CSV. Se mostrarán las primeras 20 filas.")

    up = st.file_uploader("Seleccionar archivo", type=["xlsx", "xlsm", "csv", "tsv"])

    if up is not None:
        c1, c2 = st.columns([2, 1])
        sheets = list_sheets(up)
        with c1:
            sheet = st.selectbox("Hoja", sheets) if sheets else None
        with c2:
            hrow = st.number_input("Fila de encabezado (desde 1)", 1, 20, 1) - 1

        if st.button("Cargar", type="primary") or st.session_state.fname != up.name:
            try:
                raw = read_any(up, sheet, hrow)
                st.session_state.raw = raw
                st.session_state.df = raw
                st.session_state.schema = infer_schema(raw)
                st.session_state.fname = up.name
                nums = cols_by_role(st.session_state.schema, "数値")
                guess = [c for c in nums if any(k in c.upper()
                         for k in ["INGRE", "SALAR", "収入", "所得", "月収"])]
                st.session_state.ratio_base = guess[0] if guess else (nums[0] if nums else None)
                st.session_state.pops = {}
                st.rerun()
            except Exception as e:
                st.error(f"Error al cargar: {e}")

    if st.session_state.df is not None:
        df = st.session_state.df
        schema = st.session_state.schema
        st.success(f"{st.session_state.fname} — {len(df):,} filas × {len(df.columns)} columnas")

        st.subheader("Vista previa")
        st.dataframe(df.head(20), use_container_width=True, height=260)


# ---------------------------------------------------------------------
# ② Distribución
# ---------------------------------------------------------------------
with T[1]:
    st.header("② Distribución")
    if work is None:
        st.info("Primero cargue los datos.")
    else:
        nums = numeric_pool(work, schema, view_ratio)
        st.subheader("Distribución")
        if nums:
            xcol = st.selectbox("Variable", nums, key="ov_x")
            s = pd.to_numeric(work[xcol], errors="coerce").dropna()
            fig = px.histogram(s, nbins=40, color_discrete_sequence=[TEAL])
            fig.update_layout(xaxis_title=xcol, yaxis_title="Recuento")
            st.plotly_chart(base_layout(fig, 440), use_container_width=True)
        else:
            st.warning("No hay variables numéricas disponibles para mostrar la distribución.")

# ---------------------------------------------------------------------
# ③ Tabla cruzada
# ---------------------------------------------------------------------
with T[2]:
    st.header("③ Tabla cruzada")
    if work is None:
        st.info("Primero cargue los datos.")
    else:
        nums = numeric_pool(work, schema, view_ratio)
        bins_ = binary_pool(work, schema)
        cats = category_pool(work, schema)
        axis_pool = cats + bins_ + nums

        c1, c2, c3 = st.columns(3)
        with c1:
            rowv = st.selectbox("Filas", axis_pool, key="ct_r")
        with c2:
            colv = st.selectbox("Columnas", [c for c in axis_pool if c != rowv], key="ct_c")
        with c3:
            metric = st.selectbox("Medida", ["Recuento", "Promedio", "Mediana"], key="ct_m")

        valcol = None
        if metric in ("Promedio", "Mediana"):
            valcol = st.selectbox("Variable numérica a resumir", nums, key="ct_v")

        def axis_series(c):
            s = work[c]
            if c in nums:
                return make_bins(pd.to_numeric(s, errors="coerce"), 5,
                                 "分位ビン（等Recuento）").astype(str)
            return s.astype(str)

        d = pd.DataFrame({"r": axis_series(rowv), "c": axis_series(colv)})
        if valcol is not None:
            d["v"] = pd.to_numeric(work[valcol], errors="coerce")
        d = d.replace("nan", np.nan).dropna(subset=["r", "c"])

        cnt = d.pivot_table(index="r", columns="c", aggfunc="size", fill_value=0)
        if metric == "Recuento":
            pv = cnt
            fmt = "{:,.0f}"
        else:
            agg = {"Promedio": "mean", "Mediana": "median"}[metric]
            pv = d.pivot_table(index="r", columns="c", values="v", aggfunc=agg)
        fig = px.imshow(pv, text_auto=True, aspect="auto",
                        color_continuous_scale=["#FFFFFF", TEAL])
        fig.update_xaxes(title=colv, tickangle=-30)
        fig.update_yaxes(title=rowv)
        st.plotly_chart(base_layout(fig, 500), use_container_width=True)

# ---------------------------------------------------------------------
# ④ 決定木 / ⑤ Predicción con aprendizaje automático
# ---------------------------------------------------------------------
if True:
    if work is None:
        for tab in (T[3], T[4]):
            with tab:
                st.info("Primero cargue los datos.")
    else:
        nums = numeric_pool(work, schema, False)
        bins_ = binary_pool(work, schema)
        cats = category_pool(work, schema)
        ratios = [c for c in work.columns if "÷" in c]

        m1, m2 = T[3], T[4]

        # ===== 決定木 =====
        with m1:
            st.header("④ Árbol de decisión (estructura de datos)")
            note("Muestra en un solo árbol la estructura de ramificación entre la variable objetivo y las variables explicativas.")
            tgt_pool = bins_ + nums
            c1, c2, c3 = st.columns(3)
            with c1:
                ty = st.selectbox("Variable objetivo", tgt_pool, key="dt_y")
            task_t = "Clasificación" if ty in bins_ else "Regresión"
            with c2:
                depth = st.slider("Profundidad máxima", 2, 8, 4)
            with c3:
                leaf = st.slider("Tamaño mínimo de hoja (%)", 1, 15, 3)

            default_x = [c for c in cats + nums + bins_ if c != ty][:20]
            xs = st.multiselect("Variables explicativas", [c for c in cats + nums + bins_ + ratios if c != ty],
                                default=default_x, key="dt_x")

            if xs and st.button("Crear árbol de decisión", type="primary"):
                d = work[xs + [ty]].copy()
                d[ty] = pd.to_numeric(d[ty], errors="coerce")
                d = d.dropna(subset=[ty])
                X, y = d[xs], d[ty].values
                pre = make_preprocessor(X, scale=False)
                Xt = pre.fit_transform(X)
                fn = feature_names(pre, X)
                ml = max(int(len(d) * leaf / 100), 5)
                if task_t == "Clasificación":
                    tr = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=ml,
                                                class_weight="balanced", random_state=0)
                else:
                    tr = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=ml, random_state=0)
                tr.fit(Xt, y)

                st.subheader("Estructura del árbol")
                import matplotlib.pyplot as plt
                import matplotlib
                from matplotlib import font_manager
                avail = {f.name for f in font_manager.fontManager.ttflist}
                for jp in ["Yu Gothic", "Meiryo", "MS Gothic", "BIZ UDPGothic",
                           "Hiragino Sans", "IPAexGothic", "Noto Sans CJK JP",
                           "Noto Sans JP", "TakaoGothic"]:
                    if jp in avail:
                        matplotlib.rcParams["font.family"] = jp
                        break
                else:
                    matplotlib.rcParams["font.family"] = "DejaVu Sans"
                    st.caption("No se encontró una fuente compatible; algunos caracteres podrían mostrarse como cuadrados.")
                matplotlib.rcParams["axes.unicode_minus"] = False
                fig, ax = plt.subplots(figsize=(min(4 + 2.2 ** depth, 26), 3 + depth * 1.5))
                plot_tree(tr, feature_names=fn, filled=True, rounded=False, fontsize=8,
                          impurity=False, proportion=True, ax=ax,
                          class_names=["0", "1"] if task_t == "Clasificación" else None)
                st.pyplot(fig, use_container_width=True)

        # ===== 予測Modelo =====
        with m2:
            st.header("⑤ Predicción con aprendizaje automático")
            note("Los datos se dividen aleatoriamente en 70% para entrenamiento y 30% para prueba. "
                 "Siempre se comparan cuatro modelos: lineal, XGBoost, bosque aleatorio y SVM.")
            tgt_pool = bins_ + nums
            if not tgt_pool:
                st.warning("No hay una variable disponible como objetivo.")
            else:
                ty2 = st.selectbox("Variable objetivo", tgt_pool, key="pm_y")
                task = "Clasificación" if ty2 in bins_ else "Regresión"
                leak_kw = ["SALDO", "CUOTA", "残高", "月賦"]
                cand = [c for c in cats + nums + bins_ + ratios if c != ty2]
                leaks = [c for c in cand if any(k in c.upper() for k in leak_kw)]
                default_x2 = [c for c in cand if c not in leaks][:18]
                xs2 = st.multiselect("Variables explicativas", cand, default=default_x2, key="pm_x")

                models = get_models(task, task == "Clasificación")
                linear_name = "Regresión logística" if task == "Clasificación" else "Regresión Ridge"
                st.caption(f"{linear_name}, XGBoost, bosque aleatorio y SVM se ejecutan automáticamente. Métrica: "
                           f"{'ROC-AUC' if task == 'Clasificación' else 'R²'}")

                if xs2 and st.button("Entrenar", type="primary"):
                    if "XGBoost" not in models:
                        st.error("XGBoost no está instalado. Ejecute `pip install xgboost`.")
                    else:
                        d = work[xs2 + [ty2]].copy()
                        d[ty2] = pd.to_numeric(d[ty2], errors="coerce")
                        d = d.dropna(subset=[ty2])
                        X, y = d[xs2], d[ty2].values
                        if task == "Clasificación":
                            y = y.astype(int)
                            if len(np.unique(y)) != 2:
                                st.error("Para clasificar, la variable objetivo debe contener tanto 0 como 1.")
                                st.stop()
                            counts = pd.Series(y).value_counts()
                            if counts.min() < 2:
                                st.error("Para clasificar, se necesitan al menos dos casos de 0 y dos de 1.")
                                st.stop()
                            stratify = y
                        else:
                            stratify = None

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.30, random_state=42, stratify=stratify)
                        selected = [linear_name, "XGBoost", "Bosque aleatorio", "SVM (RBF)"]
                        fitted = {}
                        scores = []
                        with st.spinner("Entrenando y evaluando los cuatro modelos..."):
                            for name in selected:
                                pipe = Pipeline([("pre", make_preprocessor(X_train)),
                                                 ("m", models[name])])
                                pipe.fit(X_train, y_train)
                                if task == "Clasificación":
                                    train_pred = pipe.predict_proba(X_train)[:, 1]
                                    test_pred = pipe.predict_proba(X_test)[:, 1]
                                    train_score = roc_auc_score(y_train, train_pred)
                                    test_score = roc_auc_score(y_test, test_pred)
                                else:
                                    train_score = r2_score(y_train, pipe.predict(X_train))
                                    test_score = r2_score(y_test, pipe.predict(X_test))
                                fitted[name] = pipe
                                scores.append({"Modelo": name, "Puntuación de entrenamiento": train_score,
                                               "Puntuación de prueba": test_score})

                        score_df = pd.DataFrame(scores)
                        best = score_df.sort_values("Puntuación de prueba", ascending=False).iloc[0]
                        best_name = best["Modelo"]
                        st.success(f"Mejor modelo: **{best_name}**")
                        c1, c2 = st.columns(2)
                        c1.metric("Puntuación de entrenamiento", f"{best['Puntuación de entrenamiento']:.4f}")
                        c2.metric("Puntuación de prueba", f"{best['Puntuación de prueba']:.4f}")

                        st.subheader("Importancia de variables del mejor modelo")
                        scoring = "roc_auc" if task == "Clasificación" else "r2"
                        pi = permutation_importance(
                            fitted[best_name], X_test, y_test, n_repeats=8,
                            random_state=42, scoring=scoring, n_jobs=1)
                        pid = pd.DataFrame({"Variable": X_test.columns,
                                            "Importancia": np.clip(pi.importances_mean, 0, None)})
                        pid = pid.sort_values("Importancia", ascending=False).head(15)
                        fig = px.bar(pid.iloc[::-1], x="Importancia", y="Variable", orientation="h",
                                     color_discrete_sequence=[TEAL])
                        st.plotly_chart(base_layout(fig, 430), use_container_width=True)

                        st.subheader(f"Dirección del modelo lineal ({linear_name})")
                        linear_pipe = fitted[linear_name]
                        transformed_names = feature_names(linear_pipe.named_steps["pre"], X_train)
                        coef = np.ravel(linear_pipe.named_steps["m"].coef_)
                        Xt = np.asarray(linear_pipe.named_steps["pre"].transform(X_train), dtype=float)
                        design = np.column_stack([np.ones(len(Xt)), Xt])
                        beta = np.r_[linear_pipe.named_steps["m"].intercept_, coef]

                        if task == "Clasificación":
                            prob = linear_pipe.predict_proba(X_train)[:, 1]
                            weight = np.clip(prob * (1 - prob), 1e-9, None)
                            hessian = design.T @ (design * weight[:, None])
                            penalty = np.diag(np.r_[0.0, np.repeat(
                                1.0 / linear_pipe.named_steps["m"].C, Xt.shape[1])])
                            covariance = np.linalg.pinv(hessian + penalty)
                            standard_error = np.sqrt(np.clip(np.diag(covariance), 0, None))
                            statistic = np.divide(beta, standard_error,
                                                  out=np.zeros_like(beta), where=standard_error > 0)
                            p_values = 2 * stats.norm.sf(np.abs(statistic[1:]))
                            odds_ratio = np.exp(np.clip(coef, -50, 50))
                            effects = pd.DataFrame({"Variable explicativa (nivel)": transformed_names,
                                                    "Razón de momios": odds_ratio, "Valor P": p_values})
                            effects["Dirección"] = np.where(effects["Razón de momios"] >= 1,
                                                       "Positiva", "Negativa")
                            effects["Valor de ordenación"] = np.abs(np.log(effects["Razón de momios"]))
                            value_col = "Razón de momios"
                            reference = 1.0
                        else:
                            alpha = linear_pipe.named_steps["m"].alpha
                            gram = design.T @ design
                            penalty = np.diag(np.r_[0.0, np.repeat(alpha, Xt.shape[1])])
                            inv_penalized = np.linalg.pinv(gram + penalty)
                            fitted_values = linear_pipe.predict(X_train)
                            residual = y_train - fitted_values
                            effective_df = np.trace(gram @ inv_penalized)
                            residual_df = max(len(y_train) - effective_df, 1.0)
                            variance = np.sum(residual ** 2) / residual_df
                            covariance = variance * inv_penalized @ gram @ inv_penalized
                            standard_error = np.sqrt(np.clip(np.diag(covariance), 0, None))
                            statistic = np.divide(beta, standard_error,
                                                  out=np.zeros_like(beta), where=standard_error > 0)
                            p_values = 2 * stats.t.sf(np.abs(statistic[1:]), df=residual_df)
                            effects = pd.DataFrame({"Variable explicativa (nivel)": transformed_names,
                                                    "Coeficiente": coef, "Valor P": p_values})
                            effects["Dirección"] = np.where(effects["Coeficiente"] >= 0,
                                                       "Positiva", "Negativa")
                            effects["Valor de ordenación"] = effects["Coeficiente"].abs()
                            value_col = "Coeficiente"
                            reference = 0.0

                        effects = effects.sort_values("Valor de ordenación", ascending=False)
                        chart_data = pd.concat([
                            effects.head(30), effects[effects["Valor P"] <= 0.1]
                        ]).drop_duplicates("Variable explicativa (nivel)").sort_values("Valor de ordenación")
                        chart_data["Nombre mostrado"] = np.where(
                            chart_data["Valor P"] <= 0.1,
                            "<b>" + chart_data["Variable explicativa (nivel)"].astype(str) + "</b>",
                            chart_data["Variable explicativa (nivel)"].astype(str))
                        chart_data["Texto del valor P"] = chart_data["Valor P"].map(lambda p: f"P={p:.4f}")
                        fig = px.bar(
                            chart_data, x=value_col, y="Nombre mostrado", orientation="h", text="Texto del valor P",
                            color="Dirección", color_discrete_map={"Positiva": "#2F6BFF", "Negativa": "#D1495B"},
                            hover_data={"Variable explicativa (nivel)": True, "Valor P": ":.4f", "Nombre mostrado": False,
                                        "Texto del valor P": False, "Valor de ordenación": False})
                        fig.update_traces(textposition="outside", cliponaxis=False)
                        fig.add_vline(x=reference, line_dash="dash", line_color=GRAY)
                        fig.update_yaxes(title=None)
                        chart_height = max(620, min(1400, 28 * len(chart_data) + 170))
                        fig = base_layout(fig, chart_height, f"{linear_name}: {value_col} y valor P")
                        fig.update_layout(margin=dict(l=60, r=110, t=50, b=50))
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("Se muestran las 30 magnitudes de efecto principales y todas las "
                                   "características con valor P ≤ 0,1. Los nombres en negrita tienen "
                                   "valor P ≤ 0,1. Los valores P corresponden a una prueba de Wald "
                                   "aproximada que considera la regularización. La dirección del "
                                   "coeficiente o de la razón de momios no implica causalidad.")
                        if task == "Clasificación":
                            note("La <b>razón de momios</b> indica por cuánto se multiplican los momios "
                                 "de que la variable objetivo sea 1, manteniendo constantes las demás "
                                 "condiciones. Un valor mayor que 1 indica aumento; menor que 1, "
                                 "disminución; y 1, ausencia de cambio. Por ejemplo, 1,20 equivale "
                                 "aproximadamente a un aumento del 20% y 0,80 a una disminución del 20%.")

# ---------------------------------------------------------------------
# ⑥ 因果探索 / ⑦ Detección de anomalías
# ---------------------------------------------------------------------
if True:
    if work is None:
        for tab in (T[5], T[6]):
            with tab:
                st.info("Primero cargue los datos.")
    else:
        nums = numeric_pool(work, schema, False)
        ratios = [c for c in work.columns if "÷" in c]
        bins_ = binary_pool(work, schema)
        e1, e2 = T[5], T[6]

        # ===== LiNGAM =====
        with e1:
            st.header("⑥ Exploración causal (LiNGAM)")
            if not HAS_LINGAM:
                st.warning("El paquete lingam no está instalado. Ejecute `pip install lingam`.")
            else:
                warn("Este gráfico es <b>una hipótesis estadística, no una prueba de causalidad</b>. "
                     "Los resultados dependen de factores de confusión no observados, la selección "
                     "de variables y los supuestos de linealidad y no gaussianidad. Verifique siempre "
                     "su coherencia con el conocimiento del negocio.")
                sel = st.multiselect("Variables a analizar (solo numéricas)", nums + bins_,
                                     default=nums[:min(7, len(nums))], key="lg_v")
                nboot = st.slider("Número de remuestreos bootstrap (0 para desactivar)", 0, 200, 50, 10)

                if len(sel) >= 3:
                    d = work[sel].apply(pd.to_numeric, errors="coerce").dropna()
                    st.caption(f"Casos completos: {len(d):,}")

                    with st.expander("Diagnóstico de supuestos (no gaussianidad)", expanded=True):
                        diag = pd.DataFrame({
                            "Variable": sel,
                            "Asimetría": [d[c].skew() for c in sel],
                            "Curtosis": [d[c].kurtosis() for c in sel],
                            "Valores únicos": [d[c].nunique() for c in sel]})
                        diag["Evaluación"] = np.where(
                            diag["Valores únicos"] <= 5, "Discreta (precaución)",
                            np.where(diag[["Asimetría", "Curtosis"]].abs().max(axis=1) < 0.5,
                                     "Cercana a normal (poco adecuada para LiNGAM)", "No gaussiana (adecuada)"))
                        st.dataframe(diag.style.format(
                            {"Asimetría": "{:+.2f}", "Curtosis": "{:+.2f}"}),
                            use_container_width=True, hide_index=True)

                    if len(d) > 30 and st.button("Ejecutar exploración causal", type="primary"):
                        with st.spinner("Estimando..."):
                            model = lingam.DirectLiNGAM()
                            model.fit(d.values)
                            adj = model.adjacency_matrix_
                            prob = None
                            if nboot > 0:
                                bs = model.bootstrap(d.values, n_sampling=nboot)
                                prob = bs.get_probabilities(min_causal_effect=0.01)

                        st.subheader("Estructura causal estimada")
                        edges = []
                        for i in range(len(sel)):
                            for j in range(len(sel)):
                                if i != j and abs(adj[i, j]) > 1e-6:
                                    edges.append({
                                        "Causa": sel[j], "Efecto": sel[i],
                                        "Coeficiente": adj[i, j],
                                        "Probabilidad de aparición": prob[i, j] if prob is not None else np.nan})
                        if not edges:
                            st.info("No se detectaron aristas relevantes.")
                        else:
                            ed = pd.DataFrame(edges).sort_values(
                                "Probabilidad de aparición" if prob is not None else "Coeficiente",
                                ascending=False, key=abs)
                            thr_p = st.slider("Umbral de probabilidad de aparición", 0.0, 1.0, 0.3, 0.05) \
                                if prob is not None else 0.0
                            show = ed[ed["Probabilidad de aparición"].fillna(1) >= thr_p] if prob is not None else ed

                            c1, c2 = st.columns([3, 2])
                            with c1:
                                mx = show["Coeficiente"].abs().max() or 1
                                def dot_escape(value):
                                    return str(value).replace("\\", "\\\\").replace('"', '\\"')

                                dot = [
                                    "digraph LiNGAM {",
                                    'rankdir=LR; graph [bgcolor="transparent", pad="0.2"];',
                                    'node [shape=box, style="rounded,filled", fillcolor="#E8F0EF", '
                                    'color="#15706B", fontname="Yu Gothic"];',
                                    'edge [fontname="Yu Gothic", fontsize=10];',
                                ]
                                for node_name in sel:
                                    q = dot_escape(node_name)
                                    dot.append(f'"{q}" [label="{q}"];')
                                for _, r in show.iterrows():
                                    cause = dot_escape(r["Causa"])
                                    effect = dot_escape(r["Efecto"])
                                    color = TEAL if r["Coeficiente"] > 0 else ACCENT
                                    width = 1 + 4 * abs(r["Coeficiente"]) / mx
                                    label = f'{r["Coeficiente"]:+.3f}'
                                    if prob is not None:
                                        label += f' / {r["Probabilidad de aparición"]:.0%}'
                                    dot.append(f'"{cause}" -> "{effect}" [color="{color}", '
                                               f'fontcolor="{color}", penwidth={width:.2f}, '
                                               f'label="{label}"];')
                                dot.append("}")
                                st.graphviz_chart("\n".join(dot), use_container_width=True)
                                st.caption("Visualización Graphviz. Verde indica efecto positivo y rojo, negativo. La etiqueta muestra el coeficiente (/ probabilidad de aparición).")
                            with c2:
                                st.dataframe(show.style.format(
                                    {"Coeficiente": "{:+.4f}", "Probabilidad de aparición": "{:.0%}"}),
                                    use_container_width=True, hide_index=True, height=470)

        # ===== Anómalo検知 =====
        with e2:
            st.header("⑦ Detección de anomalías")
            note("La detección utiliza <b>razones</b>, no solo importes, para evitar marcar como "
                 "anomalía una relación natural como «mayores ingresos, mayores ahorros».")
            pool = ratios + nums
            default_f = ratios[:6] if ratios else nums[:6]
            feats = st.multiselect("Características utilizadas", pool, default=default_f, key="an_f")
            cont = st.slider("Proporción esperada de anomalías", 0.01, 0.20, 0.05, 0.01)

            if len(feats) >= 2 and st.button("Ejecutar detección de anomalías", type="primary"):
                d = work[feats].apply(pd.to_numeric, errors="coerce")
                d = d.replace([np.inf, -np.inf], np.nan)
                keep = d.notna().all(axis=1)
                dd = d.loc[keep]
                st.caption(f"Casos analizados: {len(dd):,}")
                Z = StandardScaler().fit_transform(dd)
                iso = IsolationForest(contamination=cont, random_state=0, n_estimators=300)
                lab = iso.fit_predict(Z)
                score = -iso.score_samples(Z)

                out = work.loc[keep].copy()
                out["Puntuación de anomalía"] = score
                out["Evaluación"] = np.where(lab == -1, "Anómalo", "Normal")

                med = dd.median()
                sd = dd.std().replace(0, np.nan)
                zdev = (dd - med) / sd
                out["Factores principales"] = zdev.abs().apply(
                    lambda r: ", ".join(f"{c}({zdev.loc[r.name, c]:+.1f}σ)"
                                        for c in r.nlargest(3).index), axis=1)

                c1, c2 = st.columns([2, 3])
                with c1:
                    f = px.histogram(out, x="Puntuación de anomalía", nbins=50,
                                     color_discrete_sequence=[TEAL])
                    thr_v = np.quantile(score, 1 - cont)
                    f.add_vline(x=thr_v, line_dash="dash", line_color=ACCENT)
                    st.plotly_chart(base_layout(f, 360, "Distribución de la puntuación de anomalía"),
                                    use_container_width=True)
                with c2:
                    pca = PCA(n_components=2).fit_transform(Z)
                    f = px.scatter(x=pca[:, 0], y=pca[:, 1], color=out["Evaluación"],
                                   color_discrete_map={"Normal": "#C7D5D4", "Anómalo": ACCENT},
                                   labels={"x": "Primer componente principal", "y": "Segundo componente principal"}, opacity=0.7)
                    st.plotly_chart(base_layout(f, 360, "Distribución en el plano de componentes principales"),
                                    use_container_width=True)

                st.subheader("Mayores puntuaciones de anomalía")
                idc = cols_by_role(schema, "ID")
                showc = idc + [c for c in category_pool(work, schema)[:3]] + \
                    feats + ["Puntuación de anomalía", "Factores principales"]
                showc = [c for c in dict.fromkeys(showc) if c in out.columns]
                st.dataframe(out.nlargest(40, "Puntuación de anomalía")[showc],
                             use_container_width=True, height=420, hide_index=True)
                st.session_state["anomaly"] = out[["Puntuación de anomalía", "Evaluación"]]

st.markdown(f"<hr style='border-color:#DDD'><div style='font-size:0.78rem;color:{GRAY}'>"
            "Todos los resultados de la exploración automática son <b>hipótesis candidatas</b>. "
            "Revise el tamaño de la muestra y los intervalos de confianza, y contrástelos con el "
            "conocimiento del negocio antes de decidir.</div>", unsafe_allow_html=True)
