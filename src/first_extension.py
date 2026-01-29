import logging
import numpy as np
import pandas as pd
import os
import knime.extension as knext

# ???????? ??????????? ?? ???????? (??? ?? ??? ???????? ?? ?????? ????)
try:
    from scipy.stats.mstats import winsorize
    from scipy.stats import pearsonr, norm, rankdata, pointbiserialr
except ImportError:
    pass # ?? ?????????????? ?? imports ???????? ?? ?????????

LOGGER = logging.getLogger(__name__)

# --- ????????? ??? ?????? ??? ??????? ???? ---
# ???? ??????????? ??? ?? KNIME ?? ???? ??? ??????? 100%
HERE = os.path.dirname(os.path.abspath(__file__))

# --- Helper Functions ---
def _require_cols(df: pd.DataFrame, x_name: str, y_name: str):
    if x_name not in df.columns or y_name not in df.columns:
        raise ValueError(f"Columns not found. Available columns: {list(df.columns)}")

def _get_xy_numeric(df: pd.DataFrame, x_name: str, y_name: str):
    x = pd.to_numeric(df[x_name], errors="coerce").to_numpy()
    y = pd.to_numeric(df[y_name], errors="coerce").to_numpy()
    mask = ~np.isnan(x) & ~np.isnan(y)
    return x[mask], y[mask]

def _single_row_out(x_name: str, y_name: str, n: int, metric_name: str, value: float):
    return pd.DataFrame([{
        "x_column": x_name,
        "y_column": y_name,
        "n_used": float(n),
        metric_name: float(value),
    }])

def _single_row_schema(metric_name: str):
    return knext.Schema.from_columns([
        knext.Column(knext.string(), "x_column"),
        knext.Column(knext.string(), "y_column"),
        knext.Column(knext.double(), "n_used"),
        knext.Column(knext.double(), metric_name),
    ])

# --- Logic Functions ---
def _kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    if n < 2: return np.nan
    C = D = Tx = Ty = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = np.sign(x[i] - x[j])
            dy = np.sign(y[i] - y[j])
            if dx == 0 and dy == 0: continue
            if dx == 0: Tx += 1; continue
            if dy == 0: Ty += 1; continue
            if dx == dy: C += 1
            else: D += 1
    denom = np.sqrt((C + D + Tx) * (C + D + Ty))
    if denom == 0: return np.nan
    return float((C - D) / denom)

def _distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < 2: return np.nan
    a = np.abs(x.reshape(-1, 1) - x.reshape(1, -1))
    b = np.abs(y.reshape(-1, 1) - y.reshape(1, -1))
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
    dcov2 = (A * B).mean()
    dvarx = (A * A).mean()
    dvary = (B * B).mean()
    if dvarx <= 0 or dvary <= 0: return 0.0
    return float(np.sqrt(dcov2) / np.sqrt(np.sqrt(dvarx * dvary)))

def blomqvist_beta(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx = np.median(x)
    my = np.median(y)
    concordant = np.sum((x - mx) * (y - my) > 0)
    return float(4 * (concordant / len(x)) - 1)

def _tetrachoric_approx(x: np.ndarray, y: np.ndarray) -> float:
    x_uniques = np.unique(x)
    y_uniques = np.unique(y)
    if len(x_uniques) > 2 or len(y_uniques) > 2: return np.nan 
    x_binary = (x == x_uniques[-1]).astype(int) 
    y_binary = (y == y_uniques[-1]).astype(int)
    n00 = np.sum((x_binary == 0) & (y_binary == 0))
    n01 = np.sum((x_binary == 0) & (y_binary == 1))
    n10 = np.sum((x_binary == 1) & (y_binary == 0))
    n11 = np.sum((x_binary == 1) & (y_binary == 1))
    if n00 == 0 or n01 == 0 or n10 == 0 or n11 == 0:
        n00 += 0.5; n01 += 0.5; n10 += 0.5; n11 += 0.5
    ratio = (n00 * n11) / (n01 * n10)
    rtet = np.cos(np.pi / (1 + np.sqrt(ratio)))
    return float(rtet)

def _gaussian_rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    r_x = rankdata(x)
    r_y = rankdata(y)
    n = len(x)
    g_x = norm.ppf(r_x / (n + 1))
    g_y = norm.ppf(r_y / (n + 1))
    corr, _ = pearsonr(g_x, g_y)
    return float(corr)

def _calculate_mic_approx(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n < 2: return 0.0
    max_grid = int(n ** 0.6)
    if max_grid < 2: max_grid = 2
    max_mic = 0.0
    grids = set()
    for i in range(2, max_grid + 1):
        grids.add((i, i))
        if i * 2 <= max_grid: grids.add((i, i*2)); grids.add((i*2, i))
    for nx, ny in grids:
        try: hist, _, _ = np.histogram2d(x, y, bins=[nx, ny])
        except Exception: continue
        n_points = np.sum(hist)
        if n_points == 0: continue
        p_xy = hist / n_points
        p_x = np.sum(p_xy, axis=1); p_y = np.sum(p_xy, axis=0)
        mask = p_xy > 0
        if not np.any(mask): mi = 0.0
        else:
            p_x_y_indep = np.outer(p_x, p_y)
            valid_denom = (p_x_y_indep[mask] > 0)
            if not np.all(valid_denom): mi = 0.0
            else:
                 term = p_xy[mask] * np.log(p_xy[mask] / p_x_y_indep[mask])
                 mi = np.sum(term)
        denom = np.log(min(nx, ny))
        if denom > 0:
            current_mic = mi / denom
            if current_mic > max_mic: max_mic = current_mic
    return float(max_mic)

def _hoeffdings_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x); y = np.asarray(y)
    n = len(x)
    if n < 5: return np.nan
    R = rankdata(x); S = rankdata(y)
    Q = np.zeros(n)
    for i in range(n):
        less_x = x < x[i]; less_y = y < y[i]
        Q[i] = 1 + np.sum(less_x & less_y)
        eq_x = x == x[i]; eq_y = y == y[i]
        Q[i] += 0.25 * (np.sum(eq_x & eq_y) - 1)
        Q[i] += 0.5 * np.sum(less_x & eq_y)
        Q[i] += 0.5 * np.sum(eq_x & less_y)
    D1 = np.sum((Q - 1) * (Q - 2))
    D2 = np.sum((R - 1) * (R - 2) * (S - 1) * (S - 2))
    D3 = np.sum((R - 2) * (S - 2) * (Q - 1))
    numerator = (n - 2) * (n - 3) * D1 + D2 - 2 * (n - 2) * D3
    denominator = n * (n - 1) * (n - 2) * (n - 3) * (n - 4)
    if denominator == 0: return np.nan
    return 30 * numerator / denominator

def _biserial_corr(x_binary: np.ndarray, y_continuous: np.ndarray) -> float:
    r_pb, _ = pointbiserialr(x_binary, y_continuous)
    n = len(x_binary)
    if n < 2: return np.nan
    uniques = np.unique(x_binary)
    if len(uniques) != 2: return np.nan 
    p = np.mean(x_binary == uniques[1]) 
    q = 1 - p
    z = norm.ppf(q)
    y = norm.pdf(z)
    if y == 0: return np.nan
    r_b = r_pb * np.sqrt(p * q) / y
    return float(r_b)

# --- Category Definition ---
# ? ??????? ??????: path="/community" ??? ????? ?????????
category_icon = os.path.join(HERE, "blomqvist.png") 

main_category = knext.category(
    path="/community", 
    level_id="correlation_nodes_group", 
    name="Correlation Nodes",     
    description="Statistical Correlation Nodes",
    icon=category_icon,
)

# --- Nodes ---

@knext.node(
    name="Blomqvist's Beta", 
    node_type=knext.NodeType.MANIPULATOR, 
    icon_path=os.path.join(HERE, "blomqvist.png"), 
    category=main_category
)
@knext.input_table(name="Input table", description="Input table")
@knext.output_table(name="Output table", description="Result table")
class BlomqvistNode:
    x_col = knext.ColumnParameter("X column", "Select X column", port_index=0)
    y_col = knext.ColumnParameter("Y column", "Select Y column", port_index=0)

    def configure(self, config_context, input_table_schema):
        return knext.Schema.from_columns([knext.Column(knext.string(), "x_column"), knext.Column(knext.string(), "y_column"), knext.Column(knext.double(), "blomqvist_beta")])

    def execute(self, exec_context, input_table):
        df = input_table.to_pandas()
        x_name = self.x_col
        y_name = self.y_col
        _require_cols(df, x_name, y_name)
        x = pd.to_numeric(df[x_name], errors="coerce").to_numpy()
        y = pd.to_numeric(df[y_name], errors="coerce").to_numpy()
        mask = ~np.isnan(x) & ~np.isnan(y); x, y = x[mask], y[mask]
        beta = blomqvist_beta(x, y)
        out = pd.DataFrame([{"x_column": x_name, "y_column": y_name, "blomqvist_beta": beta}])
        return knext.Table.from_pandas(out)

@knext.node(
    name="Kendall Correlation (tau-b)", 
    node_type=knext.NodeType.MANIPULATOR, 
    icon_path=os.path.join(HERE, "kendall.png"), 
    category=main_category
)
@knext.input_table(name="Input table", description="Input table")
@knext.output_table(name="Output table", description="Single-row result table")
class KendallTauNode:
    x_col = knext.ColumnParameter("X column", "Select X column", port_index=0)
    y_col = knext.ColumnParameter("Y column", "Select Y column", port_index=0)

    def configure(self, config_context, input_table_schema):
        return knext.Schema.from_columns([knext.Column(knext.string(), "x_column"), knext.Column(knext.string(), "y_column"), knext.Column(knext.double(), "n_used"), knext.Column(knext.double(), "kendall_tau_b")])

    def execute(self, exec_context, input_table):
        df = input_table.to_pandas()
        x_name = self.x_col
        y_name = self.y_col
        _require_cols(df, x_name, y_name)
        x = pd.to_numeric(df[x_name], errors="coerce").to_numpy()
        y = pd.to_numeric(df[y_name], errors="coerce").to_numpy()
        mask = ~np.isnan(x) & ~np.isnan(y); x, y = x[mask], y[mask]
        if len(x) < 2: raise ValueError("Need at least 2 valid numeric pairs.")
        tau = _kendall_tau_b(x, y)
        out = pd.DataFrame([{"x_column": x_name, "y_column": y_name, "n_used": float(len(x)), "kendall_tau_b": float(tau)}])
        return knext.Table.from_pandas(out)

@knext.node(
    name="Distance Correlation", 
    node_type=knext.NodeType.MANIPULATOR, 
    icon_path=os.path.join(HERE, "distance.png"), 
    category=main_category
)
@knext.input_table(name="Input table", description="Input table")
@knext.output_table(name="Output table", description="Single-row result table")
class DistanceCorrelationNode:
    x_col = knext.ColumnParameter("X column", "Select X column", port_index=0)
    y_col = knext.ColumnParameter("Y column", "Select Y column", port_index=0)

    def configure(self, config_context, input_table_schema): return _single_row_schema("distance_corr")
    
    def execute(self, exec_context, input_table):
        df = input_table.to_pandas()
        x_name = self.x_col
        y_name = self.y_col
        _require_cols(df, x_name, y_name)
        x, y = _get_xy_numeric(df, x_name, y_name)
        if len(x) == 0: raise ValueError("No valid numeric pairs.")
        val = _distance_correlation(x, y)
        return knext.Table.from_pandas(_single_row_out(x_name, y_name, len(x), "distance_corr", val))

@knext.node(
    name="Winsorised Correlation", 
    node_type=knext.NodeType.MANIPULATOR, 
    icon_path=os.path.join(HERE, "winsor.png"), 
    category=main_category
)
@knext.input_table(name="Input table", description="Input table with numeric columns")
@knext.output_table(name="Output table", description="Single-row result with Winsorised Pearson correlation")
class WinsorisedCorrelationNode:
    x_col = knext.ColumnParameter("X column", "Select X column", port_index=0)
    y_col = knext.ColumnParameter("Y column", "Select Y column", port_index=0)
    limit = knext.DoubleParameter("Winsorisation limit", "Fraction to winsorize", default_value=0.05, min_value=0.0, max_value=0.49)

    def configure(self, config_context, input_table_schema):
        return knext.Schema.from_columns([knext.Column(knext.double(), "winsorised_corr"), knext.Column(knext.int64(), "n"), knext.Column(knext.string(), "x_col"), knext.Column(knext.string(), "y_col"), knext.Column(knext.double(), "winsor_limit")])
    
    def execute(self, exec_context, input_table):
        df = input_table.to_pandas()
        x_name = self.x_col
        y_name = self.y_col
        _require_cols(df, x_name, y_name)
        x, y = _get_xy_numeric(df, x_name, y_name)
        if len(x) < 3: raise ValueError("Winsorised correlation requires at least 3 valid numeric pairs.")
        x_w = np.asarray(winsorize(x, limits=self.limit), dtype=float); y_w = np.asarray(winsorize(y, limits=self.limit), dtype=float)
        corr, _ = pearsonr(x_w, y_w)
        out = pd.DataFrame({"winsorised_corr": [float(corr)], "n": [int(len(x_w))], "x_col": [x_name], "y_col": [y_name], "winsor_limit": [float(self.limit)]})
        return knext.Table.from_pandas(out)

@knext.node(
    name="Tetrachoric Correlation", 
    node_type=knext.NodeType.MANIPULATOR, 
    icon_path=os.path.join(HERE, "tetrachoric.png"), 
    category=main_category
)
@knext.input_table(name="Input table", description="Input table with binary columns")
@knext.output_table(name="Output table", description="Single-row result with Tetrachoric correlation")
class TetrachoricCorrelationNode:
    x_col = knext.ColumnParameter("X column (Binary)", "Select Binary X", port_index=0)
    y_col = knext.ColumnParameter("Y column (Binary)", "Select Binary Y", port_index=0)

    def configure(self, config_context, input_table_schema):
        return knext.Schema.from_columns([knext.Column(knext.double(), "tetrachoric_corr"), knext.Column(knext.int64(), "n"), knext.Column(knext.string(), "x_col"), knext.Column(knext.string(), "y_col")])
    
    def execute(self, exec_context, input_table):
        df = input_table.to_pandas()
        x_name = self.x_col
        y_name = self.y_col
        _require_cols(df, x_name, y_name)
        temp_df = df[[x_name, y_name]].dropna()
        if len(temp_df) < 2: raise ValueError("Not enough data after removing missing values.")
        x = temp_df[x_name].values; y = temp_df[y_name].values
        if len(np.unique(x)) > 2 or len(np.unique(y)) > 2: raise ValueError(f"Columns must be binary.")
        val = _tetrachoric_approx(x, y)
        out = pd.DataFrame({"tetrachoric_corr": [val], "n": [int(len(x))], "x_col": [x_name], "y_col": [y_name]})
        return knext.Table.from_pandas(out)

@knext.node(
    name="Gaussian Rank Correlation", 
    node_type=knext.NodeType.MANIPULATOR, 
    icon_path=os.path.join(HERE, "gaussian.png"), 
    category=main_category
)
@knext.input_table(name="Input table", description="Input table with numeric columns")
@knext.output_table(name="Output table", description="Single-row result with Gaussian Rank Correlation")
class GaussianRankCorrelationNode:
    x_col = knext.ColumnParameter("X column", "Select X column", port_index=0)
    y_col = knext.ColumnParameter("Y column", "Select Y column", port_index=0)

    def configure(self, config_context, input_table_schema):
        return knext.Schema.from_columns([knext.Column(knext.double(), "gaussian_rank_corr"), knext.Column(knext.int64(), "n"), knext.Column(knext.string(), "x_col"), knext.Column(knext.string(), "y_col")])
    
    def execute(self, exec_context, input_table):
        df = input_table.to_pandas()
        x_name = self.x_col
        y_name = self.y_col
        _require_cols(df, x_name, y_name)
        x, y = _get_xy_numeric(df, x_name, y_name)
        if len(x) < 2: raise ValueError("Not enough data points.")
        val = _gaussian_rank_correlation(x, y)
        out = pd.DataFrame({"gaussian_rank_corr": [val], "n": [int(len(x))], "x_col": [x_name], "y_col": [y_name]})
        return knext.Table.from_pandas(out)

@knext.node(
    name="Maximal Info Coefficient (MIC)", 
    node_type=knext.NodeType.MANIPULATOR, 
    icon_path=os.path.join(HERE, "mic.png"), 
    category=main_category
)
@knext.input_table(name="Input table", description="Input table with numeric columns")
@knext.output_table(name="Output table", description="Single-row result with MIC Score")
class MICNode:
    x_col = knext.ColumnParameter("X column", "Select X column", port_index=0)
    y_col = knext.ColumnParameter("Y column", "Select Y column", port_index=0)

    def configure(self, config_context, input_table_schema):
        return knext.Schema.from_columns([knext.Column(knext.double(), "mic_score"), knext.Column(knext.int64(), "n"), knext.Column(knext.string(), "x_col"), knext.Column(knext.string(), "y_col")])
    
    def execute(self, exec_context, input_table):
        df = input_table.to_pandas()
        x_name = self.x_col
        y_name = self.y_col
        _require_cols(df, x_name, y_name)
        x, y = _get_xy_numeric(df, x_name, y_name)
        if len(x) < 2: raise ValueError("Not enough data points.")
        val = _calculate_mic_approx(x, y)
        out = pd.DataFrame({"mic_score": [val], "n": [int(len(x))], "x_col": [x_name], "y_col": [y_name]})
        return knext.Table.from_pandas(out)

@knext.node(
    name="Hoeffding's D", 
    node_type=knext.NodeType.MANIPULATOR, 
    icon_path=os.path.join(HERE, "hoeffding.png"), 
    category=main_category
)
@knext.input_table(name="Input table", description="Input table with numeric columns")
@knext.output_table(name="Output table", description="Single-row result with Hoeffding's D")
class HoeffdingDNode:
    x_col = knext.ColumnParameter("X column", "Select X column", port_index=0)
    y_col = knext.ColumnParameter("Y column", "Select Y column", port_index=0)

    def configure(self, config_context, input_table_schema):
        return knext.Schema.from_columns([knext.Column(knext.double(), "hoeffding_d"), knext.Column(knext.int64(), "n"), knext.Column(knext.string(), "x_col"), knext.Column(knext.string(), "y_col")])
    
    def execute(self, exec_context, input_table):
        df = input_table.to_pandas()
        x_name = self.x_col
        y_name = self.y_col
        _require_cols(df, x_name, y_name)
        x, y = _get_xy_numeric(df, x_name, y_name)
        if len(x) < 5: raise ValueError("Hoeffding's D requires at least 5 data points.")
        val = _hoeffdings_d(x, y)
        out = pd.DataFrame({"hoeffding_d": [val], "n": [int(len(x))], "x_col": [x_name], "y_col": [y_name]})
        return knext.Table.from_pandas(out)

@knext.node(
    name="Point-Biserial Correlation", 
    node_type=knext.NodeType.MANIPULATOR, 
    icon_path=os.path.join(HERE, "point_biserial.png"), 
    category=main_category
)
@knext.input_table(name="Input table", description="Input table")
@knext.output_table(name="Output table", description="Result table")
class PointBiserialNode:
    binary_col = knext.ColumnParameter("Binary column", "Select Binary Column", port_index=0)
    continuous_col = knext.ColumnParameter("Continuous column", "Select Numeric Column", port_index=0)

    def configure(self, config_context, input_table_schema):
        return knext.Schema.from_columns([
            knext.Column(knext.double(), "point_biserial_r"),
            knext.Column(knext.double(), "p_value"),
            knext.Column(knext.int64(), "n"),
            knext.Column(knext.string(), "binary_col"),
            knext.Column(knext.string(), "continuous_col")
        ])

    def execute(self, exec_context, input_table):
        df = input_table.to_pandas()
        bin_name = self.binary_col
        cont_name = self.continuous_col
        
        # Clean NaNs
        temp = df[[bin_name, cont_name]].dropna()
        if len(temp) < 2: raise ValueError("Not enough data.")
        
        # Check binary
        if len(temp[bin_name].unique()) != 2:
             raise ValueError(f"Column '{bin_name}' must be binary (have exactly 2 unique values).")
        
        uniques = temp[bin_name].unique()
        x_bin = np.where(temp[bin_name] == uniques[1], 1, 0)
        y_cont = pd.to_numeric(temp[cont_name], errors='coerce').to_numpy()
        
        mask = ~np.isnan(y_cont)
        x_bin = x_bin[mask]
        y_cont = y_cont[mask]
        
        if len(y_cont) < 2: raise ValueError("Not enough valid numeric data.")

        r_pb, p_val = pointbiserialr(x_bin, y_cont)

        out = pd.DataFrame({
            "point_biserial_r": [float(r_pb)],
            "p_value": [float(p_val)],
            "n": [int(len(x_bin))],
            "binary_col": [bin_name],
            "continuous_col": [cont_name]
        })
        return knext.Table.from_pandas(out)

@knext.node(
    name="Biserial Correlation", 
    node_type=knext.NodeType.MANIPULATOR, 
    icon_path=os.path.join(HERE, "biserial.png"), 
    category=main_category
)
@knext.input_table(name="Input table", description="Input table")
@knext.output_table(name="Output table", description="Result table")
class BiserialNode:
    binary_col = knext.ColumnParameter("Binary column", "Select Binary Column", port_index=0)
    continuous_col = knext.ColumnParameter("Continuous column", "Select Numeric Column", port_index=0)

    def configure(self, config_context, input_table_schema):
        return knext.Schema.from_columns([
            knext.Column(knext.double(), "biserial_r"),
            knext.Column(knext.int64(), "n"),
            knext.Column(knext.string(), "binary_col"),
            knext.Column(knext.string(), "continuous_col")
        ])

    def execute(self, exec_context, input_table):
        df = input_table.to_pandas()
        bin_name = self.binary_col
        cont_name = self.continuous_col
        _require_cols(df, bin_name, cont_name)
        
        temp = df[[bin_name, cont_name]].dropna()
        if len(temp) < 2: raise ValueError("Not enough data.")
        
        if len(temp[bin_name].unique()) != 2:
             raise ValueError(f"Column '{bin_name}' must be binary.")
             
        uniques = temp[bin_name].unique()
        x_bin = np.where(temp[bin_name] == uniques[1], 1, 0)
        y_cont = pd.to_numeric(temp[cont_name], errors='coerce').to_numpy()
        
        mask = ~np.isnan(y_cont)
        x_bin = x_bin[mask]
        y_cont = y_cont[mask]
        
        val = _biserial_corr(x_bin, y_cont)
        
        out = pd.DataFrame({
            "biserial_r": [val],
            "n": [int(len(x_bin))],
            "binary_col": [bin_name],
            "continuous_col": [cont_name]
        })
        return knext.Table.from_pandas(out)