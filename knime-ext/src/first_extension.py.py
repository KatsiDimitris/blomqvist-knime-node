import logging
import os
import numpy as np
import pandas as pd
import knime.extension as knext

try:
    from scipy.stats.mstats import winsorize
    from scipy.stats import pearsonr, norm, rankdata, pointbiserialr
    from minepy import MINE  
except ImportError:
    pass
# Safe imports for scientific libraries to prevent crashes if a package is missing
try:
    from scipy.stats.mstats import winsorize
    from scipy.stats import pearsonr, norm, rankdata, pointbiserialr
except ImportError:
    pass 

LOGGER = logging.getLogger(__name__)

# --- PATH CONFIGURATION ---
# Determine the absolute path of the current file (.../src/first_extension.py)
HERE = os.path.dirname(os.path.abspath(__file__))

# Go up one level (parent directory of 'src') and find the 'icons' folder
PROJECT_ROOT = os.path.dirname(HERE)
ICONS_PATH = os.path.join(PROJECT_ROOT, "icons")

# --- Helper Functions ---

def _require_cols(df: pd.DataFrame, x_name: str, y_name: str):
    """Checks if the selected columns exist in the DataFrame."""
    if x_name not in df.columns or y_name not in df.columns:
        raise ValueError(f"Selected columns not found. Available columns: {list(df.columns)}")

def _get_xy_numeric(df: pd.DataFrame, x_name: str, y_name: str):
    """Converts columns to numeric arrays and removes rows containing NaNs."""
    x = pd.to_numeric(df[x_name], errors="coerce").to_numpy()
    y = pd.to_numeric(df[y_name], errors="coerce").to_numpy()
    mask = ~np.isnan(x) & ~np.isnan(y)
    return x[mask], y[mask]

def _single_row_out(x_name: str, y_name: str, n: int, metric_name: str, value: float):
    """Creates a Pandas DataFrame for a single result row with human-readable headers."""
    return pd.DataFrame([{
        "Variable X": x_name,
        "Variable Y": y_name,
        "Sample Size (N)": float(n),
        metric_name: float(value),
    }])

def _single_row_schema(metric_name: str):
    """Defines the KNIME output schema with human-readable column names."""
    return knext.Schema.from_columns([
        knext.Column(knext.string(), "Variable X"),
        knext.Column(knext.string(), "Variable Y"),
        knext.Column(knext.double(), "Sample Size (N)"),
        knext.Column(knext.double(), metric_name),
    ])

# --- Mathematical Logic Functions ---

def blomqvist_beta(x, y):
    """Calculates Blomqvist's Beta (Medial Correlation)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx = np.median(x)
    my = np.median(y)
    concordant = np.sum((x - mx) * (y - my) > 0)
    discordant = np.sum((x - mx) * (y - my) < 0)
    
    if concordant + discordant == 0:
        return np.nan
        
    return float(concordant - discordant) / (concordant + discordant)
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
    """Calculates Gaussian Rank Correlation according to Boudt et al. (2012)."""
    n = len(x)
    r_x = rankdata(x)
    r_y = rankdata(y)
    
    g_x = norm.ppf(r_x / (n + 1))
    g_y = norm.ppf(r_y / (n + 1))
    
    numerator = np.sum(g_x * g_y)
    
    ideal_ranks = np.arange(1, n + 1)
    ideal_scores = norm.ppf(ideal_ranks / (n + 1))
    denominator = np.sum(ideal_scores ** 2)
    
    return float(numerator / denominator)

def _calculate_mic_exact(x: np.ndarray, y: np.ndarray) -> float:
    """Calculates the exact MIC using the minepy library."""
    n = len(x)
    if n < 2: return 0.0    
    mine = MINE(alpha=0.6, c=15, est="mic_approx")
    mine.compute_score(x, y)
    
    return float(mine.mic())

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

category_icon = os.path.join(ICONS_PATH, "blomqvist.png") 

main_category = knext.category(
    path="/community", 
    level_id="correlation_nodes", 
    name="Correlation Nodes",     
    description="A collection of advanced statistical correlation nodes implemented in Python.",
    icon=category_icon,
)

# --- Nodes ---

@knext.node(
    name="Blomqvist's Beta", 
    node_type=knext.NodeType.MANIPULATOR, 
    icon_path=os.path.join(ICONS_PATH, "blomqvist.png"), 
    category=main_category
)
@knext.input_table(name="Input Data", description="Table containing the data to be analyzed.")
@knext.output_table(name="Result Data", description="Table containing the calculated Blomqvist's Beta correlation.")
class BlomqvistNode:
    """
    Calculates Blomqvist's Beta (Medial Correlation).
    
    This node computes a non-parametric measure of dependence between two variables based on their medians. It is highly robust to outliers.
    """
    x_col = knext.ColumnParameter("X Variable", "Select the first numeric column for the correlation.", port_index=0)
    y_col = knext.ColumnParameter("Y Variable", "Select the second numeric column for the correlation.", port_index=0)

    def configure(self, config_context, input_table_schema):
        return knext.Schema.from_columns([
            knext.Column(knext.string(), "Variable X"), 
            knext.Column(knext.string(), "Variable Y"), 
            knext.Column(knext.double(), "Blomqvist's Beta")
        ])

    def execute(self, exec_context, input_table):
        df = input_table.to_pandas()
        x_name = self.x_col
        y_name = self.y_col
        _require_cols(df, x_name, y_name)
        x = pd.to_numeric(df[x_name], errors="coerce").to_numpy()
        y = pd.to_numeric(df[y_name], errors="coerce").to_numpy()
        mask = ~np.isnan(x) & ~np.isnan(y); x, y = x[mask], y[mask]
        beta = blomqvist_beta(x, y)
        out = pd.DataFrame([{
            "Variable X": x_name, 
            "Variable Y": y_name, 
            "Blomqvist's Beta": beta
        }])
        return knext.Table.from_pandas(out)


@knext.node(
    name="Kendall Correlation (tau-b)", 
    node_type=knext.NodeType.MANIPULATOR, 
    icon_path=os.path.join(ICONS_PATH, "kendall.png"), 
    category=main_category
)
@knext.input_table(name="Input Data", description="Table containing numeric data.")
@knext.output_table(name="Result Data", description="Table containing Kendall's Tau-b value.")
class KendallTauNode:
    """
    Calculates Kendall's Tau-b rank correlation coefficient.
    
    It measures the ordinal association between two quantities and is adjusted for ties.
    """
    x_col = knext.ColumnParameter("X Variable", "Select the first numeric column.", port_index=0)
    y_col = knext.ColumnParameter("Y Variable", "Select the second numeric column.", port_index=0)

    def configure(self, config_context, input_table_schema):
        return knext.Schema.from_columns([
            knext.Column(knext.string(), "Variable X"), 
            knext.Column(knext.string(), "Variable Y"), 
            knext.Column(knext.double(), "Sample Size (N)"), 
            knext.Column(knext.double(), "Kendall's Tau-b")
        ])

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
        out = pd.DataFrame([{
            "Variable X": x_name, 
            "Variable Y": y_name, 
            "Sample Size (N)": float(len(x)), 
            "Kendall's Tau-b": float(tau)
        }])
        return knext.Table.from_pandas(out)


@knext.node(
    name="Distance Correlation", 
    node_type=knext.NodeType.MANIPULATOR, 
    icon_path=os.path.join(ICONS_PATH, "distance.png"), 
    category=main_category
)
@knext.input_table(name="Input Data", description="Table containing numeric data.")
@knext.output_table(name="Result Data", description="Table containing the Distance Correlation value.")
class DistanceCorrelationNode:
    """
    Calculates Distance Correlation.
    
    This measure detects both linear and non-linear dependencies between two variables. A value of zero implies independence.
    """
    x_col = knext.ColumnParameter("X Variable", "Select the first numeric column.", port_index=0)
    y_col = knext.ColumnParameter("Y Variable", "Select the second numeric column.", port_index=0)

    def configure(self, config_context, input_table_schema): 
        return _single_row_schema("Distance Correlation")
    
    def execute(self, exec_context, input_table):
        df = input_table.to_pandas()
        x_name = self.x_col
        y_name = self.y_col
        _require_cols(df, x_name, y_name)
        x, y = _get_xy_numeric(df, x_name, y_name)
        if len(x) == 0: raise ValueError("No valid numeric pairs.")
        val = _distance_correlation(x, y)
        return knext.Table.from_pandas(_single_row_out(x_name, y_name, len(x), "Distance Correlation", val))


@knext.node(
    name="Winsorised Correlation", 
    node_type=knext.NodeType.MANIPULATOR, 
    icon_path=os.path.join(ICONS_PATH, "winsor.png"), 
    category=main_category
)
@knext.input_table(name="Input Data", description="Table containing numeric data.")
@knext.output_table(name="Result Data", description="Table containing the Winsorised Correlation.")
class WinsorisedCorrelationNode:
    """
    Calculates Pearson correlation after Winsorizing the data.
    
    This reduces the effect of outliers by clipping extreme values to specified percentiles before calculation.
    """
    x_col = knext.ColumnParameter("X Variable", "Select the first numeric column.", port_index=0)
    y_col = knext.ColumnParameter("Y Variable", "Select the second numeric column.", port_index=0)
    limit = knext.DoubleParameter("Winsorisation limit", "The fraction of data to winsorize on each tail (e.g., 0.05 for 5%).", default_value=0.05, min_value=0.0, max_value=0.49)

    def configure(self, config_context, input_table_schema):
        return knext.Schema.from_columns([
            knext.Column(knext.double(), "Winsorised Correlation"), 
            knext.Column(knext.int64(), "Sample Size (N)"), 
            knext.Column(knext.string(), "Variable X"), 
            knext.Column(knext.string(), "Variable Y"), 
            knext.Column(knext.double(), "Winsor Limit")
        ])
    
    def execute(self, exec_context, input_table):
        df = input_table.to_pandas()
        x_name = self.x_col
        y_name = self.y_col
        _require_cols(df, x_name, y_name)
        x, y = _get_xy_numeric(df, x_name, y_name)
        if len(x) < 3: raise ValueError("Winsorised correlation requires at least 3 valid numeric pairs.")
        x_w = np.asarray(winsorize(x, limits=self.limit), dtype=float); y_w = np.asarray(winsorize(y, limits=self.limit), dtype=float)
        corr, _ = pearsonr(x_w, y_w)
        out = pd.DataFrame({
            "Winsorised Correlation": [float(corr)], 
            "Sample Size (N)": [int(len(x_w))], 
            "Variable X": [x_name], 
            "Variable Y": [y_name], 
            "Winsor Limit": [float(self.limit)]
        })
        return knext.Table.from_pandas(out)
@knext.node(
    name="Tetrachoric Correlation", 
    node_type=knext.NodeType.MANIPULATOR, 
    icon_path=os.path.join(ICONS_PATH, "tetrachoric.png"), 
    category=main_category
)
@knext.input_table(name="Input Data", description="Table containing binary data.")
@knext.output_table(name="Result Data", description="Table containing the Tetrachoric Correlation.")
class TetrachoricCorrelationNode:
    """
    Calculates Tetrachoric Correlation.
    
    Used when both variables are binary (dichotomous) but assumed to represent underlying continuous normal distributions.
    """
    x_col = knext.ColumnParameter("X Variable (Binary)", "Select the first binary column.", port_index=0)
    y_col = knext.ColumnParameter("Y Variable (Binary)", "Select the second binary column.", port_index=0)

    def configure(self, config_context, input_table_schema):
        return knext.Schema.from_columns([
            knext.Column(knext.double(), "Tetrachoric Correlation"), 
            knext.Column(knext.int64(), "Sample Size (N)"), 
            knext.Column(knext.string(), "Variable X"), 
            knext.Column(knext.string(), "Variable Y")
        ])
    
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
        out = pd.DataFrame({
            "Tetrachoric Correlation": [val], 
            "Sample Size (N)": [int(len(x))], 
            "Variable X": [x_name], 
            "Variable Y": [y_name]
        })
        return knext.Table.from_pandas(out)


@knext.node(
    name="Gaussian Rank Correlation", 
    node_type=knext.NodeType.MANIPULATOR, 
    icon_path=os.path.join(ICONS_PATH, "gaussian.png"), 
    category=main_category
)
@knext.input_table(name="Input Data", description="Table containing numeric data.")
@knext.output_table(name="Result Data", description="Table containing the Gaussian Rank Correlation.")
class GaussianRankCorrelationNode:
    """
    Calculates Gaussian Rank Correlation.
    
    Transforms values into ranks, maps them to Gaussian scores (using the Van der Waerden transformation), and calculates the Pearson correlation.
    """
    x_col = knext.ColumnParameter("X Variable", "Select the first numeric column.", port_index=0)
    y_col = knext.ColumnParameter("Y Variable", "Select the second numeric column.", port_index=0)

    def configure(self, config_context, input_table_schema):
        return knext.Schema.from_columns([
            knext.Column(knext.double(), "Gaussian Rank Correlation"), 
            knext.Column(knext.int64(), "Sample Size (N)"), 
            knext.Column(knext.string(), "Variable X"), 
            knext.Column(knext.string(), "Variable Y")
        ])
    
    def execute(self, exec_context, input_table):
        df = input_table.to_pandas()
        x_name = self.x_col
        y_name = self.y_col
        _require_cols(df, x_name, y_name)
        x, y = _get_xy_numeric(df, x_name, y_name)
        if len(x) < 2: raise ValueError("Not enough data points.")
        val = _gaussian_rank_correlation(x, y)
        out = pd.DataFrame({
            "Gaussian Rank Correlation": [val], 
            "Sample Size (N)": [int(len(x))], 
            "Variable X": [x_name], 
            "Variable Y": [y_name]
        })
        return knext.Table.from_pandas(out)


@knext.node(
    name="Maximal Info Coefficient (MIC)", 
    node_type=knext.NodeType.MANIPULATOR, 
    icon_path=os.path.join(ICONS_PATH, "mic.png"), 
    category=main_category
)
@knext.input_table(name="Input Data", description="Table containing numeric data.")
@knext.output_table(name="Result Data", description="Table containing the MIC Score.")
class MICNode:
    """
    Calculates the Maximal Information Coefficient (MIC).
    
    This index is capable of detecting both linear and complex non-linear relationships. Returns a value between 0 and 1.
    """
    x_col = knext.ColumnParameter("X Variable", "Select the first numeric column.", port_index=0)
    y_col = knext.ColumnParameter("Y Variable", "Select the second numeric column.", port_index=0)

    def configure(self, config_context, input_table_schema):
        return knext.Schema.from_columns([
            knext.Column(knext.double(), "MIC Score"), 
            knext.Column(knext.int64(), "Sample Size (N)"), 
            knext.Column(knext.string(), "Variable X"), 
            knext.Column(knext.string(), "Variable Y")
        ])
    
    def execute(self, exec_context, input_table):
        df = input_table.to_pandas()
        x_name = self.x_col
        y_name = self.y_col
        _require_cols(df, x_name, y_name)
        x, y = _get_xy_numeric(df, x_name, y_name)
        if len(x) < 2: raise ValueError("Not enough data points.")
        val = _calculate_mic_exact(x, y)
        out = pd.DataFrame({
            "MIC Score": [val], 
            "Sample Size (N)": [int(len(x))], 
            "Variable X": [x_name], 
            "Variable Y": [y_name]
        })
        return knext.Table.from_pandas(out)


@knext.node(
    name="Hoeffding's D", 
    node_type=knext.NodeType.MANIPULATOR, 
    icon_path=os.path.join(ICONS_PATH, "hoeffding.png"), 
    category=main_category
)
@knext.input_table(name="Input Data", description="Table containing numeric data.")
@knext.output_table(name="Result Data", description="Table containing Hoeffding's D value.")
class HoeffdingDNode:
    """
    Calculates Hoeffding's D.
    
    A non-parametric measure that detects broad deviations from independence between two variables.
    """
    x_col = knext.ColumnParameter("X Variable", "Select the first numeric column.", port_index=0)
    y_col = knext.ColumnParameter("Y Variable", "Select the second numeric column.", port_index=0)

    def configure(self, config_context, input_table_schema):
        return knext.Schema.from_columns([
            knext.Column(knext.double(), "Hoeffding's D"), 
            knext.Column(knext.int64(), "Sample Size (N)"), 
            knext.Column(knext.string(), "Variable X"), 
            knext.Column(knext.string(), "Variable Y")
        ])
    
    def execute(self, exec_context, input_table):
        df = input_table.to_pandas()
        x_name = self.x_col
        y_name = self.y_col
        _require_cols(df, x_name, y_name)
        x, y = _get_xy_numeric(df, x_name, y_name)
        if len(x) < 5: raise ValueError("Hoeffding's D requires at least 5 data points.")
        val = _hoeffdings_d(x, y)
        out = pd.DataFrame({
            "Hoeffding's D": [val], 
            "Sample Size (N)": [int(len(x))], 
            "Variable X": [x_name], 
            "Variable Y": [y_name]
        })
        return knext.Table.from_pandas(out)


@knext.node(
    name="Point-Biserial Correlation", 
    node_type=knext.NodeType.MANIPULATOR, 
    icon_path=os.path.join(ICONS_PATH, "point_biserial.png"), 
    category=main_category
)
@knext.input_table(name="Input Data", description="Table containing the data to be analyzed.")
@knext.output_table(name="Result Data", description="Table containing the Point-Biserial Correlation.")
class PointBiserialNode:
    """
    Calculates Point-Biserial Correlation.
    
    Measures the relationship between one truly continuous variable and one naturally binary variable.
    """
    binary_col = knext.ColumnParameter("Binary Variable", "Select the binary column.", port_index=0)
    continuous_col = knext.ColumnParameter("Continuous Variable", "Select the continuous numeric column.", port_index=0)

    def configure(self, config_context, input_table_schema):
        return knext.Schema.from_columns([
            knext.Column(knext.double(), "Point-Biserial Correlation"),
            knext.Column(knext.double(), "P-Value"),
            knext.Column(knext.int64(), "Sample Size (N)"),
            knext.Column(knext.string(), "Binary Variable"),
            knext.Column(knext.string(), "Continuous Variable")
        ])

    def execute(self, exec_context, input_table):
        df = input_table.to_pandas()
        bin_name = self.binary_col
        cont_name = self.continuous_col
        
        temp = df[[bin_name, cont_name]].dropna()
        if len(temp) < 2: raise ValueError("Not enough data points after removing missing values.")
        
        if len(temp[bin_name].unique()) != 2:
             raise ValueError(f"Column '{bin_name}' must be binary (exactly 2 unique values).")
        
        uniques = temp[bin_name].unique()
        x_bin = np.where(temp[bin_name] == uniques[1], 1, 0)
        y_cont = pd.to_numeric(temp[cont_name], errors='coerce').to_numpy()
        
        mask = ~np.isnan(y_cont)
        x_bin = x_bin[mask]
        y_cont = y_cont[mask]
        
        if len(y_cont) < 2: raise ValueError("Not enough valid numeric data.")

        r_pb, p_val = pointbiserialr(x_bin, y_cont)

        out = pd.DataFrame({
            "Point-Biserial Correlation": [float(r_pb)],
            "P-Value": [float(p_val)],
            "Sample Size (N)": [int(len(x_bin))],
            "Binary Variable": [bin_name],
            "Continuous Variable": [cont_name]
        })
        return knext.Table.from_pandas(out)


@knext.node(
    name="Biserial Correlation", 
    node_type=knext.NodeType.MANIPULATOR, 
    icon_path=os.path.join(ICONS_PATH, "biserial.png"), 
    category=main_category
)
@knext.input_table(name="Input Data", description="Table containing the data to be analyzed.")
@knext.output_table(name="Result Data", description="Table containing the Biserial Correlation.")
class BiserialNode:
    """
    Calculates Biserial Correlation.
    
    Used when one variable is continuous and the other is an artificial dichotomy of an underlying normal distribution.
    """
    binary_col = knext.ColumnParameter("Binary Variable", "Select the binary column.", port_index=0)
    continuous_col = knext.ColumnParameter("Continuous Variable", "Select the continuous numeric column.", port_index=0)

    def configure(self, config_context, input_table_schema):
        return knext.Schema.from_columns([
            knext.Column(knext.double(), "Biserial Correlation"),
            knext.Column(knext.int64(), "Sample Size (N)"),
            knext.Column(knext.string(), "Binary Variable"),
            knext.Column(knext.string(), "Continuous Variable")
        ])

    def execute(self, exec_context, input_table):
        df = input_table.to_pandas()
        bin_name = self.binary_col
        cont_name = self.continuous_col
        _require_cols(df, bin_name, cont_name)
        
        temp = df[[bin_name, cont_name]].dropna()
        if len(temp) < 2: raise ValueError("Not enough data points after removing missing values.")
        
        if len(temp[bin_name].unique()) != 2:
             raise ValueError(f"Column '{bin_name}' must be binary (exactly 2 unique values).")
             
        uniques = temp[bin_name].unique()
        x_bin = np.where(temp[bin_name] == uniques[1], 1, 0)
        y_cont = pd.to_numeric(temp[cont_name], errors='coerce').to_numpy()
        
        mask = ~np.isnan(y_cont)
        x_bin = x_bin[mask]
        y_cont = y_cont[mask]
        
        val = _biserial_corr(x_bin, y_cont)
        
        out = pd.DataFrame({
            "Biserial Correlation": [val],
            "Sample Size (N)": [int(len(x_bin))],
            "Binary Variable": [bin_name],
            "Continuous Variable": [cont_name]
        })
        return knext.Table.from_pandas(out)
