import logging
from typing import Any

import numpy as np
import pandas as pd
import knime.extension as knext

# ------------------------------------------------------------------
# ????????? ??? Node Repository (???? ??? Community Nodes)
# ------------------------------------------------------------------
main_category = knext.category(
    path="/community",
    level_id="blomqvist_extension",
    name="Blomqvist Extension",
    description="Nodes for Blomqvist's Beta correlation.",
    icon="icon.png",  # ?? ????????? ??? ??? ?????? icons
)

# ------------------------------------------------------------------
# ????????? Blomqvist's Beta
# ------------------------------------------------------------------
def blomqvist_beta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Blomqvist's Beta between two numeric arrays.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    mx = np.median(x)
    my = np.median(y)

    concordant = np.sum((x - mx) * (y - my) > 0)
    beta = 4 * (concordant / len(x)) - 1
    return float(beta)


LOGGER = logging.getLogger(__name__)

# ------------------------------------------------------------------
# ? ??????
# ------------------------------------------------------------------
@knext.node(
    name="Blomqvist's Beta (demo)",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icon.png",
    category=main_category,
)
@knext.input_table(
    name="Input table",
    description="Input table with at least two numeric columns.",
)
@knext.output_table(
    name="Output table",
    description=(
        "Same as input table. The computed Blomqvist's Beta is "
        "written to the KNIME console."
    ),
)
class BlomqvistNode:
    """
    Demo version of a Blomqvist's Beta node.

    ???? ?? ????? ? ?????? ????? ??????? ??? ?????? ???? ?????
    ???? ?????????? ?? Blomqvist's Beta ??? ??? ??? ?????? ??????
    ??? ?? ?????? ??? KNIME Console.
    """

    def configure(
        self,
        config_context: knext.ConfigurationContext,
        input_table_schema: knext.Schema,
    ) -> knext.Schema:
        # ??? ????????? ?? schema – ???? ??????? ?? ???????? ???? ?????.
        return input_table_schema

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        input_table: knext.Table,
    ) -> knext.Table:
        # ????????? ??? KNIME table ?? pandas DataFrame
        df: pd.DataFrame = input_table.to_pandas()

        if df.shape[1] < 2:
            exec_context.set_warning(
                "Input table must have at least two columns. "
                "Node passes data unchanged."
            )
            return input_table

        col_x = df.columns[0]
        col_y = df.columns[1]

        x = df[col_x].to_numpy()
        y = df[col_y].to_numpy()

        beta = blomqvist_beta(x, y)

        msg = f"Blomqvist's Beta for '{col_x}' and '{col_y}' = {beta:.4f}"
        LOGGER.info(msg)
        exec_context.set_progress(1.0, msg)

        # ???? ?? ????? ???????????? ??? ?????? ??????.
        return input_table