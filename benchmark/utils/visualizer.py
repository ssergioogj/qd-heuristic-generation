import matplotlib.pyplot as plt
import itertools
import pandas as pd
import numpy as np

from ..online_bin_packing_1d.simulator import BPPSimulator

class Visualizer:
    MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">"]
    LINE_STYLES = ["-", "--", "-.", ":"]
    COLORS = ["b", "g", "r", "c", "m", "y", "k"]

    # Generate style combinations
    STYLE_COMBINATIONS = [
        f"{c}{line}{m}"
        for line, m, c in itertools.product(LINE_STYLES, MARKERS, COLORS)
    ]

    @classmethod
    def plot_metric_comparison(cls, results_by_group: dict, series_styles: dict, metric: str, group_keys: list = None, sharey: bool = True, figsize: tuple = None):
        """
        Generate comparative subplots
        """
        if group_keys is None:
            group_keys = list(results_by_group.keys())

        n_groups = len(group_keys)
        if n_groups == 0:
            return

        if figsize is None:
            figsize = (8 * n_groups, 5)

        fig, axes = plt.subplots(1, n_groups, figsize=figsize, sharey=sharey, squeeze=False)
        axes = axes.flatten()

        for i, (ax, grp_name) in enumerate(zip(axes, group_keys)):
            grp_data = results_by_group[grp_name]
            x_axis = grp_data["x_values"]

            for name, style in series_styles.items():
                y_values = grp_data["results"][name][metric]
                ax.plot(x_axis, y_values, style, label=name, markersize=5)

            ax.set_title(f"Distribution: {str(grp_name).capitalize()}")
            ax.set_xlabel("Instance")
            ax.set_ylabel(metric.upper() if i == 0 else "")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.tick_params(axis="y", labelleft=True)
            ax.tick_params(axis="x", rotation=45)

        axes[-1].legend(loc="upper right")
        fig.suptitle(f"Comparative - {metric.upper()}", fontsize=14)
        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_table(cls, rows: list|np.array, columns:list|np.array, cellText: list|np.array, rowColor: str = "#5894f5", colColor: str = "#5894f5", cellColor: str = "#ffffff"):
        fig, ax = plt.subplots()
        ax.axis("off")

        table = plt.table(rowLabels=rows, colLabels=columns, cellText = cellText, loc="center", cellLoc="center", cellColor=cellColor, colColours=colColor, rowColours=rowColor)

        table.auto_set_font_size(False)

        table.set_fontsize(14)

        table.scale(1.2, 2.0)

        plt.show()

    @staticmethod
    def plot_academic_table(
        df_results: pd.DataFrame, 
        heuristic_cols: list[str], 
        title: str = "Comparison of Online Bin Packing Heuristics",
        output_filename: str = "table_results.png"
    ):
        """
        Renderiza un DataFrame como una tabla estilo paper científico usando Matplotlib.
        """
        # 1. Calcular fila 'Average'
        avg_row = {"Capacity": "Average", "Size": ""}
        for h in heuristic_cols:
            avg_row[h] = df_results[h].mean()

        display_df = pd.concat([df_results, pd.DataFrame([avg_row])], ignore_index=True)

        # 2. Configurar la figura limpia sin ejes
        n_rows, n_cols = display_df.shape
        fig, ax = plt.subplots(figsize=(1.8 * n_cols, 0.45 * n_rows + 1.2), dpi=300)
        ax.axis("off")

        # 3. Preparar los datos textuales
        table_data = []
        cell_bold_flags = []  # Para marcar en negrita los mínimos de cada fila

        prev_cap = None
        for row_idx, row in display_df.iterrows():
            row_vals = []
            is_avg = (row_idx == len(display_df) - 1)

            # Control de visualización de capacidad (solo en la primera fila del bloque)
            cap = row["Capacity"]
            if not is_avg:
                cap_str = str(cap) if cap != prev_cap else ""
                prev_cap = cap
            else:
                cap_str = "Average"

            row_vals.append(cap_str)
            row_vals.append(str(row["Size"]))

            # Extraer scores y encontrar el mejor (mínimo AEB)
            scores = {h: row[h] for h in heuristic_cols}
            min_score = min(scores.values())

            row_bolds = [False, False]
            for h in heuristic_cols:
                val = scores[h]
                row_vals.append(f"{val:.2f}%")
                # Marcar en negrita si empata con el mejor
                row_bolds.append(np.isclose(val, min_score, atol=1e-3))

            table_data.append(row_vals)
            cell_bold_flags.append(row_bolds)

        headers = ["Capacity", "Size"] + heuristic_cols

        # 4. Crear la tabla de Matplotlib
        mpl_table = ax.table(
            cellText=table_data,
            colLabels=headers,
            loc="center",
            cellLoc="center"
        )
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(10)
        mpl_table.scale(1, 1.4)

        # 5. Aplicar formato visual tipo 'booktabs'
        for (r, c), cell in mpl_table.get_celld().items():
            cell.set_edgecolor("none")  # Eliminar bordes por defecto
            cell.set_facecolor("white")

            # Encabezado (r == 0)
            if r == 0:
                cell.set_text_props(weight="bold", color="#111111")
                cell.visible_edges = "TB"
                cell.set_edgecolor("black")
                cell.set_linewidth(1.5)
            else:
                data_row_idx = r - 1
                is_bold = cell_bold_flags[data_row_idx][c]
                if is_bold:
                    cell.set_text_props(weight="bold")

                # Línea intermedia si cambia la capacidad
                cap_val = display_df.iloc[data_row_idx]["Capacity"]
                if data_row_idx > 0 and cap_val != display_df.iloc[data_row_idx - 1]["Capacity"] and cap_val != "Average":
                    cell.visible_edges = "T"
                    cell.set_edgecolor("#cccccc")
                    cell.set_linewidth(0.8)

                # Línea superior e inferior de la fila Average
                if data_row_idx == len(display_df) - 1:
                    cell.visible_edges = "TB"
                    cell.set_edgecolor("black")
                    cell.set_linewidth(1.5 if r == len(display_df) else 1.0)

        plt.title(title, fontsize=12, pad=15, weight="bold")
        plt.tight_layout()
        plt.savefig(output_filename, bbox_inches="tight", dpi=300)
        plt.close()