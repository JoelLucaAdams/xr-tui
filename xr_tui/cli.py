import argparse
from functools import partial
from typing import Iterable
import numpy as np
import xarray as xr
from textual.app import App, ComposeResult, SystemCommand
from textual.widgets import Footer, Header, Tree, Static, DataTable
from textual.containers import Grid
from textual.screen import Screen
from textual.command import Hit, Hits, Provider
from textual_plotext import PlotextPlot


class StatisticsScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Pop screen")]

    def __init__(self, variable: xr.DataArray, **kwargs) -> None:
        super().__init__(**kwargs)
        self.variable = variable

    def compose(self) -> ComposeResult:
        """Create child widgets for the screen."""
        stats = self._compute_statistics(self.variable)
        data = self.variable.values.flatten()
        data = data[~np.isnan(data)]  # Remove NaN values

        plot_widget = PlotextPlot(id="hist-widget")
        plot_widget.plt.hist(data, bins=100)

        table = DataTable(id="stats-grid")
        table.add_column("Statistic")
        table.add_column("Value")

        for stat_name, stat_value in stats.items():
            table.add_row(stat_name, f"{stat_value:.4f}")

        yield Grid(plot_widget, table, id="stats-container")

    def _compute_statistics(self, variable: xr.DataArray) -> dict:
        """Compute basic statistics for the variable."""
        data = variable.values.flatten()
        data = data[~np.isnan(data)]  # Remove NaN values

        stats = {
            "Mean": data.mean(),
            "Median": np.median(data),
            "Standard Deviation": data.std(),
            "Minimum": data.min(),
            "Maximum": data.max(),
            "Count": len(data),
        }
        return stats


class PlotScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Pop screen")]

    def __init__(
        self, variable: xr.DataArray, plot_type: str = "line", **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.plot_type = plot_type
        self.variable = variable

    def compose(self) -> ComposeResult:
        """Create child widgets for the screen."""
        if self.plot_type == "line":
            plot_widget = self._plot_variable_1d()
        elif self.plot_type == "heatmap":
            plot_widget = self._plot_variable_2d()
        else:
            return

        yield plot_widget

    def _plot_variable_1d(self):
        variable = self.variable

        # Assume variable is 1D
        y_data = variable.values

        if len(variable.coords) == 0:
            x_data = range(len(y_data))
        else:
            x_data = variable[variable.dims[0]].values

        plot_widget = PlotextPlot(id="plot-widget")
        plot_widget.plt.plot(x_data, y_data, label="Array Data")
        plot_widget.plt.xlabel(variable.dims[0])
        plot_widget.plt.ylabel(f"{variable.name}")

        return plot_widget

    def _plot_variable_2d(self):
        variable = self.variable

        # Assume variable is 2D
        y_data = variable.values

        if len(variable.coords) < 2:
            x_data = range(y_data.shape[1])
            y_coords = range(y_data.shape[0])
        else:
            x_data = variable[variable.dims[1]].values
            y_coords = variable[variable.dims[0]].values

        plot_widget = PlotextPlot(id="plot-widget")
        plot_widget.plt.heatmap(x_data, y_coords, y_data)

        return plot_widget


class XarrayTUI(App):
    """A Textual app to view xarray Datasets."""

    CSS_PATH = "xr-tui.tcss"

    SCREENS = {"plot_screen": PlotScreen}

    BINDINGS = [
        ("q", "quit_app", "Quit"),
        ("escape", "quit_app", "Quit"),
        ("t", "toggle_expand", "Toggle expand/collapse of current node"),
        ("e", "expand_all", "Expand all nodes"),
        ("c", "collapse_all", "Collapse all nodes"),
        ("d", "toggle_dark", "Toggle dark mode"),
        ("p", "plot_variable", "Plot variable"),
        ("s", "show_statistics", "Show statistics"),
    ]

    def __init__(self, file: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.title = "xr-tui"
        self.theme = "monokai"
        self.file = file
        self.dataset = xr.open_datatree(file, chunks=None, create_default_indexes=False)

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        yield from super().get_system_commands(screen)
        yield SystemCommand("Bell", "Ring the bell", self.bell)

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield Footer()

        tree: Tree[str] = Tree(f"xarray Dataset: [bold]{self.file} [/bold]")
        tree.root.expand()

        def add_group_node(parent_node: Tree, group, group_name: str = "") -> None:
            """Recursively add group nodes to the tree."""
            num_vars = len(group.data_vars)
            num_coords = len(group.coords)
            label = f"Group: {group_name}" if group_name else "Root"
            group_node = parent_node.add(
                f"{label} (Data Variables: [blue]{num_vars}[/blue], Coordinates: [blue]{num_coords}[/blue])"
            )
            group_node.expand()

            self._add_dims_node(group_node, group)
            self._add_coords_node(group_node, group)
            self._add_data_vars_node(group_node, group)

            # Recursively add child groups
            for child_name in group.children:
                child_group = group[child_name]
                add_group_node(group_node, child_group, child_name)

        add_group_node(tree.root, self.dataset)

        yield tree

    def _add_dims_node(self, parent_node: Tree, group) -> None:
        """Helper method to add dimension nodes to the tree."""
        dims_node = parent_node.add("Dimensions")
        dims_node.expand()
        for dim_name, dim_size in group.dims.items():
            dims_node.add_leaf(f"{dim_name}: [blue]{dim_size}[/blue]")

    def _add_data_vars_node(self, parent_node: Tree, group) -> None:
        """Helper method to add data variable nodes to the tree."""
        data_vars_node = parent_node.add("Data Variables")
        data_vars_node.expand()
        for var_name in group.data_vars.keys():
            self._add_var_node(data_vars_node, group.data_vars[var_name])

    def _add_coords_node(self, parent_node: Tree, group) -> None:
        """Helper method to add coordinate nodes to the tree."""
        coords_node = parent_node.add("Coordinates")
        coords_node.expand()
        for coord_name in group.coords.keys():
            self._add_var_node(coords_node, group.coords[coord_name])

    def _add_var_node(self, parent_node: Tree, var: xr.DataArray) -> None:
        """Helper method to add a variable node to the tree."""
        nbytes = self._convert_nbytes_to_readable(var.nbytes)
        var_node = parent_node.add(
            f"{var.name}: [red]{var.dims}[/] [green]{var.dtype}[/] [blue]{nbytes}[/]",
        )
        var_node.data = {"name": var.name, "type": "variable_node", "item": var}

        num_attributes = len(var.attrs)
        attr_node = var_node.add(f"Attributes ([blue]{num_attributes}[/blue])")
        for attr, value in var.attrs.items():
            attr_node.add_leaf(f"[yellow]{attr}[/]: {value}")

    def _convert_nbytes_to_readable(self, nbytes: int) -> str:
        """Convert bytes to a human-readable format."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if nbytes < 1024:
                return f"{nbytes:.2f} {unit}"
            nbytes /= 1024
        return f"{nbytes:.2f} PB"

    def action_plot_variable(self) -> None:
        """An action to plot the currently selected variable."""
        tree = self.query_one(Tree)
        current_node = tree.cursor_node
        if current_node is None:
            return

        if (
            current_node.data is None
            or current_node.data.get("type") != "variable_node"
        ):
            return

        self.push_screen(PlotScreen(current_node.data["item"]))

    def action_show_statistics(self) -> None:
        """An action to show statistics of the currently selected variable."""
        tree = self.query_one(Tree)
        current_node = tree.cursor_node
        if current_node is None:
            return

        if (
            current_node.data is None
            or current_node.data.get("type") != "variable_node"
        ):
            return

        self.push_screen(StatisticsScreen(current_node.data["item"]))

    def action_expand_all(self) -> None:
        """An action to expand all tree nodes."""
        self.query_one(Tree).root.expand_all()

    def action_collapse_all(self) -> None:
        """An action to collapse all tree nodes."""
        self.query_one(Tree).root.collapse_all()

    def action_toggle_expand(self) -> None:
        """An action to collapse all tree nodes."""
        current_node = self.query_one(Tree).cursor_node
        if current_node.is_collapsed:
            current_node.expand()
        else:
            current_node.collapse()

    def action_quit_app(self) -> None:
        self.exit()

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light"
        )


def main():
    parser = argparse.ArgumentParser(
        description="A Textual TUI for managing xarray Datasets."
    )
    parser.add_argument("file", type=str, help="Path to the xarray Dataset file.")
    args = parser.parse_args()

    app = XarrayTUI(args.file)
    app.run()


if __name__ == "__main__":
    main()
