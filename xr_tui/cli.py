import argparse
from typing import Iterable
import numpy as np
import xarray as xr
from textual.app import App, ComposeResult, SystemCommand
from textual.widgets import (
    Footer,
    Header,
    Tree,
    DataTable,
    RadioButton,
    RadioSet,
)
from textual import on
from textual.containers import Grid, Horizontal, Vertical
from textual.screen import Screen
from textual_slider import Slider
from textual_plotext import PlotextPlot


class StatisticsScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Pop screen")]

    def __init__(self, variable: xr.DataArray, **kwargs) -> None:
        super().__init__(**kwargs)
        self.variable = variable
        self.n_bins = 100

    def compose(self) -> ComposeResult:
        """Create child widgets for the screen."""
        stats = self._compute_statistics(self.variable)
        data = self.variable.values.flatten()
        data = data[~np.isnan(data)]  # Remove NaN values

        plot_widget = PlotextPlot(id="hist-widget")
        plot_widget.plt.hist(data, bins=self.n_bins)
        plot_widget.plt.title(f"Histogram of {self.variable.name}")
        plot_widget.plt.xlabel("Value")
        plot_widget.plt.ylabel("Frequency")

        table = DataTable(id="stats-table")
        table.add_column("Statistic")
        table.add_column("Value")

        for stat_name, stat_value in stats.items():
            table.add_row(stat_name, f"{stat_value:.4f}")

        yield Grid(plot_widget, table, id="stats-container")

    def _compute_statistics(self, variable: xr.DataArray) -> dict:
        """Compute basic statistics for the variable."""
        data = variable.values.flatten()

        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        pct_nan = (nan_count / data.size) * 100
        pct_inf = (inf_count / data.size) * 100

        data = data[~np.isnan(data)]  # Remove NaN values

        pct_25 = np.percentile(data, 25)
        pct_50 = np.percentile(data, 50)
        pct_75 = np.percentile(data, 75)

        stats = {
            "Mean": data.mean(),
            "Median": np.median(data),
            "Standard Deviation": data.std(),
            "Range": data.max() - data.min(),
            "Minimum": data.min(),
            "25%": pct_25,
            "50%": pct_50,
            "75%": pct_75,
            "Maximum": data.max(),
            "Count": len(data),
            "NaN Count": nan_count,
            "NaN %": pct_nan,
            "Inf Count": inf_count,
            "Inf %": pct_inf,
        }
        return stats


class PlotScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Pop screen")]

    def __init__(self, variable: xr.DataArray, **kwargs) -> None:
        super().__init__(**kwargs)
        self.variable = variable

    def compose(self) -> ComposeResult:
        """Create child widgets for the screen."""

        if len(self.variable.dims) == 1:
            plot_widget = self._plot_variable_1d()
        elif len(self.variable.dims) == 2:
            plot_widget = self._plot_variable_2d()
        else:
            dims = list(self.variable.dims)

            dim1 = 0
            dim2 = 1

            r1_buttons = []
            for i, dim in enumerate(dims):
                disabled = i == dim2
                button = RadioButton(dim, value=i == dim1, disabled=disabled)
                r1_buttons.append(button)

            r2_buttons = []
            for i, dim in enumerate(dims):
                disabled = i == dim1
                button = RadioButton(dim, value=i == dim2, disabled=disabled)
                r2_buttons.append(button)

            r1 = RadioSet(*r1_buttons, id="x-dim-select-1")
            r1.border_title = "Y Dimension"
            r2 = RadioSet(*r2_buttons, id="y-dim-select-2")
            r2.border_title = "x Dimension"

            slice_inputs = self.create_slice_sliders(dim1, dim2)

            plot_widget = self._plot_variable_nd()
            plot_widget = Vertical(
                Horizontal(r1, r2, id="dim-select-container"),
                slice_inputs,
                plot_widget,
                id="plot-container",
            )
            plot_widget.border_title = f"[bold]Slice Plot of {self.variable.name}[/]"

        yield plot_widget

    def create_slice_sliders(self, dim1: int = 0, dim2: int = 1) -> None:
        slice_inputs = []
        dims = list(self.variable.dims)
        for dim in dims:
            if dim not in [dims[dim1], dims[dim2]]:
                dim_size = self.variable.sizes[dim]
                slider = Slider(
                    0,
                    dim_size - 1,
                    step=1,
                    id=f"slice-{dim}",
                    name=dim,
                    value=dim_size // 2,
                )
                slider.border_title = f"Slice Position for {dim}"
                slice_inputs.append(slider)

        slice_inputs = Horizontal(*slice_inputs, id="slice-inputs-container")
        return slice_inputs

    @on(Slider.Changed)
    async def on_slider_changed_normal(self, event: Slider.Changed) -> None:
        slicers = self.query_one("#slice-inputs-container")
        slicers = slicers.children
        slice_positions = {
            slicer.name: slicer.value
            for slicer in slicers
            if isinstance(slicer, Slider)
        }

        dim1_group = self.query_one("#x-dim-select-1")
        dim2_group = self.query_one("#y-dim-select-2")

        dim1 = self._get_selected_dim(dim1_group)
        dim2 = self._get_selected_dim(dim2_group)

        new_plot = self._plot_variable_nd(dim1, dim2, slice_positions)
        plot_container = self.query_one("#plot-container")

        # Swap out the old plot with the new one
        await plot_container.children[-1].remove()
        await plot_container.mount(new_plot)

    def _get_selected_dim(self, radio_set: RadioSet) -> int:
        for i, radio in enumerate(radio_set.children):
            if isinstance(radio, RadioButton) and radio.value:
                return i
        return 0

    async def on_radio_set_changed(self, message: RadioSet.Changed):
        dim1_group = self.query_one("#x-dim-select-1")
        dim2_group = self.query_one("#y-dim-select-2")

        dim1 = self._get_selected_dim(dim1_group)
        dim2 = self._get_selected_dim(dim2_group)

        for i, radio in enumerate(dim1_group.children):
            if isinstance(radio, RadioButton):
                radio.disabled = i == dim2

        for i, radio in enumerate(dim2_group.children):
            if isinstance(radio, RadioButton):
                radio.disabled = i == dim1

        dim1_group.refresh()
        dim2_group.refresh()

        slice_inputs = self.create_slice_sliders(dim1, dim2)

        # Re-plot with new dimensions
        new_plot = self._plot_variable_nd(dim1, dim2)
        plot_container = self.query_one("#plot-container")
        await plot_container.children[-1].remove()
        await plot_container.children[-1].remove()
        await plot_container.mount(slice_inputs)
        await plot_container.mount(new_plot)

    def _plot_variable_1d(self):
        variable = self.variable
        x_dim_name = variable.dims[0]

        if x_dim_name in variable.coords:
            x_coords = variable.coords[x_dim_name].values
        else:
            x_coords = np.arange(variable.shape[0])

        y_values = variable.values
        y_values = np.nan_to_num(y_values, nan=0.0)

        plot_widget = PlotextPlot(id="plot-container")
        plot_widget.plt.plot(x_coords.tolist(), y_values.tolist())
        xunit = variable.coords[x_dim_name].attrs.get("units", "")
        xlabel = f"{x_dim_name} ({xunit})" if xunit else x_dim_name
        plot_widget.plt.xlabel(xlabel)
        plot_widget.plt.ylabel(variable.name)
        plot_widget.plt.title(f"1D Plot of {variable.name}")
        return plot_widget

    def _plot_variable_2d(self):
        variable = self.variable

        x_dim_name = variable.dims[1]
        y_dim_name = variable.dims[0]

        z = variable.values
        z = np.nan_to_num(z, nan=0.0)

        # Get coordinate values
        if x_dim_name in variable.coords:
            x_coords = variable.coords[x_dim_name].values
        else:
            x_coords = np.arange(z.shape[1])

        if y_dim_name in variable.coords:
            y_coords = variable.coords[y_dim_name].values
        else:
            y_coords = np.arange(z.shape[0])

        plot_widget = PlotextPlot(id="plot-container")
        plot_widget.plt.matrix_plot(z.tolist())
        plot_widget.plt.xticks(
            np.arange(len(x_coords)), labels=[f"{val:.4f}" for val in x_coords]
        )
        plot_widget.plt.yticks(
            np.arange(len(y_coords)), labels=[f"{val:.4f}" for val in y_coords]
        )

        xunit = variable.coords[x_dim_name].attrs.get("units", "")
        yunit = variable.coords[y_dim_name].attrs.get("units", "")

        xlabel = f"{x_dim_name} ({xunit})" if xunit else x_dim_name
        ylabel = f"{y_dim_name} ({yunit})" if yunit else y_dim_name
        plot_widget.plt.xlabel(xlabel)
        plot_widget.plt.ylabel(ylabel)

        plot_widget.plt.title(f"2D Plot of {variable.name}")

        return plot_widget

    def _plot_variable_nd(
        self, dim1: int = 0, dim2: int = 1, slice_positions: dict = {}
    ) -> PlotextPlot:
        variable = self.variable
        variable = variable.dropna(
            dim="major_radius", how="all"
        )  # Example of dropping NaNs along a dimension

        # Get all dimension names
        dims = list(variable.dims)

        # Default to last two dimensions for x and y
        y_dim_name = dims[dim1] if len(dims) >= 2 else dims[0]
        x_dim_name = dims[dim2]

        # Get the indices for other dimensions (set to middle slice)
        slice_dict = {}
        for dim in dims:
            if dim not in [x_dim_name, y_dim_name]:
                if dim in slice_positions:
                    slice_dict[dim] = slice_positions[dim]
                else:
                    dim_size = variable.sizes[dim]
                    slice_dict[dim] = dim_size // 2

        # Slice the variable to get 2D data
        sliced_var = variable.isel(slice_dict)

        z = sliced_var.values
        z = np.nan_to_num(z, nan=0.0)

        # Get coordinate values
        if x_dim_name in sliced_var.coords:
            x_coords = sliced_var.coords[x_dim_name].values
        else:
            x_coords = np.arange(z.shape[1])

        if y_dim_name in sliced_var.coords:
            y_coords = sliced_var.coords[y_dim_name].values
        else:
            y_coords = np.arange(z.shape[0])

        plot_widget = PlotextPlot(id="plot-widget")
        plot_widget.plt.matrix_plot(z.tolist())
        plot_widget.plt.xticks(
            np.arange(len(x_coords)), labels=[f"{val:.4f}" for val in x_coords]
        )
        plot_widget.plt.yticks(
            np.arange(len(y_coords)), labels=[f"{val:.4f}" for val in y_coords]
        )

        xunit = sliced_var.coords[x_dim_name].attrs.get("units", "")
        yunit = sliced_var.coords[y_dim_name].attrs.get("units", "")

        xlabel = f"{x_dim_name} ({xunit})" if xunit else x_dim_name
        ylabel = f"{y_dim_name} ({yunit})" if yunit else y_dim_name
        plot_widget.plt.xlabel(xlabel)
        plot_widget.plt.ylabel(ylabel)

        # Add info about sliced dimensions to title
        slice_info = ", ".join([f"{dim}={idx}" for dim, idx in slice_dict.items()])
        title = f"{variable.name} ({slice_info})" if slice_info else f"{variable.name}"
        plot_widget.plt.title(title)
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
        self.dataset = xr.open_datatree(
            file, chunks=None, create_default_indexes=False, engine="zarr"
        )

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
