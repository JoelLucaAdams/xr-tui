import argparse
import xarray as xr
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Tree


class XarrayTUIApp(App):
    """A Textual app to manage stopwatches."""

    BINDINGS = [("d", "toggle_dark", "Toggle dark mode"), ("q", "quit_app", "Quit")]

    def __init__(self, file: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.file = file
        self.dataset = xr.open_datatree(file)

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield Footer()

        tree: Tree[str] = Tree(f"xarray Dataset: {self.file}")
        tree.root.expand()

        name = "summary"
        group = self.dataset[name]

        num_vars = len(group.data_vars)
        group_node = tree.root.add(f"Group: {name} (Data Variables: {num_vars})")
        group_node.expand()

        coords_node = group_node.add("Coordinates")
        coords_node.expand()

        for coord in group.coords.keys():
            coord_node = coords_node.add(coord)
            coord_node.add_leaf(f"Dimensions: {group[coord].dims}")
            coord_node.add_leaf(f"Shape: {group[coord].shape}")
            coord_node.add_leaf(f"Data Type: {group[coord].dtype}")

            num_attributes = len(group[coord].attrs)
            attr_node = coord_node.add(f"Attributes ({num_attributes})")
            for attr, value in group[coord].attrs.items():
                attr_node.add_leaf(f"{attr}: {value}")

            coord_node.expand()

        data_vars_node = group_node.add("Data Variables")
        data_vars_node.expand()

        for var in group.data_vars.keys():
            var_node = data_vars_node.add(var)
            var_node.add_leaf(f"Dimensions: {group[var].dims}")
            var_node.add_leaf(f"Shape: {group[var].shape}")
            var_node.add_leaf(f"Data Type: {group[var].dtype}")

            num_attributes = len(group[var].attrs)
            attr_node = var_node.add(f"Attributes ({num_attributes})")
            for attr, value in group[var].attrs.items():
                attr_node.add_leaf(f"{attr}: {value}")

            var_node.expand()

        yield tree

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

    app = XarrayTUIApp(args.file)
    app.run()


if __name__ == "__main__":
    main()
