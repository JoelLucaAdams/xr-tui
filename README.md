# xr-tui

xr-tui is an interactive terminal user interface (TUI) for exploring and visualizing xarray Datasets and DataArrays. It leverages the power of xarray for handling multi-dimensional arrays and provides a user-friendly interface for data exploration directly in the terminal.

## Installation
You can install xr-tui via pip:

```bash
pip install xr-tui
```

Or as a uv tool:

```bash
uv tool install xr-tui
```

## Features
- Interactive navigation through xarray Datasets and DataArrays.
- Visualization of 1D and 2D data using plotext for terminal-based plotting.
- Support for slicing multi-dimensional data.
- Easy-to-use command-line interface.

## Usage
To start xr-tui, simply run the following command in your terminal:

```bash
xr data.nc
```

This will launch the TUI, allowing you to explore the contents of `data.nc`.

## Key Commands

| Key | Action |
|-----|--------|
| `q` | Quit the application. |
| `h` | Show help menu. |
| `e` | Expand all nodes in the dataset tree. |
| `space` | Collapse all nodes in the dataset tree. |
| Arrow keys | Navigate through the dataset. |
| `Enter` | Select an item or open a variable |
| `s` | Show statistics of the selected variable. |
| `p` | Plot the selected variable. |
