
import traceback
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import json
import random
import colorsys
import geopandas as gpd
import fiona
import numpy as np


def generate_distinct_colors(n, min_lightness=0.25, max_lightness=0.75):
    """Generate n visually distinct colors using HSL spacing."""
    colors = []
    step = 360 / n
    for i in range(n):
        h = (i * step + random.uniform(-10, 10)) % 360
        s = random.uniform(0.5, 0.9)
        l = random.uniform(min_lightness, max_lightness)
        r, g, b = colorsys.hls_to_rgb(h / 360, l, s)
        colors.append('#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255)))
    return colors


def generate_random_color_dict(gpkg_path, layer_name=None, column_name=None, output_txt="color_dict.json"):
    """
    Generate a dictionary assigning random, visually distinct colors to unique values in a column
    of a GeoPackage layer. Automatically detects layer if only one exists.

    :param gpkg_path: Path to GeoPackage
    :param layer_name: Optional layer name (auto-detects if only one layer exists)
    :param column_name: Column with categorical values (e.g. 'straatnaam')
    :param output_txt: Path to save JSON file
    :return: dict {value: color_hex}
    """
    if not os.path.exists(gpkg_path):
        raise FileNotFoundError(f"GeoPackage not found: {gpkg_path}")

    # Detect layer name if not provided
    if layer_name is None:
        layers = fiona.listlayers(gpkg_path)
        if len(layers) == 1:
            layer_name = layers[0]
            print(f"Auto-detected layer: {layer_name}")
        else:
            raise ValueError(f"Multiple layers found: {layers}. Please specify one.")

    gdf = gpd.read_file(gpkg_path, layer=layer_name)
    if column_name not in gdf.columns:
        raise ValueError(f"Column '{column_name}' not found in layer '{layer_name}'.")

    unique_vals = sorted(gdf[column_name].dropna().unique())
    print(f"Found {len(unique_vals)} unique values in '{column_name}'.")

    colors = generate_distinct_colors(len(unique_vals))
    color_dict = dict(zip(unique_vals, colors))

    with open(output_txt, "w", encoding="utf-8") as f:
        json.dump(color_dict, f, indent=2, ensure_ascii=False)

    print(f"✓ Color dictionary saved to {output_txt}")
    return color_dict

def create_maps_from_gpkg(
    gpkg_path,
    area_layer,
    output_dir,
    name_field,
    color_dict_file,
    line_gpkg_path,
    line_layer,
    line_street_col,
    dpi=700
):
    """
    Create maps per area using:
    - Fixed-color polygon layers
    - A random-colored polygon layer (based on pre-generated color dictionary)
    - Optional line layer colored using the same dictionary

    :param gpkg_path: GeoPackage with polygon layers (including 'wijken').
    :param area_layer: Name of the polygon layer defining areas.
    :param output_dir: Output folder for maps.
    :param name_field: Field in the area layer used for naming maps.
    :param color_dict_file: JSON file containing streetname-color mapping.
    :param line_gpkg_path: Optional path to second GeoPackage (with line features).
    :param line_layer: Optional line layer name.
    :param line_street_col: Column in line layer for streetnames.
    :param dpi: Output DPI.
    """

    fixed_colors = {
        "zonder_straatnaam": "#cccccc",
        "met_meerdere_straatnamen": "#ff9999",
    }
    random_layer = "met_een_straatnaam"

    os.makedirs(output_dir, exist_ok=True)

    # Load area polygons
    areas = gpd.read_file(gpkg_path, layer=area_layer)
    crs = areas.crs

    # Load all polygon layers except 'wijken'
    layers = fiona.listlayers(gpkg_path)
    # layers = gpd.io.file.fiona.listlayers(gpkg_path)
    polygon_layers = {
        lyr: gpd.read_file(gpkg_path, layer=lyr).to_crs(crs)
        for lyr in layers if lyr != area_layer
    }

    # Load color dictionary
    if not os.path.exists(color_dict_file):
        raise FileNotFoundError(f"Color dictionary not found: {color_dict_file}")
    with open(color_dict_file, "r", encoding="utf-8") as f:
        color_dict = json.load(f)

    # Optional line layer
    line_layer_data = None
    if line_gpkg_path and line_layer:
        line_layer_data = gpd.read_file(line_gpkg_path, layer=line_layer).to_crs(crs)
        print("line layer data is not none")

    print(f"Loaded layers: {list(polygon_layers.keys())}")

    # Assign polygons to areas
    for name, gdf in polygon_layers.items():
        polygon_layers[name] = gpd.sjoin(gdf, areas, how="left", predicate="intersects")

    area_indices = pd.concat(
        [ov["index_right"].dropna() for ov in polygon_layers.values()]
    ).unique()

    for idx in area_indices:
        try:
            idx = int(idx)
            area = areas.iloc[[idx]]
            area_name = str(area.iloc[0].get(name_field, f"area_{idx}"))
            bounds = area.total_bounds
            buffer = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.05

            fig, ax = plt.subplots(1, 1, figsize=(20, 12))
            legend_elements = []

            for layer_name, gdf in polygon_layers.items():
                ov_area = gdf[gdf["index_right"] == idx]
                if ov_area.empty:
                    continue

                # Fixed color layers
                if layer_name in fixed_colors:
                    color = fixed_colors[layer_name]
                    ov_area.plot(ax=ax, color=color, edgecolor="black", linewidth = 0.1, alpha=0.8)
                    legend_elements.append(Patch(facecolor=color, label=layer_name))

                # Random color layer based on dictionary
                elif layer_name == random_layer:
                    col_name = "naam"  # adjust if needed
                    for val, subset in ov_area.groupby(col_name):
                        color = color_dict.get(val, "#999999")
                        subset.plot(ax=ax, color=color, edgecolor="black", linewidth = 0.1, alpha=0.8)
                    legend_elements.append(Patch(facecolor="#999999", label=layer_name))

            # --- Plot line layer (AFTER all polygons) ---
            if line_layer_data is not None:
                print("Plotting line layer...")
                lines_clip = gpd.clip(line_layer_data, area)
                for _, row in lines_clip.iterrows():
                    val = row.get(line_street_col)
                    color = color_dict.get(val, "#999999")
                    geom = row.geometry

                    # Plot the line
                    if geom is not None:
                        gpd.GeoSeries([geom]).plot(ax=ax, color=color, linewidth=0.8, zorder=5)

                        # Add label
                        if val and geom.geom_type == "LineString":
                            # get line midpoint
                            midpoint = geom.interpolate(0.5, normalized=True)
                            x, y = midpoint.x, midpoint.y

                            # approximate angle for label rotation
                            x1, y1, x2, y2 = geom.coords[0][0], geom.coords[0][1], geom.coords[-1][0], geom.coords[-1][
                                1]
                            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

                            ax.text(
                                x, y, val,
                                fontsize=2,
                                rotation=angle,
                                rotation_mode='anchor',
                                ha='center',
                                va='center',
                                color='black',
                                alpha=0.6,
                                zorder=6
                            )

                legend_elements.append(Patch(facecolor="none", edgecolor="black", label="Street lines"))

            # Draw area boundary
            area.plot(ax=ax, color="none", edgecolor="black", linewidth=2)

            ax.set_xlim(bounds[0] - buffer, bounds[2] + buffer)
            ax.set_ylim(bounds[1] - buffer, bounds[3] + buffer)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)

            ax.set_title(area_name, fontsize=18, fontweight="bold", pad=25)
            if legend_elements:
                ax.legend(
                    handles=legend_elements,
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    frameon=True,
                    fontsize=10,
                )

            output_path = os.path.join(output_dir, f"{area_name}.png")
            plt.tight_layout()
            plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            plt.close()
            print(f"✓ Map created for {area_name}")

        except Exception as e:
            print(f"✗ Error creating map for area index {idx}: {e}")
            traceback.print_exc()

    print(f"Maps created in {output_dir}")


# Step 1: Generate color dictionary once
# color_dict = generate_random_color_dict(
#     gpkg_path="input/wegvakken_nwb.gpkg",
#     layer_name=None,           # auto-detects the only layer
#     column_name="sttnaam",
#     output_txt="street_colors.txt"
# )

# Step 2: Create maps using that dictionary
create_maps_from_gpkg(
    gpkg_path="input/percelen_adres.gpkg",
    area_layer="wijken",
    output_dir="output/maps",
    name_field="wijknaam",
    color_dict_file="street_colors.txt",
    line_gpkg_path="input/wegvakken_nwb.gpkg",
    line_layer= "uitgenomen_locatie",
    line_street_col="sttnaam"
)
