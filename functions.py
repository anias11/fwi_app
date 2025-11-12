import pandas as pd
import numpy as np
import streamlit as st
import base64
from cartopy.io import shapereader
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
# ==========================


def img_to_data_uri(path: str) -> str:
    mime = "image/svg+xml" if path.lower().endswith(".svg") else "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode("utf-8")


# ==========================
# Función de plotting
# ==========================
# Máscara para ocultar datos sobre el mar
def mask_ocean(var_data, ax):
    """
    Aplica una máscara para ocultar datos sobre el océano.
    """
    import shapely.geometry as sgeom
    
    # Obtener geometrías de tierra
    land_geom = list(shapereader.natural_earth(resolution='10m', 
                                                category='physical', 
                                                name='land').geometries())
    
    # Crear máscara de tierra
    land_mask = None
    for geom in land_geom:
        if land_mask is None:
            land_mask = geom
        else:
            land_mask = land_mask.union(geom)
    
    return land_mask

def plot_variable_cartopy(ds, var_name, variable_cmaps, variable_display_names, 
                          title=None, time_index=0):
    """
    Plot a variable from the dataset using Cartopy projection.
    If multiple times exist, select one via `time_index`.
    """
    display_name = variable_display_names.get(var_name, var_name)
    cmap_local = variable_cmaps.get(var_name, 'viridis')

    var_data = ds[var_name]

    # --- Seleccionar una data si hi ha múltiples temps ---
    if 'time' in var_data.dims:
        n_times = var_data.sizes['time']
        if time_index >= n_times:
            raise IndexError(f"time_index {time_index} is out of range (dataset has {n_times} times)")
        var_data = var_data.isel(time=time_index)
        date_str = str(np.datetime_as_string(ds['time'].isel(time=time_index).values, unit='D'))
    elif 'date' in ds.coords:
        date_str = str(np.datetime_as_string(ds['date'].values, unit='D'))
    else:
        date_str = None

    # --- Crear figura i eixos Cartopy ---
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Afegir elements del mapa
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.OCEAN, color='lightblue')
    ax.add_feature(cfeature.LAND, color='lightgray')

    # --- Plot especial per FWI_risk ---
    if var_name == 'FWI_risk':
        # risk: 0 = nodata, 1..5 = categories
        risk = var_data.astype(float)

        # Mascarem zeros perquè no es pintin
        risk = risk.where(risk != 0)

        colors = ['#a6d96a', '#ffffbf', '#fdae61', '#f46d43', '#d73027']
        labels = ['Bajo', 'Moderado', 'Alto', 'Muy Alto', 'Extremo']

        cmap = ListedColormap(colors)
        # Franges: (0.5-1.5)->1, (1.5-2.5)->2, ..., (4.5-5.5)->5
        bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        norm = BoundaryNorm(bounds, cmap.N)

        im = ax.pcolormesh(
            risk.longitude, risk.latitude, risk,
            cmap=cmap, norm=norm,
            transform=ccrs.PlateCarree(),
            shading='auto'
        )

        cbar = plt.colorbar(
            im, ax=ax, shrink=0.6,
            ticks=[1, 2, 3, 4, 5]
        )
        cbar.set_ticklabels(labels, fontfamily='Poppins')
        cbar.set_label(display_name, rotation=270, labelpad=20, fontfamily='Poppins')

    elif var_name == 'FWI_anomalies':
        # risk: 0 = nodata, 1..5 = categories
        anomaly = var_data.astype(float)

        # Mascarem zeros perquè no es pintin
        anomaly = anomaly.where(anomaly != 0)

        colors = [ '#67a9cf', '#ffffbf', '#fdae61', '#d73027',  "#701d19",]
        labels = ['Bajo', 'Moderado', 'Alto', 'Muy Alto', 'Extremo']

        cmap = ListedColormap(colors)
        # Franges: (0.5-1.5)->1, (1.5-2.5)->2, ..., (4.5-5.5)->5
        bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        norm = BoundaryNorm(bounds, cmap.N)

        im = ax.pcolormesh(
            anomaly.longitude, anomaly.latitude, anomaly,
            cmap=cmap, norm=norm,
            transform=ccrs.PlateCarree(),
            shading='auto'
        )

        cbar = plt.colorbar(
            im, ax=ax, shrink=0.6,
            ticks=[1, 2, 3, 4, 5]
        )
        cbar.set_ticklabels(labels, fontfamily='Poppins')
        cbar.set_label(display_name, rotation=270, labelpad=20, fontfamily='Poppins')

    else:
        # --- Unitats segons variable ---
        units_dict = {
            't2m': '°C',
            'rh': '%',
            'wind10m': 'km/h',
            'rain_24h': 'mm'
        }
        units = units_dict.get(var_name, None)

        if var_name == 'rain_24h':
            data_vals = var_data.values
            if np.allclose(data_vals, 0):
                im = ax.imshow(
                    np.zeros_like(data_vals),
                    extent=[
                        float(var_data.longitude.min()), float(var_data.longitude.max()),
                        float(var_data.latitude.min()), float(var_data.latitude.max())
                    ],
                    origin='lower', cmap='Blues', vmin=0, vmax=1,
                    transform=ccrs.PlateCarree(), interpolation='nearest'
                )
                cbar = plt.colorbar(im, ax=ax, shrink=0.6)
                cbar.set_ticks([0])
                cbar.set_ticklabels(['0 mm'], fontfamily='Poppins')
                cbar.set_label(display_name, rotation=270, labelpad=20, fontfamily='Poppins')
            else:
                im = ax.pcolormesh(
                    var_data.longitude, var_data.latitude, var_data,
                    cmap=cmap_local, transform=ccrs.PlateCarree(), shading='auto'
                )
                cbar = plt.colorbar(im, ax=ax, shrink=0.6)
                cbar_label = f"{display_name} ({units})" if units else display_name
                cbar.set_label(cbar_label, rotation=270, labelpad=20, fontfamily='Poppins')
                for label in cbar.ax.get_yticklabels():
                    label.set_fontfamily('Poppins')
        else:
            im = ax.pcolormesh(
                var_data.longitude, var_data.latitude, var_data,
                cmap=cmap_local, transform=ccrs.PlateCarree(), shading='auto'
            )
            cbar = plt.colorbar(im, ax=ax, shrink=0.6)
            cbar_label = f"{display_name} ({units})" if units else display_name
            cbar.set_label(cbar_label, rotation=270, labelpad=20, fontfamily='Poppins')
            for label in cbar.ax.get_yticklabels():
                label.set_fontfamily('Poppins')

    # Extensió del mapa
    lon_min = float(var_data.longitude.min())
    lon_max = float(var_data.longitude.max())
    lat_min = float(var_data.latitude.min())
    lat_max = float(var_data.latitude.max())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.8,
                      linestyle='--', x_inline=False, y_inline=False)
    gl.xlocator = plt.MultipleLocator(0.25)
    gl.ylocator = plt.MultipleLocator(0.25)
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.xlabel_style = {'fontfamily': 'Poppins'}
    gl.ylabel_style = {'fontfamily': 'Poppins'}

    # Títol
    if title is None:
        title = f"{display_name} — {date_str}" if date_str else display_name

    if var_name in ['t2m', 'rh', 'wind10m', 'rain_24h']:
        title += " — 11:00 h"

    ax.set_title(title, fontsize=14, fontfamily='Poppins')

    plt.tight_layout()
    return fig

