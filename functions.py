import pandas as pd
import numpy as np
import streamlit as st
import base64
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature



def img_to_data_uri(path: str) -> str:
    mime = "image/svg+xml" if path.lower().endswith(".svg") else "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode("utf-8")


# ==========================
# Función de plotting
# ==========================

def plot_variable_cartopy(ds, var_name, variable_cmaps, variable_display_names, 
                          title=None, fire_location=None, time_index=0):
    """
    Plot a variable from the dataset using Cartopy projection.
    If multiple times exist, select one via `time_index`.
    """
    display_name = variable_display_names.get(var_name, var_name)
    cmap_local = variable_cmaps.get(var_name, 'viridis')

    var_data = ds[var_name]

    # ---  Seleccionar una data si hi ha múltiples temps ---
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

    # --- Resta del codi de ploteig ---
    if var_name == 'FWI_risk':
        colors = ['green', 'yellow', 'orange', 'red', 'darkred']
        levels = [0, 1, 2, 3, 4, 5]
        labels = ['Bajo', 'Moderado', 'Alto', 'Muy Alto', 'Extremo']

        risk = var_data.astype(int)
        im = ax.contourf(
            risk.longitude, risk.latitude, risk,
            levels=levels, colors=colors, transform=ccrs.PlateCarree()
        )

        cbar = plt.colorbar(im, ax=ax, shrink=0.6, ticks=np.arange(0.5, 5.5))
        cbar.set_ticklabels(labels)
        cbar.set_label(display_name, rotation=270, labelpad=20)

    else:
        # --- Unitats segons variable ---
        units_dict = {
            't2m': '°C',
            'rh': '%',
            'wind10m': 'km/h',
            'rain_24h': 'mm'
        }
        units = units_dict.get(var_name, None)

        # --- Control especial per precipitació ---
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
                    transform=ccrs.PlateCarree()
                )
                cbar = plt.colorbar(im, ax=ax, shrink=0.6)
                cbar.set_ticks([0])
                cbar.set_ticklabels(['0 mm'])
                cbar.set_label(display_name, rotation=270, labelpad=20)
            else:
                im = ax.contourf(
                    var_data.longitude, var_data.latitude, var_data,
                    levels=20, cmap=cmap_local, transform=ccrs.PlateCarree()
                )
                cbar = plt.colorbar(im, ax=ax, shrink=0.6)
                cbar_label = f"{display_name} ({units})" if units else display_name
                cbar.set_label(cbar_label, rotation=270, labelpad=20)
        else:
            im = ax.contourf(
                var_data.longitude, var_data.latitude, var_data,
                levels=20, cmap=cmap_local, transform=ccrs.PlateCarree()
            )
            cbar = plt.colorbar(im, ax=ax, shrink=0.6)
            cbar_label = f"{display_name} ({units})" if units else display_name
            cbar.set_label(cbar_label, rotation=270, labelpad=20)

    # Punt d'incendi
    if fire_location is not None:
        fire_lon, fire_lat = fire_location
        ax.plot(fire_lon, fire_lat, marker='*', color='black', markersize=15,
                transform=ccrs.PlateCarree(), label='Incendio')
        ax.legend(loc='upper right')

    # Extensió del mapa
    lon_min = float(var_data.longitude.min())
    lon_max = float(var_data.longitude.max())
    lat_min = float(var_data.latitude.min())
    lat_max = float(var_data.latitude.max())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # --- Gridlines i etiquetes ---
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5,
                      linestyle='--', x_inline=False, y_inline=False)
    gl.xlocator = plt.MultipleLocator(0.25)
    gl.ylocator = plt.MultipleLocator(0.25)
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = True
    gl.left_labels = True

    # --- Títol ---
    if title is None:
        if date_str:
            title = f"{display_name} — {date_str}"
        else:
            title = display_name

    # Afegir “a les 11:00 h” si la variable és una de les principals
    if var_name in ['t2m', 'rh', 'wind10m', 'rain_24h']:
        title += " — 11:00 h"

    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    return fig

