"""Plotting methods"""

import math
import string
import textwrap
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy.stats import pearsonr
import seaborn as sns
import pandas as pd
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geocollection import GeoQuadMesh

_MODEL_COLS = {
    "FATES": "#3B7D23",
    "CLM-FATES": "#3B7D23",
    "CLM-FATES standard configuration": "#3B7D23",
    "CLM": "#78206E",
    "CLM standard configuration": "#78206E",
    "CLM-FATES parameter update": "#2066a8",
    "CLM-FATES parameter & water stress update": "#ea801c",
}

_MODEL_MARKERS = {
    "FATES only": "o",
    "CLM only": "o",
    "common": "^",
}

_CATEGORY_LABELS = {
    "hydrology": "Hydrology",
    "biophysics": "Biophysics",
    "stomatal": "Stomatal \nConductance & \nPhotosynthesis",
    "biogeochemistry": "Biogeochemistry",
    "land use": "Land Use",
    "fire": "Fire",
}

_CATEGORY_COLORS = {
    "hydrology": "#104E8B",
    "biophysics": "#8B008B",
    "stomatal": "#008B00",
    "biogeochemistry": "#8B5A2B",
}

_SUBCATEGORY_LABELS = {
    "fire": "Fire",
    "land use": "Land use",
    "allocation": "Allocation",
    "allometry": "Allometry",
    "decomposition": "Decomposition",
    "mortality": "Mortality",
    "nutrient uptake": "Nutrient uptake",
    "phenology": "Phenology",
    "recruitment": "Recruitment",
    "respiration": "Respiration",
    "vegetation dynamics": "Vegetation dynamics",
    "acclimation": "Acclimation",
    "photosynthesis": "Photosynthesis",
    "biomass heat storage": "Biomass heat storage",
    "bhs": "Biomass heat storage",
    "LUNA": "LUNA",
    "vegetation water": "Vegetation water",
    "canopy aerodynamics": "Canopy Aerodynamics",
    "canopy evaporation": "Canopy Evaporation",
    "radiation": "Radiation",
    "soil water": "Soil hydraulics",
    "snow": "Snow",
    "thermal": "Soil thermal properties",
}

_BIOME_NAMES = {
    0.0: "Ice sheet",
    1.0: "Tropical rain forest",
    2.0: "Tropical seasonal forest/savanna",
    3.0: "Subtropical desert",
    4.0: "Temperate rain forest",
    5.0: "Temperate seasonal forest",
    6.0: "Woodland/shrubland",
    7.0: "Temperate grassland/desert",
    8.0: "Boreal forest",
    9.0: "Tundra",
}


def choose_subplot_dimensions(num_plots: int) -> tuple[int, int]:
    """Chooses a nice array size/dimension for plotting subplots based on the total
    number of input plots

    Args:
        num_plots (int): total number of plots

    Returns:
        tuple[int, int]: nrow, ncol for subplot dimensions
    """

    if num_plots < 2:
        return num_plots, 1
    if num_plots < 11:
        return math.ceil(num_plots / 2), 2
    # I've chosen to have a maximum of 3 columns
    return math.ceil(num_plots / 3), 3


def generate_subplots(
    num_plots: int,
    row_wise: bool = False,
    width=13,
    height=6,
) -> tuple[plt.figure, np.ndarray]:
    """Generates subplots based on the number of input plots and adds ticks for the last axis in
    each column

    Args:
        num_plots (int): number of plots
        row_wise (bool, optional): row wise?. Defaults to False.

    Returns:
        tuple[plt.figure, np.ndarray]: figure, array of axes
    """

    nrow, ncol = choose_subplot_dimensions(num_plots)
    figure, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(width, height),
        subplot_kw=dict(projection=ccrs.Robinson()),
        layout="compressed",
    )

    if not isinstance(axes, np.ndarray):
        return figure, np.array([axes])
    else:
        axes = axes.flatten(order=("C" if row_wise else "F"))
        for idx, ax in enumerate(axes[num_plots:]):
            figure.delaxes(ax)
            # Turn ticks on for the last ax in each column, wherever it lands
            idx_to_turn_on_ticks = (
                idx + num_plots - ncol if row_wise else idx + num_plots - 1
            )
            for tk in axes[idx_to_turn_on_ticks].get_xticklabels():
                tk.set_visible(True)
        axes = axes[:num_plots]

        return figure, axes


def map_function(
    ax: plt.Axes,
    dat: xr.DataArray,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    diverging_cmap: bool = False,
) -> GeoQuadMesh:
    """Plots a color mesh along with coastlines and ocean for a global data array

    Args:
        ax (plt.Axes): axes to plot on
        dat (xr.DataArray): data array to plot
        title (str): title of subplot/axes
        cmap (str): colormap to use
        vmax (float): maximum value for colormap
        vmin (float): minimum value for colormap
        diverging_cmap (bool, optional): whether a diverging colormap is used. Defaults to False.

    Returns:
        GeoQuadMesh: color mesh
    """

    # if we have a diverging colormap, make the min/max values even
    if diverging_cmap:
        vmin = min(vmin, -1.0 * vmax)
        vmax = max(vmax, np.abs(vmin))

    # add title, coastlines, ocean
    ax.set_title(title, loc="left", fontsize="large", fontweight="bold")
    ax.coastlines()
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "ocean", "110m", facecolor="white")
    )
    # plot the color mesh
    pcm = ax.pcolormesh(
        dat.lon,
        dat.lat,
        dat,
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    return pcm


def get_biome_palette() -> tuple[dict[float, str], str]:
    """Returns a palette for plotting whittaker biomes

    Returns:
       tuple[dict[float, str], str]: color palette, biome names
    """

    # set the hue palette as a dict for custom mapping
    biome_names = [
        "Ice sheet",
        "Tropical rain forest",
        "Tropical seasonal forest/savanna",
        "Subtropical desert",
        "Temperate rain forest",
        "Temperate seasonal forest",
        "Woodland/shrubland",
        "Temperate grassland/desert",
        "Boreal forest",
        "Tundra",
    ]
    colors = [
        "#ADADC9",
        "#317A22",
        "#A09700",
        "#DCBB50",
        "#75A95E",
        "#97B669",
        "#D16E3F",
        "#FCD57A",
        "#A5C790",
        "#C1E1DD",
    ]

    palette = {}
    for i in range(len(colors)):
        palette[float(i)] = colors[i]

    return palette, biome_names


def plot_whittaker_biomes(whit_ds, lats=None, lons=None, height=6, width=12):

    colors, names = get_biome_palette()
    cmap = matplotlib.colors.ListedColormap(list(colors.values()))

    figure, axes = generate_subplots(1, height=height, width=width)
    pcm = map_function(axes[0], whit_ds, None, cmap, -0.5, 9.5)
    cbar = figure.colorbar(pcm, ax=axes[0], fraction=0.03, orientation="vertical")
    if lats is not None:
        axes[0].scatter(
            lons, lats, s=15, c="none", edgecolor="black", transform=ccrs.PlateCarree()
        )
    cbar.set_label("Biome", size=12, fontweight="bold")
    cbar.set_ticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    cbar.set_ticklabels(names)


def plot_oaat_params(all_params: pd.DataFrame, nonzero_params: dict):

    clm_only_params = all_params[
        all_params.parameter_name.isin(nonzero_params["clm_only"])
    ].copy()
    clm_only_params["type"] = "CLM only"

    fates_only_params = all_params[
        all_params.parameter_name.isin(nonzero_params["fates_only"])
    ].copy()
    fates_only_params["type"] = "FATES only"

    common_params = all_params[
        all_params.parameter_name.isin(nonzero_params["common"])
    ].copy()
    common_params["type"] = "common"

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    le_fates = plot_model_oaat(axes[0], fates_only_params, "FATES Only")
    le_clm = plot_model_oaat(axes[1], clm_only_params, "CLM Only")
    le_joint = plot_model_oaat(axes[2], common_params, "Common")

    legend_elements = le_fates
    legend_elements.update(le_clm)
    legend_elements.update(le_joint)

    handles = legend_elements.values()
    labels = [_CATEGORY_LABELS.get(label, label) for label in legend_elements.keys()]
    axes[2].legend(
        handles=handles,
        labels=labels,
        title=None,
        loc="lower center",
        bbox_to_anchor=(1.5, 0.5),
        ncol=1,
    )
    for k in range(3):
        label = string.ascii_lowercase[k]
        axes[k].text(
            -0.1,
            1.1,
            f"({label})",
            transform=axes[k].transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
        )

    fig.delaxes(axes[3])
    plt.tight_layout()


def plot_model_oaat(ax, param_dat: pd.DataFrame, model: str):
    """Plots a column graph of number of parameters in OAAT ensemble
    Args:
        param_dat (pd.DataFrame): data frame with information about parameters
        model (str): model name for title
    """

    # count up totals, update names of subcategory
    param_counts_total = (
        param_dat.groupby(["category", "subcategory"]).size().reset_index(name="num")
    )
    param_counts_total["subcategory_label"] = param_counts_total["subcategory"].map(
        _SUBCATEGORY_LABELS
    )

    # createa pivot table
    pivot = param_counts_total.pivot_table(
        index=["subcategory_label", "category"], values="num", fill_value=0
    )

    # re-order subcategory
    subcategory_order = [
        "Fire",
        "Land use",
        "Allocation",
        "Allometry",
        "Decomposition",
        "Mortality",
        "Nutrient uptake",
        "Phenology",
        "Recruitment",
        "Respiration",
        "Vegetation dynamics",
        "LUNA",
        "Acclimation",
        "Photosynthesis",
        "Vegetation water",
        "Biomass heat storage",
        "Radiation",
        "Canopy Aerodynamics",
        "Canopy Evaporation",
        "Soil thermal properties",
        "Soil hydraulics",
        "Snow",
    ]

    # get rid of subcategories that are not present
    subcategory_present = param_counts_total["subcategory_label"].unique()
    filtered_order = [s for s in subcategory_order if s in subcategory_present]

    subcategory_to_y = {label: i for i, label in enumerate(filtered_order)}

    # re-order pivot table
    pivot = pivot.reset_index()
    pivot = pivot[pivot["subcategory_label"].isin(subcategory_order)]
    pivot["y"] = pivot["subcategory_label"].map(subcategory_to_y)
    pivot = pivot.sort_values("y")

    legend_elements = param_col_graph(ax, pivot)

    yticks = [
        subcategory_to_y[label]
        for label in subcategory_order
        if label in pivot["subcategory_label"].unique()
    ]
    yticklabels = [
        label
        for label in subcategory_order
        if label in pivot["subcategory_label"].unique()
    ]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xticklabels(["0", "5", "10", "15", "20"])

    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("")
    ax.set_title(f"{model} Parameters")

    return legend_elements


def param_col_graph(ax, pivot):

    legend_elements = {}
    for _, row in pivot.iterrows():

        y = row["y"]
        category = row["category"]
        val = row["num"]
        color = _CATEGORY_COLORS.get(category, "#cccccc")

        ax.barh(y, val, color=color)
        t = ax.text(
            val - 0.72,
            y,
            str(int(val)),
            va="center",
            ha="center",
            fontsize=12,
            c="white",
            weight="bold",
        )
        legend_elements[category] = Patch(facecolor=color, label=category)

    return legend_elements


def plot_cumulative_variance(variables, variances, var_dict):

    n_vars = len(variables)
    ncols = 2
    nrows = (n_vars + 1) // 2

    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(10, 4 * nrows), sharex=False
    )

    for idx, variable in enumerate(variables):
        row = idx // ncols
        col = idx % ncols
        ax = axs[row, col] if nrows > 1 else axs[col]

        width = 1.5
        fates_cumvar = variances[variable]["FATES"]
        clm_cumvar = variances[variable]["CLM"]
        ax.bar(
            fates_cumvar.nparams[:10] - width / 2,
            fates_cumvar[:10],
            width=width,
            label="CLM-FATES",
            color=_MODEL_COLS["FATES"],
        )
        ax.bar(
            clm_cumvar.nparams[:10] + width / 2,
            clm_cumvar[:10],
            width=width,
            label="CLM",
            color=_MODEL_COLS["CLM"],
        )

        ax.set_ylim([0, 1])
        ax.set_xticks(clm_cumvar.nparams[:10])
        ax.set_title(var_dict[variable]["long_name"], fontsize=14)
        ax.tick_params(axis="both", labelsize=14)
        label = string.ascii_lowercase[idx]
        ax.text(
            -0.15,
            1.12,
            f"({label})",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
        )

    fig.text(0.5, 0.07, "Number of parameters", ha="center", va="center", fontsize=14)
    fig.text(
        0.04,
        0.5,
        "Fraction of total variance",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=14,
    )

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=14,
        title_fontsize=14,
        ncol=2,
        frameon=False,
    )
    plt.tight_layout(rect=[0.05, 0.08, 1, 1])


def plot_variable_variance(
    ax,
    ds1,
    ds2,
    ds1_name,
    ds2_name,
    ds1_default,
    ds2_default,
    variable,
    long_name,
    units,
    obs_range=None,
):

    both = pd.concat([ds1, ds2])

    ds1_mean_val = ds1[variable].mean()
    ds1_yerr = np.array(
        [ds1_mean_val - ds1[variable].min(), ds1[variable].max() - ds1_mean_val]
    )

    ds2_mean_val = ds2[variable].mean()
    ds2_yerr = np.array(
        [ds2_mean_val - ds2[variable].min(), ds2[variable].max() - ds2_mean_val]
    )
    g = sns.stripplot(
        data=both,
        x="model_name",
        y=variable,
        hue="category",
        jitter=True,
        dodge=True,
        alpha=0.7,
        ax=ax,
        size=6,
        palette=_CATEGORY_COLORS,
    )

    ax.errorbar(
        x=[ds1_name],
        y=ds1_mean_val,
        yerr=[[ds1_yerr[0]], [ds1_yerr[1]]],
        fmt="o",
        color="black",
        ecolor="black",
        capsize=15,
        markersize=8,
    )

    ax.errorbar(
        x=[ds2_name],
        y=ds2_mean_val,
        yerr=[[ds2_yerr[0]], [ds2_yerr[1]]],
        fmt="o",
        color="black",
        ecolor="black",
        capsize=15,
        markersize=8,
    )

    if obs_range is not None:
        ax.axhspan(obs_range[0], obs_range[1], color="gray", alpha=0.15, zorder=0)

    ax.tick_params(labelsize=14)
    ax.set_xlabel(None, fontsize=14)
    ax.set_ylabel(f"{long_name} ({units})", fontsize=16)


def plot_ensemble_variance(
    active_df,
    ds1_name,
    ds2_name,
    ds1_default,
    ds2_default,
    variables,
    var_dict,
    obs_range=None,
    width=10,
    height=7,
):

    ds1 = active_df[active_df.model == "FATES"].copy()
    ds1["model_name"] = "CLM-FATES"
    ds2 = active_df[active_df.model == "CLM"].copy()
    ds2["model_name"] = "CLM"

    handles, labels = None, None
    fig, axes = plt.subplots(1, len(variables), figsize=(width, height), sharex=True)
    for i, variable in enumerate(variables):
        obs = obs_range[i] if obs_range is not None else None
        plot_variable_variance(
            axes[i],
            ds1,
            ds2,
            ds1_name,
            ds2_name,
            ds1_default,
            ds2_default,
            variable,
            var_dict[variable]["long_name"],
            var_dict[variable]["global_units"],
            obs_range=obs,
        )
        # collect legend once
        if handles is None:
            handles, labels = axes[i].get_legend_handles_labels()
        # remove individual legend
        axes[i].legend_.remove()
        label = string.ascii_lowercase[i]

        axes[i].text(
            -0.13,
            1.04,
            f"({label})",
            transform=axes[i].transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
        )

    spacer_handle = mpatches.Rectangle(
        (0, 0), 0, 0, fill=False, edgecolor="none", visible=False
    )
    mean_proxy = mlines.Line2D(
        [],
        [],
        color="black",
        marker="o",
        linestyle="None",
        markersize=8,
        label="mean & range",
    )
    handles.extend([spacer_handle, mean_proxy])
    labels.extend(["", "Ensemble Mean"])

    if obs_range is not None:
        obs_proxy = mpatches.Patch(color="gray", alpha=0.15)
        handles.extend([obs_proxy])
        labels.extend(["Observational Range"])

    fig.legend(
        handles,
        [_CATEGORY_LABELS.get(lbl, lbl) for lbl in labels],
        title="Parameter Grouping",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        borderaxespad=0.0,
        fontsize=16,
        title_fontsize=16,
    )

    plt.tight_layout()


def plot_2_top_n(
    ds1, ds2, default1, default2, all_params, variable, ylabel, units, obs_range=None
):

    ds1 = pd.merge(ds1, all_params[["parameter_name", "type"]])
    ds2 = pd.merge(ds2, all_params[["parameter_name", "type"]])

    max_val = np.max(
        [
            ds1["max_val"].max(),
            ds1["min_val"].max(),
            ds2["max_val"].max(),
            ds2["min_val"].max(),
        ]
    )
    min_val = np.min(
        [
            ds1["max_val"].min(),
            ds1["min_val"].min(),
            ds2["max_val"].min(),
            ds2["min_val"].min(),
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
    _plot_top_n(axes[0], ds1, default1, variable, obs_range=obs_range)
    _plot_top_n(axes[1], ds2, default2, variable, obs_range=obs_range)

    axes[0].set_xlabel(f"{ylabel} ({units})", fontsize=16)
    axes[1].set_xlabel(f"{ylabel} ({units})", fontsize=16)
    axes[0].set_ylabel("Parameter", fontsize=16)
    axes[0].set_title("CLM-FATES", fontsize=16)
    axes[1].set_title("CLM", fontsize=16)
    axes[0].tick_params(axis="both", labelsize=14)
    axes[1].tick_params(axis="both", labelsize=14)

    for k in range(2):
        label = string.ascii_lowercase[k]
        axes[k].text(
            -0.1,
            1.1,
            f"({label})",
            transform=axes[k].transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
        )

    # create custom legend handles for categories
    category_handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=10,
            label=category,
        )
        for category, color in _CATEGORY_COLORS.items()
    ]
    max_value_handle = mlines.Line2D(
        [],
        [],
        marker="o",
        color="w",
        markerfacecolor="black",
        markersize=9,
        label="Max Value",
    )
    min_value_handle = mlines.Line2D(
        [],
        [],
        marker="o",
        color="w",
        markerfacecolor="none",
        markeredgecolor="black",
        markersize=8,
        markeredgewidth=1,
        label="Min Value",
    )

    only_value_handle = mlines.Line2D(
        [],
        [],
        marker="o",
        color="w",
        markerfacecolor="black",
        markersize=9,
        label="FATES- or CLM-only",
    )
    common_value_handle = mlines.Line2D(
        [],
        [],
        marker="^",
        color="w",
        markerfacecolor="black",
        markersize=9,
        label="Common",
    )

    default_line_handle = mlines.Line2D(
        [0], [0], color="black", linestyle="--", linewidth=1, label="Default Value"
    )

    # combine the category legend and the min/max markers into one legend
    handles = [
        max_value_handle,
        min_value_handle,
        only_value_handle,
        common_value_handle,
        default_line_handle,
    ] + category_handles
    labels = [
        "Max parameter",
        "Min parameter",
        "FATES- or CLM-only",
        "Common",
        "Default",
    ] + [_CATEGORY_LABELS[label] for label in list(_CATEGORY_COLORS.keys())]

    # add some white space
    handles = handles[:2] + [mlines.Line2D([], [], color="white")] + handles[2:]
    labels = labels[:2] + [""] + labels[2:]

    handles = handles[:5] + [mlines.Line2D([], [], color="white")] + handles[5:]
    labels = labels[:5] + [""] + labels[5:]

    handles = handles[:7] + [mlines.Line2D([], [], color="white")] + handles[7:]
    labels = labels[:7] + [""] + labels[7:]

    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper left",
        bbox_to_anchor=(0.99, 0.99),
        fontsize=14,
        title_fontsize=16,
        frameon=False,
    )
    plt.tight_layout()


def _plot_top_n(ax, ds, default, variable, biome_label=None, obs_range=None):

    ds["parameter_name"] = [name.replace("fates_", "") for name in ds["parameter_name"]]
    # loop through the rows of the sorted dataframe
    for _, row in ds.iterrows():

        # get the color for the current category
        category_color = _CATEGORY_COLORS.get(row["category"], "#000000")

        # line connecting min and max variable
        ax.plot(
            [row["min_val"], row["max_val"]],
            [row["parameter_name"], row["parameter_name"]],
            color=category_color,
            linewidth=1,
            zorder=1,
        )

    for model, group in ds.groupby("type"):
        # plot max values as filled circles with category color
        ax.scatter(
            group["max_val"],
            group["parameter_name"],
            c=group["category"].map(_CATEGORY_COLORS),
            marker=_MODEL_MARKERS[model],
            label=None,
            zorder=2,
        )

        # plot min values as open circles with category color
        ax.scatter(
            group["min_val"],
            group["parameter_name"],
            facecolors="none",
            edgecolors=group["category"].map(_CATEGORY_COLORS),
            marker=_MODEL_MARKERS[model],
            label=None,
            zorder=2,
        )

    if obs_range is not None:
        ax.axvspan(obs_range[0], obs_range[1], color="gray", alpha=0.1, zorder=0)

    ax.axvline(
        x=default[variable].mean().values, color="k", linestyle="--", linewidth=1
    )

    ax.set_title(biome_label, fontsize=16)
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(axis="both", labelsize=16)


def plot_relative_diffs(df, vars, var_dict, include_sd=True):

    fig, axes = plt.subplots(2, 1, figsize=(7, 12))
    axes = axes.flatten()
    plot_var_diff(
        axes[0], df, vars[0], 1, var_dict[vars[0]]["long_name"], include_sd=include_sd
    )
    plot_var_diff(
        axes[1], df, vars[1], 1, var_dict[vars[0]]["long_name"], include_sd=include_sd
    )
    axes[1].legend(loc="best")
    for k in range(2):
        label = string.ascii_lowercase[k]
        axes[k].text(
            -0.2,
            1.01,
            f"({label})",
            transform=axes[k].transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
        )
    plt.tight_layout()


def plot_var_diff(
    ax, df, variable, tol, long_name, units=None, param_sub=None, include_sd=False
):

    if param_sub is not None:
        df = df[df.index.isin(param_sub)]

    if units is None:
        units = "%"
        type = "Percent"
    else:
        type = "Absolute"

    if include_sd:
        var_name = f"{variable}_mean"
        df[f"{variable}_se"] = 2.0 * df[f"{variable}_sd"] / np.sqrt(20)
    else:
        var_name = variable

    # filter out parameters below tolerence
    below_tol = df.groupby("parameter")[var_name].apply(
        lambda g: (np.abs(g) <= tol).all().all()
    )
    params_to_keep = below_tol[~below_tol].index
    df_filtered = df[df["parameter"].isin(params_to_keep)].copy()

    # sort parameters by magnitude
    df_filtered["abs_diff"] = np.abs(df_filtered[var_name])
    sorted_params = (
        df_filtered.groupby("parameter")["abs_diff"]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )

    # get the unique models
    models = df_filtered["model"].unique()
    bar_height = 0.35

    # offset bars for each model
    for i, model in enumerate(models):
        subset = (
            df_filtered[df_filtered["model"] == model]
            .set_index("parameter")
            .reindex(sorted_params)
        )

        # Position bars vertically based on parameter index
        y_positions = np.arange(len(sorted_params)) + (i - 0.5) * bar_height

        ax.barh(
            y=y_positions,
            width=subset[var_name],
            height=bar_height,
            label="CLM-FATES" if model == "FATES" else model,
            xerr=subset[f"{variable}_se"] if include_sd else None,
            color=_MODEL_COLS[model],
            capsize=4,
            alpha=0.9,
            edgecolor="black",
        )

        # set y-ticks to parameter names
        ax.set_yticks(np.arange(len(sorted_params)))
        ax.set_yticklabels(sorted_params)

        label = f"{type} Difference in {long_name} ({units})"
        wrapped_label = "\n".join(
            textwrap.wrap(label, width=20)
        )  # adjust width as needed
        ax.set_xlabel(wrapped_label)
        ax.set_ylabel("Parameter")


def plot_top_n(
    ds,
    default,
    all_params,
    variable,
    ylabel,
    units,
    xmin=None,
    xmax=None,
    by_biomes=False,
    width=15,
):

    ds = pd.merge(ds, all_params[["parameter_name", "type"]])

    if by_biomes:

        biomes = np.unique(ds.biome.values)
        biomes = biomes[1:]  # don't plot ice
        n_biomes = len(biomes)

        # determine grid size
        ncols = 2
        nrows = math.ceil(n_biomes / ncols)
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(width, 12), sharex=False
        )
        axes = axes.flatten()

        for i, biome in enumerate(biomes):

            biome_default = default.sel(biome=biome)
            biome_ds = ds[ds.biome == biome].copy()

            _plot_top_n(axes[i], biome_ds, biome_default, variable, _BIOME_NAMES[biome])

            label = string.ascii_lowercase[i]
            axes[i].text(
                -0.2,
                1.2,
                f"({label})",
                transform=axes[i].transAxes,
                fontsize=14,
                fontweight="bold",
                va="top",
                ha="left",
            )

        for j in range(len(biomes), len(axes)):
            fig.delaxes(axes[j])

    else:

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        _plot_top_n(ax, ds, default, variable)

    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin, xmax)

    # create custom legend handles for categories
    category_handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=10,
            label=category,
        )
        for category, color in _CATEGORY_COLORS.items()
    ]
    max_value_handle = mlines.Line2D(
        [],
        [],
        marker="o",
        color="w",
        markerfacecolor="black",
        markersize=9,
        label="Max Value",
    )
    min_value_handle = mlines.Line2D(
        [],
        [],
        marker="o",
        color="w",
        markerfacecolor="none",
        markeredgecolor="black",
        markersize=8,
        markeredgewidth=1,
        label="Min Value",
    )

    only_value_handle = mlines.Line2D(
        [],
        [],
        marker="o",
        color="w",
        markerfacecolor="black",
        markersize=9,
        label="FATES- or CLM-only",
    )
    common_value_handle = mlines.Line2D(
        [],
        [],
        marker="^",
        color="w",
        markerfacecolor="black",
        markersize=9,
        label="Common",
    )

    default_line_handle = mlines.Line2D(
        [0], [0], color="black", linestyle="--", linewidth=1, label="Default Value"
    )

    # combine the category legend and the min/max markers into one legend
    handles = [
        max_value_handle,
        min_value_handle,
        only_value_handle,
        common_value_handle,
        default_line_handle,
    ] + category_handles
    labels = [
        "Max parameter",
        "Min parameter",
        "FATES- or CLM-only",
        "Common",
        "Default",
    ] + [_CATEGORY_LABELS[label] for label in list(_CATEGORY_COLORS.keys())]

    # add some white space
    handles = handles[:2] + [mlines.Line2D([], [], color="white")] + handles[2:]
    labels = labels[:2] + [""] + labels[2:]

    if by_biomes:
        fig.legend(
            handles=handles,
            labels=labels,
            ncols=4,
            loc="lower left",
            bbox_to_anchor=(0.0, -0.15),
            fontsize=16,
            title_fontsize=16,
            frameon=False,
        )
    else:
        plt.legend(
            handles=handles,
            labels=labels,
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            fontsize=16,
            title_fontsize=16,
            frameon=False,
        )

    if by_biomes:
        fig.text(0.5, -0.01, f"{ylabel} ({units})", ha="center", size=18)
        fig.text(0.01, 0.5, "Parmaeter", va="center", rotation="vertical", size=18)
    else:
        plt.xlabel(f"{ylabel} ({units})")
        plt.ylabel("Parameter")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()


def plot_multiple_compare_vars(
    df, fates_params, corresponding_params, param_names, param_order
):

    df["analagous_parameter"] = (
        df["parameter"].map(corresponding_params).fillna(df["parameter"])
    )
    df["analagous_parameter"] = [
        param.replace("fates_", "") for param in df["analagous_parameter"]
    ]

    fig, axes = plt.subplots(2, 2, figsize=(17 * 0.7, 14 * 0.7))
    axes = axes.flatten()
    for k, param in enumerate(param_order):
        df_sub = df[df.analagous_parameter == param]
        g = sns.barplot(
            data=df_sub,
            x="variable_name",
            y="mean_value",
            hue="model_name",
            palette=_MODEL_COLS,
            dodge=True,
            errorbar=None,
            ax=axes[k],
        )
        g.legend_.remove()
        n_models = df_sub["model"].nunique()
        bar_width = 0.8 / n_models
        variable_order = df_sub["variable_name"].unique()
        x_ticks = []
        for i, var in enumerate(variable_order):
            for j, model in enumerate(df_sub["model"].unique()):
                x = i - 0.4 + (j + 0.5) * bar_width
                y = df_sub[
                    (df_sub["variable_name"] == var) & (df_sub["model"] == model)
                ]["mean_value"].values[0]
                yerr = df_sub[
                    (df_sub["variable_name"] == var) & (df_sub["model"] == model)
                ]["sd_value"].values[0]
                axes[k].errorbar(
                    x, y, yerr=np.abs(yerr), fmt="none", ecolor="black", capsize=4
                )
                x_ticks.append(x)
        axes[k].set_xticks(
            ticks=range(len(variable_order)),
            labels=variable_order,
            rotation=45,
            ha="right",
        )
        axes[k].set_ylabel("Mean Percent Difference (%)")
        axes[k].set_xlabel(None)

        axes[k].set_title(param_names[fates_params[k].replace("fates_", "")])

        label = string.ascii_lowercase[k]
        axes[k].text(
            -0.1,
            1.1,
            f"({label})",
            transform=axes[k].transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    # Define your desired order
    order = ["CLM-FATES", "CLM"]

    # Reorder handles and labels based on that list
    ordered = [(h, l) for h, l in zip(handles, labels) if l in order]
    ordered.sort(key=lambda x: order.index(x[1]))
    handles, labels = zip(*ordered)

    fig.legend(
        handles,
        labels,
        loc="lower left",
        ncols=2,
        bbox_to_anchor=(0.4, -0.09),
        title="Model",
        fontsize=12,
        title_fontsize=12,
        markerscale=2,
        handlelength=2,
    )
    plt.tight_layout()


def plot_scatter(df, fates_default, clm_default, in_vars, var_dict):

    # get FATES and CLM defaults
    fates_default_var0 = fates_default[in_vars[0]].values
    fates_default_var1 = fates_default[in_vars[1]].values
    fates_default_var2 = fates_default[in_vars[2]].values
    fates_default_var3 = fates_default[in_vars[3]].values

    clm_default_var0 = clm_default[in_vars[0]].values
    clm_default_var1 = clm_default[in_vars[1]].values
    clm_default_var2 = clm_default[in_vars[2]].values
    clm_default_var3 = clm_default[in_vars[3]].values

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes = axes.flatten()

    # plot 1
    sns.scatterplot(
        data=df,
        x=in_vars[0],
        y=in_vars[1],
        hue="model_name",
        hue_order=["CLM-FATES", "CLM"],
        style="category_subset",
        palette=_MODEL_COLS,
        ax=axes[0],
    )
    axes[0].scatter(
        fates_default_var0,
        fates_default_var1,
        marker="*",
        s=200,
        color="#3B7D23",
        edgecolors="black",
    )
    axes[0].scatter(
        clm_default_var0,
        clm_default_var1,
        marker="*",
        s=200,
        color="#78206E",
        edgecolors="black",
    )

    # plot 2
    sns.scatterplot(
        data=df,
        x=in_vars[2],
        y=in_vars[3],
        hue="model_name",
        style="category_subset",
        hue_order=["CLM-FATES", "CLM"],
        palette=_MODEL_COLS,
        ax=axes[1],
    )
    axes[1].scatter(
        fates_default_var2,
        fates_default_var3,
        marker="*",
        s=200,
        color="#3B7D23",
        edgecolors="black",
    )
    axes[1].scatter(
        clm_default_var2,
        clm_default_var3,
        marker="*",
        s=200,
        color="#78206E",
        edgecolors="black",
    )

    handles, labels = axes[1].get_legend_handles_labels()
    clean_handles_labels = [
        (h, l)
        for h, l in zip(handles, labels)
        if l not in ["model_name", "category_subset"]
    ]

    handles, labels = zip(*clean_handles_labels)
    handles, labels = list(handles), list(labels)

    default_star = mlines.Line2D(
        [],
        [],
        color="none",
        marker="*",
        markerfacecolor="black",
        markeredgecolor="black",
        markersize=12,
        label="default",
    )
    handles.append(default_star)
    labels.append("default")

    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.2, -0.15),
        ncol=2,
        frameon=True,
    )
    legend.set_title(None)

    axes[0].set_xlabel(
        f"{var_dict[in_vars[0]]['long_name']} ({var_dict[in_vars[0]]['global_units']})"
    )
    axes[0].set_ylabel(
        f"{var_dict[in_vars[1]]['long_name']} ({var_dict[in_vars[1]]['global_units']})"
    )

    axes[1].set_xlabel(
        f"{var_dict[in_vars[2]]['long_name']} ({var_dict[in_vars[2]]['global_units']})"
    )
    axes[1].set_ylabel(
        f"{var_dict[in_vars[3]]['long_name']} ({var_dict[in_vars[3]]['global_units']})"
    )

    axes[0].get_legend().remove()
    axes[1].get_legend().remove()

    for k in range(2):
        label = string.ascii_lowercase[k]
        axes[k].text(
            -0.2,
            1.01,
            f"({label})",
            transform=axes[k].transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
        )

    for ax in axes:
        ax.tick_params(axis="x", labelsize=12)
    plt.tight_layout()


def plot_all_mini_oaats(df_list, variables, units, mvs, default_line):
    version_order = [
        "CLM",
        "CLM-FATES standard configuration",
        "CLM-FATES parameter update",
        "CLM-FATES parameter & water stress update",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        plot_top_n_sub(
            df_list[i], variables[i], units[i], ax, mvs[i], default_line=default_line
        )
        label = string.ascii_lowercase[i]
        axes[i].text(
            -0.2,
            1.01,
            f"({label})",
            transform=axes[i].transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
        )

    # create custom legend handles for categories
    version_handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor=_MODEL_COLS[v],
            markersize=10,
            label=v,
        )
        for v in version_order
    ]
    max_value_handle = mlines.Line2D(
        [],
        [],
        marker="o",
        color="w",
        markerfacecolor="black",
        markersize=9,
        label="Max Value",
    )
    min_value_handle = mlines.Line2D(
        [],
        [],
        marker="o",
        color="w",
        markerfacecolor="none",
        markeredgecolor="black",
        markersize=8,
        markeredgewidth=1,
        label="Min Value",
    )
    if default_line:
        linestyle = "--"
        marker = "none"
    else:
        linestyle = "none"
        marker = "x"
    default_handle = mlines.Line2D(
        [0],
        [0],
        color="black",
        linestyle=linestyle,
        marker=marker,
        linewidth=1,
        label="Default Value",
    )

    # combine the category legend and the min/max markers into one legend
    handles = [max_value_handle, min_value_handle, default_handle] + version_handles
    labels = ["Max parameter", "Min parameter", "Default"] + version_order

    fig.legend(
        handles=handles,
        labels=labels,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
    )

    plt.tight_layout()


def plot_top_n_sub(df, variable_name, units, ax, mv=1, default_line=True):

    param_order = [
        "fates_leaf_vcmax25top",
        "fates_leaf_stomatal_intercept",
        "fates_leaf_stomatal_slope_medlyn",
        "fff",
    ]
    version_order = [
        "CLM",
        "CLM-FATES standard configuration",
        "CLM-FATES parameter update",
        "CLM-FATES parameter & water stress update",
    ]
    param_name = {
        "fates_leaf_vcmax25top": "V$_{cmax}$",
        "fates_leaf_stomatal_intercept": "stomatal intercept",
        "fates_leaf_stomatal_slope_medlyn": "stomatal slope",
        "fff": "saturated soil scalar (fff)",
    }

    df["analagous_parameter"] = pd.Categorical(
        df["analagous_parameter"], categories=param_order, ordered=True
    )
    df["version"] = pd.Categorical(
        df["version"], categories=version_order, ordered=True
    )
    df = df.sort_values(["analagous_parameter", "version"]).reset_index(drop=True)

    for i, pname in enumerate(param_order):
        indices = df.index[df["analagous_parameter"] == pname].tolist()
        if len(indices) == 0:
            continue
        if i % 2 == 0:
            ax.axhspan(
                indices[0] - 0.5, indices[-1] + 0.5, color="lightgrey", alpha=0.2
            )
        y_middle = (indices[0] + indices[-1]) / 2
        ax.text(
            df["min_val"].min() - mv,
            y_middle,
            param_name[pname],
            va="center",
            ha="right",
            fontsize=10,
        )

    for i, (_, row) in enumerate(df.iterrows()):
        version_color = _MODEL_COLS.get(row["version"], "#000000")
        ax.plot(
            [row["min_val"], row["max_val"]],
            [i, i],
            color=version_color,
            linewidth=1,
            zorder=1,
        )
        ax.scatter(row["max_val"], i, c=version_color, label=None, zorder=2)
        ax.scatter(
            row["min_val"],
            i,
            facecolors="none",
            edgecolors=version_color,
            label=None,
            zorder=2,
        )
        if not default_line:
            ax.scatter(
                row["default"],
                i,
                color=version_color,
                label=None,
                zorder=2,
                marker="x",
            )

    if default_line:
        for version in version_order:
            sub = df[df.version == version]
            ax.axvline(
                x=sub["default"].mean(),
                color=_MODEL_COLS.get(version, "#000000"),
                linestyle="--",
                linewidth=1,
            )

    ax.set_yticks([])
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel(f"{variable_name} ({units})")
    ax.minorticks_on()
    ax.grid(which="minor", color="lightgray", linestyle=":", linewidth=0.5)

def plot_grid_maps(variables, fates_sparse_glob, fates_ann_means_glob,
                  clm_sparse_glob, clm_ann_means_glob, target_grid,
                  var_dict, length=12, width=8):
    fig, axes = plt.subplots(
        nrows=len(variables),
        ncols=2,
        figsize=(width, length),
        constrained_layout=True,
        subplot_kw=dict(projection=ccrs.Robinson()),
    )
    
    for i, variable in enumerate(variables):
        
        diff_fates = (fates_sparse_glob[variable] - fates_ann_means_glob[variable].mean(dim='year')) * target_grid.landfrac
        diff_clm = (clm_sparse_glob[variable] - clm_ann_means_glob[variable].mean(dim='year')) * target_grid.landfrac
        absmax = max(
            abs(diff_fates.min().values),
            abs(diff_fates.max().values),
            abs(diff_clm.min().values),
            abs(diff_clm.max().values),
        )
        vmin, vmax = -absmax, absmax
        
        plot_diff(diff_fates, 'CLM-FATES',
                  var_dict[variable]['long_name'], var_dict[variable]['annual_units'],
                  axes[i, 0], vmin=vmin, vmax=vmax, add_colorbar=False)
        pcm = plot_diff(diff_clm, 'CLM',
                  var_dict[variable]['long_name'], var_dict[variable]['annual_units'],
                  axes[i, 1], vmin=vmin, vmax=vmax, add_colorbar=False)
        label = string.ascii_lowercase[i]
        axes[i, 0].text(
            -0.2, 1.01, f"({label})",
            transform=axes[i, 0].transAxes,
            fontsize=14,
            fontweight='bold',
            va='top',
            ha='left'
        )
        cbar = fig.colorbar(pcm, ax=axes[i, :], orientation="horizontal", shrink=0.5)
        cbar.set_label(f"$\Delta$ {var_dict[variable]['long_name']} (sparsegrid - fullgrid) ({var_dict[variable]['annual_units']})",
                       size=10, fontweight="bold")

def plot_diff(diff, model, variable_name, units, ax,
              vmin=None, vmax=None, add_colorbar=True):

    pcm = map_function(
        ax,
        diff,
        model,
        "RdBu_r",
        vmin if vmin is not None else diff.min().values,
        vmax if vmax is not None else diff.max().values,
        diverging_cmap=True,
    )

    if add_colorbar:
        cbar = plt.colorbar(pcm, ax=ax, shrink=1.0, orientation="horizontal")
        cbar.set_label(f"Difference in {variable_name} ({units})", size=10, fontweight="bold")

    return pcm

def plot_scatter_compare(full, sparse, variable, units, ax, color):

    mask = ~np.isnan(full) & ~np.isnan(sparse)
    full = full[mask]
    sparse = sparse[mask]
    
    r, _ = pearsonr(full, sparse)
    r2 = r**2
    rmse = np.sqrt(np.mean((sparse - full)**2))
    
    ax.scatter(full, sparse, s=5, alpha=0.5, c=color)
    ax.set_xlabel(f'Sparsegrid {variable} ({units})')
    ax.set_ylabel(f'Fullgrid {variable} ({units})')
    
    lims = [min(full.min(), sparse.min()), max(full.max(), sparse.max())]
    ax.plot(lims, lims, 'k--', linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    text = f"R² = {r2:.2f}\nRMSE = {rmse:.2f}"
    ax.text(
        0.05, 0.95,
        text,
        transform=ax.transAxes,
        va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )

def plot_grid(variables, fates_ann_means_glob, fates_sparse_glob,
             clm_ann_means_glob, clm_sparse_glob, target_grid, var_dict,
             length=12, width=8):
    fig, axes = plt.subplots(
        nrows=len(variables),
        ncols=2,
        figsize=(width, length),
        sharex=False,
        sharey=False
    )
    
    for i, variable in enumerate(variables):
        full_fates = (fates_ann_means_glob[variable].mean(dim='year')*target_grid['landfrac']).values.ravel()
        sparse_fates = (fates_sparse_glob[variable]*target_grid['landfrac']).values.ravel()
        full_clm = (clm_ann_means_glob[variable].mean(dim='year')*target_grid['landfrac']).values.ravel()
        sparse_clm = (clm_sparse_glob[variable]*target_grid['landfrac']).values.ravel()
        
        # FATES subplot
        ax_fates = axes[i, 0]
        plot_scatter_compare(full_fates, sparse_fates,
                             var_dict[variable]['long_name'],
                             var_dict[variable]['annual_units'],
                             ax_fates, color='#3B7D23')
        ax_fates.set_title("CLM-FATES")
        
        # CLM subplot
        ax_clm = axes[i, 1]
        plot_scatter_compare(full_clm, sparse_clm,
                             var_dict[variable]['long_name'],
                             var_dict[variable]['annual_units'],
                             ax_clm, color='#78206E')
        ax_clm.set_title("CLM")
        label = string.ascii_lowercase[i]
        axes[i, 0].text(
            -0.4, 1.01, f"({label})",
            transform=axes[i, 0].transAxes,
            fontsize=14,
            fontweight='bold',
            va='top',
            ha='left'
        )
    
    plt.tight_layout()

def plot_heatmap(summary_df):
    """Plot a heatmap of dataset means and relative differences."""
    # create a mask: Keep only 'Relative Difference (%)' for coloring
    mask = summary_df.copy()
    mask.loc[:, mask.columns != "Relative Difference (%)"] = np.nan
    
    rel_diff = summary_df["Relative Difference (%)"].values
    vmax = np.nanmax(np.abs(rel_diff))
    vmax = min(vmax, 100)
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
        cbar_kws={"label": "Relative Difference (%)"},
    )

    # mannually add text for the other columns (to keep them uncolored)
    for i in range(summary_df.shape[0]):
        for j in range(summary_df.shape[1]):
            if summary_df.columns[j] != "Relative Difference (%)":
                text = (
                    f"{summary_df.iloc[i, j]:.2f}"
                    if not np.isnan(summary_df.iloc[i, j])
                    else ""
                )
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    text,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black",
                )
    plt.yticks(rotation=0)