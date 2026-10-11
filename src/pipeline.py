"""
Traffic Accident Pattern & Geospatial Hotspot Analysis Pipeline
===============================================================
Author: Raghavendra (imrpittala)
Project: SCT_DS_4 - SkillCraft Technology Data Science Internship
Objective: Analyze traffic accident frequencies, severity dynamics across
           environmental and temporal factors, and generate interactive
           geospatial hotspot visualizations.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def set_visual_style():
    """Sets publication-grade matplotlib and seaborn visual theme."""
    sns.set_theme(style="whitegrid", palette="tab10")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 1.0,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "figure.titlesize": 16,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def load_data(filepath: str) -> pd.DataFrame:
    """Loads accident records from CSV."""
    logger.info(f"Loading accident dataset from: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {df.shape[0]} accident records with {df.shape[1]} attributes.")
    return df


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses datetime and extracts cyclical and discrete temporal variables:
    - Hour, Minute
    - Day_of_Week (Monday to Sunday)
    - Month, Day_Name
    - Rush_Hour (Morning: 7-10 AM, Evening: 4-7 PM)
    - Is_Weekend
    """
    logger.info("Extracting temporal dimensions...")
    df_temp = df.copy()
    df_temp["Start_Time"] = pd.to_datetime(df_temp["Start_Time"])
    
    df_temp["Hour"] = df_temp["Start_Time"].dt.hour
    df_temp["Month"] = df_temp["Start_Time"].dt.month
    df_temp["Day_Name"] = df_temp["Start_Time"].dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df_temp["Day_Name"] = pd.Categorical(df_temp["Day_Name"], categories=day_order, ordered=True)
    
    df_temp["Is_Weekend"] = df_temp["Day_Name"].isin(["Saturday", "Sunday"]).astype(int)
    df_temp["Rush_Hour"] = df_temp["Hour"].apply(
        lambda h: "Morning Peak (7-10 AM)" if 7 <= h <= 10 else
                  "Evening Peak (4-7 PM)" if 16 <= h <= 19 else "Off-Peak"
    )
    return df_temp


def generate_static_report(df: pd.DataFrame, output_image_path: str):
    """
    Generates a 6-panel static executive dashboard:
    1. Hourly Accident Distribution with Rush Hour bands
    2. Weekly Heatmap (Day of Week vs Hour)
    3. Accident Severity Distribution by Weather Condition
    4. Road Surface Conditions vs Severity Levels
    5. Light Conditions & Severity Stacked Proportions
    6. 2D Spatial Density KDE Hotspot Map
    """
    logger.info("Generating static 6-panel analytical visualization...")
    set_visual_style()

    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    plt.subplots_adjust(hspace=0.35, wspace=0.28)

    fig.suptitle(
        "Traffic Accident Pattern & Environmental Hotspot Analysis Dashboard\n"
        "SkillCraft Technology Data Science Internship | Task 4 (SCT_DS_4)",
        fontsize=18,
        fontweight="bold",
        y=0.99
    )

    # 1. Hourly Accident Frequency
    hourly_counts = df["Hour"].value_counts().sort_index()
    axes[0, 0].plot(hourly_counts.index, hourly_counts.values, marker="o", color="#d32f2f", lw=2.5, markersize=6)
    axes[0, 0].axvspan(7, 10, color="#ffcdd2", alpha=0.5, label="Morning Rush (7-10 AM)")
    axes[0, 0].axvspan(16, 19, color="#ffe082", alpha=0.5, label="Evening Rush (4-7 PM)")
    axes[0, 0].set_title("Accident Frequency by Hour of Day", fontweight="bold")
    axes[0, 0].set_xlabel("Hour of Day (0-23)")
    axes[0, 0].set_ylabel("Accident Count")
    axes[0, 0].set_xticks(range(0, 24, 2))
    axes[0, 0].legend(loc="upper left", frameon=True)

    # 2. Day of Week vs Hour Heatmap
    pivot_day_hour = df.pivot_table(index="Day_Name", columns="Hour", values="Accident_ID", aggfunc="count", fill_value=0)
    sns.heatmap(pivot_day_hour, cmap="YlOrRd", ax=axes[0, 1], cbar_kws={"label": "Accident Count"})
    axes[0, 1].set_title("Accident Density Matrix: Day of Week vs Hour", fontweight="bold")
    axes[0, 1].set_xlabel("Hour of Day")
    axes[0, 1].set_ylabel("Day of Week")

    # 3. Severity Distribution across Weather Conditions
    weather_sev = pd.crosstab(df["Weather_Condition"], df["Severity"], normalize="index") * 100
    weather_sev.plot(
        kind="barh",
        stacked=True,
        colormap="coolwarm",
        ax=axes[1, 0],
        edgecolor="black"
    )
    axes[1, 0].set_title("Severity Level Proportions by Weather Condition", fontweight="bold")
    axes[1, 0].set_xlabel("Percentage (%)")
    axes[1, 0].set_ylabel("Weather Condition")
    axes[1, 0].legend(title="Severity Tier", bbox_to_anchor=(1.02, 1), loc="upper left")

    # 4. Road Surface Conditions vs Mean Severity
    road_sev = df.groupby("Road_Surface_Conditions")["Severity"].agg(["count", "mean"]).reset_index()
    road_sev = road_sev.sort_values(by="mean", ascending=False)
    bars = axes[1, 1].bar(
        road_sev["Road_Surface_Conditions"],
        road_sev["mean"],
        color=["#c62828", "#ef6c00", "#1565c0", "#2e7d32"][:len(road_sev)],
        edgecolor="black",
        alpha=0.85
    )
    for bar in bars:
        h = bar.get_height()
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2, h + 0.03,
            f"{h:.2f}", ha="center", fontweight="bold", fontsize=10.5
        )
    axes[1, 1].set_ylim(1.0, 3.5)
    axes[1, 1].set_title("Average Accident Severity by Road Surface Condition", fontweight="bold")
    axes[1, 1].set_xlabel("Road Surface Condition")
    axes[1, 1].set_ylabel("Mean Severity Index (1-4)")

    # 5. Light Conditions Distribution
    light_counts = df["Light_Conditions"].value_counts()
    axes[2, 0].barh(light_counts.index, light_counts.values, color=sns.color_palette("mako", len(light_counts)), edgecolor="black")
    for i, v in enumerate(light_counts.values):
        axes[2, 0].text(v + 15, i, f"{v} ({(v/len(df))*100:.1f}%)", va="center", fontweight="semibold")
    axes[2, 0].set_title("Accident Occurrences by Lighting Condition", fontweight="bold")
    axes[2, 0].set_xlabel("Number of Accidents")
    axes[2, 0].invert_yaxis()

    # 6. Geospatial 2D Density Hotspot Map (KDE Contour)
    sns.scatterplot(
        data=df,
        x="Start_Lng",
        y="Start_Lat",
        hue="Severity",
        palette="Reds",
        alpha=0.4,
        s=20,
        ax=axes[2, 1],
        legend=False
    )
    sns.kdeplot(
        data=df,
        x="Start_Lng",
        y="Start_Lat",
        cmap="Reds",
        fill=True,
        alpha=0.55,
        levels=7,
        ax=axes[2, 1]
    )
    axes[2, 1].set_title("Spatial Accident Hotspots (2D Kernel Density)", fontweight="bold")
    axes[2, 1].set_xlabel("Longitude")
    axes[2, 1].set_ylabel("Latitude")

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Static analytical dashboard saved to: {output_image_path}")


def generate_interactive_folium_map(df: pd.DataFrame, output_html_path: str):
    """
    Creates an interactive Leaflet/Folium geospatial accident map
    with HeatMap layer and MarkerCluster.
    """
    logger.info("Generating interactive Folium geospatial hotspot map...")
    center_lat = float(df["Start_Lat"].mean())
    center_lng = float(df["Start_Lng"].mean())

    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=11,
        tiles="OpenStreetMap"
    )

    # 1. HeatMap layer weighted by Severity
    heat_data = [[row["Start_Lat"], row["Start_Lng"], row["Severity"]] for _, row in df.iterrows()]
    HeatMap(heat_data, radius=13, blur=16, max_zoom=13).add_to(m)

    # 2. Marker Cluster for high severity accidents (Severity >= 3)
    severe_accidents = df[df["Severity"] >= 3]
    marker_cluster = MarkerCluster(name="High Severity Accidents (Tiers 3 & 4)").add_to(m)

    for _, row in severe_accidents.head(150).iterrows():  # Top sample for interactive performance
        folium.CircleMarker(
            location=[row["Start_Lat"], row["Start_Lng"]],
            radius=6,
            popup=(
                f"<b>ID:</b> {row['Accident_ID']}<br>"
                f"<b>Severity:</b> Tier {row['Severity']}<br>"
                f"<b>Weather:</b> {row['Weather_Condition']}<br>"
                f"<b>Road:</b> {row['Road_Surface_Conditions']}<br>"
                f"<b>Time:</b> {row['Start_Time']}"
            ),
            color="#d32f2f" if row["Severity"] == 4 else "#f57c00",
            fill=True,
            fill_opacity=0.85
        ).add_to(marker_cluster)

    folium.LayerControl().add_to(m)
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    m.save(output_html_path)
    logger.info(f"Interactive geospatial map exported to: {output_html_path}")


def export_accident_metrics(df: pd.DataFrame, output_json_path: str):
    """Saves quantitative summaries of accident dynamics to JSON."""
    summary = {
        "total_accidents_analyzed": int(len(df)),
        "overall_mean_severity": round(float(df["Severity"].mean()), 2),
        "severity_counts": df["Severity"].value_counts().to_dict(),
        "rush_hour_breakdown": df["Rush_Hour"].value_counts().to_dict(),
        "weather_severity_averages": {
            w: round(float(s), 2)
            for w, s in df.groupby("Weather_Condition")["Severity"].mean().items()
        },
        "road_condition_severity_averages": {
            r: round(float(s), 2)
            for r, s in df.groupby("Road_Surface_Conditions")["Severity"].mean().items()
        },
        "top_accident_hotspot_sectors": df["Area_Sector"].value_counts().head(5).to_dict()
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Accident metrics exported to: {output_json_path}")


def export_accident_patterns_hotspots(df: pd.DataFrame, output_image_path: str):
    """
    Exports clean, high-DPI (300 DPI) composite visualization:
    (a) Accident frequency by hour-of-day with rush-hour highlight bands.
    (b) Weather condition breakdown and severity proportions.
    (c) Road surface condition severity factors.
    """
    logger.info("Exporting dedicated accident_patterns_hotspots.png...")
    set_visual_style()

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    plt.subplots_adjust(wspace=0.30)

    fig.suptitle(
        "Urban Traffic Accident Patterns: Diurnal Cycles, Weather Sensitivity & Road Surface Hazards",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    # Panel 1: Accident frequency by hour-of-day
    hourly_counts = df["Hour"].value_counts().sort_index()
    axes[0].plot(hourly_counts.index, hourly_counts.values, marker="o", color="#c62828", lw=2.5, markersize=6)
    axes[0].axvspan(7, 10, color="#ffcdd2", alpha=0.5, label="Morning Rush (7-10 AM)")
    axes[0].axvspan(16, 19, color="#ffe082", alpha=0.5, label="Evening Rush (4-7 PM)")
    axes[0].set_title("(a) Accident Volume by Hour of Day", fontweight="bold", pad=12)
    axes[0].set_xlabel("Hour of Day (0-23)")
    axes[0].set_ylabel("Accident Frequency")
    axes[0].set_xticks(range(0, 24, 3))
    axes[0].legend(loc="upper left", frameon=True)

    # Panel 2: Weather condition breakdown
    weather_sev = pd.crosstab(df["Weather_Condition"], df["Severity"], normalize="index") * 100
    weather_sev.plot(
        kind="barh",
        stacked=True,
        colormap="coolwarm",
        ax=axes[1],
        edgecolor="black"
    )
    axes[1].set_title("(b) Collision Severity by Weather State", fontweight="bold", pad=12)
    axes[1].set_xlabel("Severity Composition (%)")
    axes[1].set_ylabel("Weather Condition")
    axes[1].legend(title="Severity Tier", loc="lower right", frameon=True, fontsize=9)

    # Panel 3: Road surface severity factors
    road_sev = df.groupby("Road_Surface_Conditions")["Severity"].agg(["count", "mean"]).reset_index()
    road_sev = road_sev.sort_values(by="mean", ascending=False)
    bars = axes[2].bar(
        road_sev["Road_Surface_Conditions"],
        road_sev["mean"],
        color=["#b71c1c", "#e65100", "#1565c0", "#2e7d32"][:len(road_sev)],
        edgecolor="black",
        alpha=0.88
    )
    for bar in bars:
        h = bar.get_height()
        axes[2].text(
            bar.get_x() + bar.get_width() / 2, h + 0.04,
            f"{h:.2f}", ha="center", fontweight="bold", fontsize=11
        )
    axes[2].set_ylim(1.0, 3.2)
    axes[2].set_title("(c) Mean Severity by Road Surface State", fontweight="bold", pad=12)
    axes[2].set_xlabel("Road Surface Condition")
    axes[2].set_ylabel("Mean Severity Index (1-4)")

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Accident patterns figure exported to: {output_image_path}")


def main():
    """Main execution function for traffic accident analysis."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "dataset.csv")
    report_img_path = os.path.join(base_dir, "reports", "visual_insights.png")
    composite_img_path = os.path.join(base_dir, "reports", "accident_patterns_hotspots.png")
    report_map_path = os.path.join(base_dir, "reports", "accident_hotspots.html")
    metrics_json_path = os.path.join(base_dir, "reports", "accident_summary.json")

    logger.info("Starting SCT_DS_4 Traffic Accident Pipeline...")
    df_raw = load_data(data_path)
    df_temp = extract_temporal_features(df_raw)

    generate_static_report(df_temp, report_img_path)
    export_accident_patterns_hotspots(df_temp, composite_img_path)
    generate_interactive_folium_map(df_temp, report_map_path)
    export_accident_metrics(df_temp, metrics_json_path)
    logger.info("SCT_DS_4 pipeline completed successfully.")


if __name__ == "__main__":
    main()
