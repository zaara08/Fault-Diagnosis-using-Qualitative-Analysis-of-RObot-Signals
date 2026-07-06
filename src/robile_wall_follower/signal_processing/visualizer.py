#!/usr/bin/env python3
"""
Signal Visualizer for Robile Wall Following Data
Creates clear plots of sensor signals with action labels
Perfect for understanding data and including in report!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


# ── CONFIGURATION ─────────────────────────────────────────────
# Colors for each action label
ACTION_COLORS = {
    'idle':                'gray',
    'searching_wall':      'yellow',
    'approaching_wall':    'orange',
    'wall_following':      'green',
    'wall_following_fail': 'red',
}

# Signals to plot
SIGNALS = [
    ('lidar_left',     'LIDAR Left (m)',      'Wall Distance'),
    ('lidar_front',    'LIDAR Front (m)',     'Front Distance'),
    ('odom_linear_x',  'Speed (m/s)',         'Forward Speed'),
    ('cmd_angular_z',  'Angular Cmd (rad/s)', 'Turn Command'),
    ('imu_angular_z',  'IMU Angular (rad/s)', 'Actual Rotation'),
]
# ──────────────────────────────────────────────────────────────


def load_csv(filepath):
    """Loads and cleans CSV file."""
    df = pd.read_csv(filepath)

    # Convert timestamp to seconds from start
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    t0 = df['timestamp'].iloc[0]
    df['time_sec'] = (df['timestamp'] - t0).dt.total_seconds()

    return df


def plot_run(filepath, title, output_dir):
    """
    Creates a complete plot of one CSV file showing:
    - All sensor signals over time
    - Action labels as colored background regions
    """
    print(f'Plotting: {os.path.basename(filepath)}')

    df = load_csv(filepath)

    # ── CREATE FIGURE ──────────────────────────────────────────
    fig, axes = plt.subplots(
        len(SIGNALS), 1,
        figsize=(14, 3 * len(SIGNALS)),
        sharex=True)

    fig.suptitle(
        f'Robile Wall Following — {title}\n'
        f'File: {os.path.basename(filepath)}',
        fontsize=14, fontweight='bold')

    # ── PLOT EACH SIGNAL ───────────────────────────────────────
    for i, (signal, ylabel, signal_name) in enumerate(SIGNALS):
        ax = axes[i]

        if signal not in df.columns:
            ax.text(0.5, 0.5, f'{signal} not found',
                   ha='center', va='center',
                   transform=ax.transAxes)
            continue

        # Plot the signal line
        ax.plot(df['time_sec'], df[signal],
                color='blue', linewidth=1.5,
                label=signal_name)

        # ── ADD ACTION LABEL BACKGROUNDS ───────────────────────
        # Color background based on action label
        prev_label = None
        start_time = 0

        for idx, row in df.iterrows():
            current_label = row['action_label']
            current_time  = row['time_sec']

            if current_label != prev_label:
                if prev_label is not None:
                    color = ACTION_COLORS.get(
                        prev_label, 'white')
                    ax.axvspan(start_time, current_time,
                              alpha=0.2, color=color)
                start_time = current_time
                prev_label = current_label

        # Fill last segment
        if prev_label is not None:
            color = ACTION_COLORS.get(prev_label, 'white')
            ax.axvspan(start_time,
                      df['time_sec'].iloc[-1],
                      alpha=0.2, color=color)

        # ── FORMAT SUBPLOT ─────────────────────────────────────
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

        # Add target line for lidar_left
        if signal == 'lidar_left':
            ax.axhline(y=0.5, color='red',
                      linestyle='--', linewidth=1,
                      label='Target 0.5m')
            ax.legend(loc='upper right', fontsize=9)

    # ── ADD X LABEL ────────────────────────────────────────────
    axes[-1].set_xlabel('Time (seconds)', fontsize=12)

    # ── ADD LEGEND FOR ACTION COLORS ───────────────────────────
    legend_patches = []
    for label, color in ACTION_COLORS.items():
        patch = mpatches.Patch(
            color=color, alpha=0.4, label=label)
        legend_patches.append(patch)

    fig.legend(
        handles=legend_patches,
        loc='lower center',
        ncol=len(ACTION_COLORS),
        fontsize=9,
        title='Action Labels',
        bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # ── SAVE FIGURE ────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(
        output_dir,
        f'{title.replace(" ", "_")}.png')

    plt.savefig(output_filename,
               dpi=150,
               bbox_inches='tight')
    print(f'  Saved: {output_filename}')
    plt.close()

    return output_filename


def plot_comparison(good_filepath, bad_filepath, output_dir):
    """
    Creates a side-by-side comparison of
    one good run vs one bad run.
    Perfect for your report!
    """
    print('\nCreating comparison plot...')

    good_df = load_csv(good_filepath)
    bad_df  = load_csv(bad_filepath)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    fig.suptitle(
        'Good Run vs Bad Run Comparison\n'
        'Robile Wall Following',
        fontsize=14, fontweight='bold')

    # Signals to compare
    compare_signals = [
        ('lidar_left',    'LIDAR Left (m)'),
        ('odom_linear_x', 'Forward Speed (m/s)'),
    ]

    for col, (signal, ylabel) in enumerate(compare_signals):
        # Good run (top row)
        ax_good = axes[0][col]
        ax_bad  = axes[1][col]

        # Plot good run
        if signal in good_df.columns:
            ax_good.plot(
                good_df['time_sec'],
                good_df[signal],
                color='green', linewidth=2)

            # Add action backgrounds
            add_action_backgrounds(ax_good, good_df)

            if signal == 'lidar_left':
                ax_good.axhline(
                    y=0.5, color='red',
                    linestyle='--',
                    label='Target 0.5m')

        ax_good.set_title(
            f'GOOD RUN — {ylabel}', fontsize=11)
        ax_good.set_ylabel(ylabel)
        ax_good.grid(True, alpha=0.3)
        ax_good.legend(fontsize=8)

        # Plot bad run
        if signal in bad_df.columns:
            ax_bad.plot(
                bad_df['time_sec'],
                bad_df[signal],
                color='red', linewidth=2)

            add_action_backgrounds(ax_bad, bad_df)

            if signal == 'lidar_left':
                ax_bad.axhline(
                    y=0.5, color='red',
                    linestyle='--',
                    label='Target 0.5m')

        ax_bad.set_title(
            f'BAD RUN — {ylabel}', fontsize=11)
        ax_bad.set_ylabel(ylabel)
        ax_bad.set_xlabel('Time (seconds)')
        ax_bad.grid(True, alpha=0.3)
        ax_bad.legend(fontsize=8)

    # Add legend
    legend_patches = []
    for label, color in ACTION_COLORS.items():
        patch = mpatches.Patch(
            color=color, alpha=0.4, label=label)
        legend_patches.append(patch)

    fig.legend(
        handles=legend_patches,
        loc='lower center',
        ncol=len(ACTION_COLORS),
        fontsize=9,
        title='Action Labels',
        bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    output_filename = os.path.join(
        output_dir, 'good_vs_bad_comparison.png')
    plt.savefig(output_filename,
               dpi=150,
               bbox_inches='tight')
    print(f'Saved comparison: {output_filename}')
    plt.close()

    return output_filename


def add_action_backgrounds(ax, df):
    """Adds colored background regions for action labels."""
    prev_label = None
    start_time = 0

    for idx, row in df.iterrows():
        current_label = row['action_label']
        current_time  = row['time_sec']

        if current_label != prev_label:
            if prev_label is not None:
                color = ACTION_COLORS.get(
                    prev_label, 'white')
                ax.axvspan(start_time, current_time,
                          alpha=0.2, color=color)
            start_time = current_time
            prev_label = current_label

    if prev_label is not None:
        color = ACTION_COLORS.get(prev_label, 'white')
        ax.axvspan(start_time,
                  df['time_sec'].iloc[-1],
                  alpha=0.2, color=color)


# ── MAIN ───────────────────────────────────────────────────────
if __name__ == '__main__':

    DATA_DIR   = os.path.expanduser('~/rnd_ws/data')
    OUTPUT_DIR = os.path.expanduser('~/rnd_ws/plots')

    # ── PICK YOUR FILES HERE ───────────────────────────────────
    GOOD_FILE = 'wall_data_2026-07-04_13-52-05.csv'
    BAD_FILE  = 'wall_data_2026-07-04_14-43-56.csv'
    # ──────────────────────────────────────────────────────────

    good_path = os.path.join(DATA_DIR, GOOD_FILE)
    bad_path  = os.path.join(DATA_DIR, BAD_FILE)

    # Plot individual runs
    plot_run(good_path, 'Good Run', OUTPUT_DIR)
    plot_run(bad_path,  'Bad Run',  OUTPUT_DIR)

    # Plot comparison
    plot_comparison(good_path, bad_path, OUTPUT_DIR)

    print('\n✅ All plots saved to:', OUTPUT_DIR)
    print('Open the plots folder to see your graphs!')