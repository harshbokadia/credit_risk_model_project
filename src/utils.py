"""
utils.py — Shared config, helpers, and plot style for all notebooks.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os
warnings.filterwarnings('ignore')

# ── Colour palette ────────────────────────────────────────────
PALETTE = {
    'default':     '#E24B4A',
    'non_default': '#378ADD',
    'neutral':     '#888780',
    'highlight':   '#534AB7',
    'accent':      '#1D9E75',
    'amber':       '#EF9F27',
}

# ── Chart style ───────────────────────────────────────────────
CHART_BG   = '#1E293B'
CHART_PAGE = '#0F172A'
TICK_COLOR = '#94A3B8'
TEXT_COLOR = '#E2E8F0'
GRID_COLOR = '#334155'

plt.rcParams.update({
    'figure.facecolor':  CHART_PAGE,
    'axes.facecolor':    CHART_BG,
    'axes.edgecolor':    '#334155',
    'axes.labelcolor':   TICK_COLOR,
    'axes.grid':         True,
    'grid.color':        GRID_COLOR,
    'grid.alpha':        0.4,
    'xtick.color':       TICK_COLOR,
    'ytick.color':       TICK_COLOR,
    'text.color':        TEXT_COLOR,
    'legend.facecolor':  CHART_BG,
    'legend.edgecolor':  '#334155',
    'legend.labelcolor': TEXT_COLOR,
    'axes.titlecolor':   TEXT_COLOR,
    'font.size':         11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

# ── Column reference ──────────────────────────────────────────
TARGET     = 'default.payment.next.month'
CAT_COLS   = ['SEX', 'EDUCATION', 'MARRIAGE']
PAY_STATUS = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
BILL_COLS  = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
              'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
PAY_COLS   = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
              'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

LABEL_MAPS = {
    'SEX':       {1: 'Male',       2: 'Female'},
    'EDUCATION': {1: 'Graduate',   2: 'University',
                  3: 'High School',4: 'Other'},
    'MARRIAGE':  {1: 'Married',    2: 'Single', 3: 'Other'},
}

# ── Helpers ───────────────────────────────────────────────────
def load_data(path='data/credit_card_default.csv'):
    """Load raw dataset, drop ID column if present."""
    df = pd.read_csv(path)
    return df.drop(columns=['ID'], errors='ignore')


def save_fig(fig, name, folder='outputs'):
    """Save a matplotlib figure to outputs/ as a PNG."""
    os.makedirs(folder, exist_ok=True)
    fig.savefig(f'{folder}/{name}.png', dpi=150,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved -> {folder}/{name}.png')


def section(title):
    """Print a clear section separator."""
    print(f'\n{"─" * 60}')
    print(f'  {title}')
    print(f'{"─" * 60}')
