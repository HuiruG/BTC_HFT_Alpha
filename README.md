#  Bitcoin HFT Alpha Engine: The "Ultimate Zoo"
> **Institutional-Grade Microstructure Signal Discovery using Polars, Kalman Filters & SE-TCN Deep Learning**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Polars](https://img.shields.io/badge/Data-Polars_Turbo-orange)
![PyTorch](https://img.shields.io/badge/Model-SE--TCN-red)
![Status](https://img.shields.io/badge/Status-Research_Complete-success)

##  Executive Summary
This repository hosts a high-frequency trading (HFT) research pipeline designed to extract predictive short-term signals from Bitcoin Level 1/Level 2 market data. Unlike traditional technical analysis, this engine employs **Signal Processing (Kalman Filters)**, **Orthogonal Feature Engineering**, and **Regime Detection** to isolate structural market inefficiencies.

The core model utilizes a **Squeeze-and-Excitation Temporal Convolutional Network (SE-TCN)** to predict directional moves over a 50-bar horizon, validated by an event-driven Triple Barrier backtest engine.

> ** Strategy Verdict:** The alpha is strong enough for **Market Making (Maker)** strategies or VIP-tier accounts. Retail Taker execution (2.5bps fee) is not recommended due to high turnover.

##  Core Architecture

### 1. Data Engineering (Polars)
- **Adaptive Bar Sampling:** Volume/Turnover-clock bars to normalize information flow.
- **Microstructure Features:** Order Flow Imbalance (OFI), Kyle's Lambda, Amihud Liquidity.
- **Signal Denoising:** 1-D Kalman Filter implementation to separate "True Price" from microstructure noise.

### 2. Deep Learning Model (PyTorch)
- **Architecture:** Temporal Convolutional Network (TCN) with Causal Convolutions.
- **Attention Mechanism:** **Squeeze-and-Excitation (SE-Block)** to dynamically weight feature channels.
- **Loss Function:** **Focal Loss** to handle class imbalance (Hold vs. Buy/Sell).

### 3. Robust Evaluation
- **Triple Barrier Method:** Path-dependent labeling (Profit Take, Stop Loss, Time Expiry).
- **Event-Driven Backtest:** Numba-accelerated simulation to account for holding periods and path dependency (avoiding look-ahead bias).

## 4. Project Structure
```bash
├── src/
│   ├── data/           # Polars pipelines (Bars, Cleaning, Labeling)
│   ├── alpha/          # Feature Zoo (Predictive, Microstructure, Regime)
│   ├── stability/      # Kalman Filters & Robustness Checks
│   └── pipeline/       # Training & Inference loops
├── notebooks/          # Research logic & Visualization (Start Here)
├── models/             # Saved PyTorch weights (.pth) - *GitIgnored*
├── data/               # Raw & Processed Data - *GitIgnored*
└── README.md           # This document
