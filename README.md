# Physical Layer Security in MIMO PLC Systems

**Undergraduate Thesis — Electrical Engineering, UFPR (2025)**  
**Author:** Pedro Henrique Sartorello  
**Advisor:** Prof. Dr. Ândrei Camponogara

---

## Overview

This work investigates **Physical Layer Security (PLS)** in broadband **Power Line Communication (PLC)** systems using **MIMO** techniques, covering the 1.8–100 MHz frequency range. The central question is: by exploiting multiple conductors in a residential electrical network, can MIMO both increase data rates *and* improve communication secrecy against passive eavesdroppers — without relying on cryptography?

The answer, supported by numerical simulations on synthetic PLC channels, is yes — and the gains are significant.

---

## System Model

The scenario follows the **Csiszár-Körner wiretap channel** framework. Alice (legitimate transmitter), Bob (legitimate receiver), and Eve (passive eavesdropper) all share the same residential electrical network. Alice has no access to Eve's Channel State Information (CSI) — Eve is purely passive and does not inject signals.

![System Model](fig_system_model.png)

- **Alice** transmits using the **Delta (Δ) mode**: Port 1 = Phase-Neutral pair (PN), Port 2 = Phase-Earth pair (PE)
- **Bob and Eve** receive using the **Star (Y) mode**: Port 1 = Phase-reference, Port 2 = Neutral-reference
- The three conductors (Phase, Neutral, Earth) create independent propagation paths, enabling spatial multiplexing via MIMO
---

## Configurations Evaluated

Five wiretap configurations are analyzed, differing in the number of ports used by each party:

![Configurations Table](fig_configurations.png)

The suffixes **SE** (Single Eavesdropper) and **ME** (Multiple Eavesdropper) indicate whether Eve uses one or two receive ports.

---

## Channel and Noise Model

- **Channel bank:** 10,000 independent synthetic PLC channel estimates generated via the top-down statistical MIMO PLC model from Pittolo & Tonello (2016)
- **Subcarriers:** N = 1,727 subcarriers spanning 1.8–100 MHz (Δf ≈ 56.86 kHz)
- **Noise PSD:** Measured from real Brazilian residential power networks (Oliveira et al., 2016)
- **SVD decomposition** is applied to each channel matrix to decompose MIMO into parallel independent streams

---

## Security Metrics

### 1. Secrecy Outage Probability (SOP)

The secrecy capacity is $C_s = [C_b - C_e]^+$. Since Alice cannot track Eve's channel, she fixes a target secrecy rate $R_s$ and evaluates:

$$P_{\text{out}}(R_s) = \mathbb{P}\{C_s < R_s\}$$

Lower SOP = better security.

### 2. Effective Secrecy Throughput (EST)

EST jointly captures **reliability** (Bob decoding correctly) and **secrecy** (Eve failing to decode):

$$\Psi(R_B, R_E) = (R_B - R_E)\,[1 - O_r(R_B)]\,[1 - O_s(R_E)]$$

where $O_r$ is the reliability outage probability and $O_s$ is the secrecy outage probability. Two scenarios are evaluated:

- **Scenario 1:** Alice knows Bob's CSI → sets $R_B = C_b$, optimizes $R_E$
- **Scenario 2:** Alice knows neither CSI → jointly optimizes $(R_B, R_E)$

---

## Key Results

### SOP Performance

![SOP Overview](fig_sop_overview.png)

**MIMO-SE consistently achieves the lowest SOP across all transmission powers and target secrecy rates.** At $P_t = 20$ dBm and $R_s = 1$ b/s/Hz, the SOP values are approximately:

| Configuration | SOP |
|:---|:---:|
| MISO-ME | ~70% |
| SISO-SE | ~65% |
| MIMO-ME | ~59% |
| MISO-SE | ~54% |
| **MIMO-SE** | **~42%** |

The MISO-ME configuration shows a counterintuitive behavior at high power: Eve's MIMO channel capacity grows faster than Bob's MISO capacity under uniform power allocation, causing SOP to plateau around 70%.

### EST Performance

![EST Scenario 1](fig_est_scenario1.png)

At $P_t = 30$ dBm (Scenario 1), MIMO-SE reaches **3.45 b/s/Hz** — more than double the next best configuration. When Alice has no CSI (Scenario 2), the EST drops (as expected), but MIMO-SE still leads at **1.31 b/s/Hz**, compared to 0.39 b/s/Hz for SISO-SE.

**The core takeaway:** MIMO in PLC does not just increase throughput — it creates a structural advantage for the legitimate link that passive eavesdroppers cannot easily overcome without adding more receive ports.

---

## Code Structure

```
.
├── main.py                  # Full simulation pipeline
├── Banco01_1000.mat         # Bob's PLC MIMO channel bank (10^4 realizations)
├── Banco02_1000.mat         # Eve's PLC MIMO channel bank (10^4 realizations)
└── PLC_PSD.mat              # Measured additive noise PSD (Brazilian residential grid)
```

The simulation script covers:

1. **Channel loading** — MATLAB `.mat` files parsed with `scipy.io.loadmat`
2. **Capacity computation** — SISO via direct SNR sum; MIMO via SVD decomposition per subcarrier
3. **SOP** — empirical CDF computed over channel realizations
4. **EST (Scenario 1)** — brute-force optimization over $R_E$ for each channel realization, then averaged
5. **EST (Scenario 2)** — 2D grid search over $(R_B, R_E)$ pairs per power level
6. **Plotting** — all figures exported as PDF at 300+ DPI

---

## Dependencies

```bash
pip install numpy scipy matplotlib tqdm
```

Tested with Python 3.10+. Channel data files (`.mat`) are required to run the simulation — they are generated using the synthetic PLC channel model described in Pittolo & Tonello (2016).

---

## Main References

- Wyner, A. D. (1975). The wire-tap channel. *Bell System Technical Journal*, 54(8).
- Pittolo, A. & Tonello, A. M. (2016). A synthetic MIMO PLC channel model. *IEEE ISPLC*.
- Yan, S. et al. (2015). Optimization of code rates in SISO-ME wiretap channels. *IEEE Trans. Wireless Commun.*, 14(11).
- Camponogara, A. et al. (2021). Physical layer security of in-home PLC systems. *IEEE Systems Journal*, 15(1).
- Oliveira, T. R. et al. (2016). Characterization of hybrid communication channel in indoor scenario. *J. Commun. Inf. Syst.*, 31(1).

---

## Summary

MIMO-SE — where Alice and Bob both use two ports but Eve is limited to one — is the optimal configuration from a physical layer security standpoint. It achieves the lowest secrecy outage probability and the highest effective secrecy throughput in all evaluated scenarios, confirming that MIMO is a viable and effective tool for strengthening PLC security without requiring cryptographic overhead.
