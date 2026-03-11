# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# =============================================================================
# 1) Carregamento de canal e parâmetros
# =============================================================================

# Bob
canais_bob  = loadmat('Banco01_1000.mat')
H_bob       = canais_bob['ans'] # M-by-Nr-by-Nt-by-Nf matrix H 

# Eve
canais_eve  = loadmat('Banco02_1000.mat')
H_eve       = canais_eve['ans']

total_ch    = len(H_bob) # número de canais
N           = 1727              # número de subcanais

# Potência de transmissão (linear)
Pt_dBm = np.arange(-20, 31, 5)  # em dBm
Pt     = 10**((Pt_dBm - 30)/10)  # em Watts

# Ruído PLC Medido
psd_med = loadmat('PLC_PSD.mat')
psd     = psd_med['psd_plc'][:,0] #N = 1727 subcanais

#%%
# =============================================================================
# 2) Cálculo das capacidades para cada cenário (SISO, MISO, MIMO 2x2, MIMO 2x3)
#    Formato final: cada matriz => [len(Pt_dBm), total_ch]
# =============================================================================

########################
# 2.1) SISO (1x1)
########################

### Bob
H_siso_bob          = H_bob[:, 0:1, 0:1, :]  # Nr=1, Nt=1
matrizCap_siso_bob  = np.zeros((len(Pt), total_ch))

R_siso = 1  # rank máximo
Lambda_siso = np.zeros((total_ch, R_siso, N))
for c in range(total_ch):
    for fn in range(N):
        U, S, Vh = np.linalg.svd(H_siso_bob[c, :, :, fn])
        Lambda_siso[c, 0, fn] = S[0]**2

for c in range(total_ch):
    for k in tqdm(range(len(Pt))):
        DeltaPt = (Pt[k] * np.ones(N)) / N  # distribuição igual por subcanal
        SNR = DeltaPt * Lambda_siso[c, 0, :] / (R_siso * psd)
        matrizCap_siso_bob[k, c] = np.sum(np.log2(1 + SNR)) / N
    
### Eve
H_siso_eve          = H_eve[:, 0:1, 0:1, :]  # Nr=1, Nt=1
matrizCap_siso_eve  = np.zeros((len(Pt), total_ch))

R_siso = 1  # rank máximo
Lambda_siso = np.zeros((total_ch, R_siso, N))
for c in range(total_ch):
    for fn in range(N):
        U, S, Vh = np.linalg.svd(H_siso_eve[c, :, :, fn])
        Lambda_siso[c, 0, fn] = S[0]**2

for c in range(total_ch):
    for k in tqdm(range(len(Pt))):
        DeltaPt = (Pt[k] * np.ones(N)) / N  # distribuição igual por subcanal
        SNR = DeltaPt * Lambda_siso[c, 0, :] / (R_siso * psd)
        matrizCap_siso_eve[k, c] = np.sum(np.log2(1 + SNR)) / N
        

########################
# 2.2) MIMO 2x2
########################

### Bob
H_mimo_bob          = H_bob  # Nr=2, Nt=2
matrizCap_mimo_bob  = np.zeros((len(Pt), total_ch))

R_mimo_2x2 = 2
Lambda_mimo_2x2 = np.zeros((total_ch, R_mimo_2x2, N))
for c in range(total_ch):
    for fn in range(N):
        U, S, Vh = np.linalg.svd(H_mimo_bob[c, :, :, fn])
        Lambda_mimo_2x2[c, :, fn] = S**2

for c in range(total_ch):
    for k in tqdm(range(len(Pt))):
        DeltaPt = (Pt[k] * np.ones(N)) / N
        soma_cap = 0
        for r in range(R_mimo_2x2):
            SNR = DeltaPt * Lambda_mimo_2x2[c, r, :] / (R_mimo_2x2 * psd)
            soma_cap += np.sum(np.log2(1 + SNR)) / N
        matrizCap_mimo_bob[k, c] = soma_cap
        
### Eve
H_mimo_eve          = H_eve  # Nr=2, Nt=2
matrizCap_mimo_eve  = np.zeros((len(Pt), total_ch))

R_mimo_2x2 = 2
Lambda_mimo_2x2 = np.zeros((total_ch, R_mimo_2x2, N))
for c in range(total_ch):
    for fn in range(N):
        U, S, Vh = np.linalg.svd(H_mimo_eve[c, :, :, fn])
        Lambda_mimo_2x2[c, :, fn] = S**2

for c in range(total_ch):
    for k in tqdm(range(len(Pt))):
        DeltaPt = (Pt[k] * np.ones(N)) / N
        soma_cap = 0
        for r in range(R_mimo_2x2):
            SNR = DeltaPt * Lambda_mimo_2x2[c, r, :] / (R_mimo_2x2 * psd)
            soma_cap += np.sum(np.log2(1 + SNR)) / N
        matrizCap_mimo_eve[k, c] = soma_cap

########################
# 2.3) MISO (2x1)
########################

### Bob
# Nr=1, Nt=2 -> 1 porta Rx e 2 portas Tx
H_miso_bob          = H_bob[:, 0:1, 0:2, :]
matrizCap_miso_bob  = np.zeros((len(Pt), total_ch))

R_miso = 1
Lambda_miso = np.zeros((total_ch, R_miso, N))
for c in range(total_ch):
    for fn in range(N):
        U, S, Vh = np.linalg.svd(H_miso_bob[c, :, :, fn])
        Lambda_miso[c, 0, fn] = S[0]**2

for c in range(total_ch):
    for k in tqdm(range(len(Pt))):
        DeltaPt = (Pt[k] * np.ones(N)) / N
        SNR = DeltaPt * Lambda_miso[c, 0, :] / (R_miso * psd)
        matrizCap_miso_bob[k, c] = np.sum(np.log2(1 + SNR)) / N
        
#%%
######################################
# 3) Plot 
######################################
plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],   
    'mathtext.fontset': 'stix',          
    'font.size': 15,
    'axes.titlesize': 17,
    'axes.labelsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
})


vibrant_colors = ["#42ADB3", "#497D80", "#28DDE5", "#3B4C4C", "#2A3333"]
line_styles = ['-', '--', '-.']
markers = ['o', 's', '^']


lineStyleMap = {'MIMO': '-.', 'MISO': '--', 'SISO': '-'}
markerMap     = {'SE': 's', 'ME': 'o'}
colorMap      = {'SE': vibrant_colors[1], 'ME': vibrant_colors[4]}

######################################
# 3.1) Cenários
######################################
cenario = {
    'SISO_bob'  : matrizCap_siso_bob,
    'MISO_bob'  : matrizCap_miso_bob,
    'MIMO_bob'  : matrizCap_mimo_bob,
    'SISO_eve'  : matrizCap_siso_eve,
    'MIMO_eve'  : matrizCap_mimo_eve
    }

parCenario = [
    ('SISO_bob', 'SISO_eve', 'SISO-SE'),
    ('MISO_bob', 'SISO_eve', 'MISO-SE'),
    ('MISO_bob', 'MIMO_eve', 'MISO-ME'),
    ('MIMO_bob', 'SISO_eve', 'MIMO-SE'),
    ('MIMO_bob', 'MIMO_eve', 'MIMO-ME')
]

#%%
######################################
# 3) Probabilidade de Falha de Sigilo
######################################

########################
# 3.1) SOP x Pt
# Rs fixo em 1 b/s/Hz
########################
plt.figure(figsize=(8, 6))
ax = plt.gca()

Rs = 1 # bps/Hz

for (bob,eve,legenda) in parCenario:
    sysType, seType = legenda.split('-')[:2]  # ex.: 'MIMO-ME' -> 'MIMO', 'ME'
    ls  = lineStyleMap[sysType]
    mk  = markerMap[seType]
    cor = colorMap[seType]
    
    outage_pt = np.zeros(len(Pt))
    for idx in range(len(Pt)):
        Cb = cenario[bob][idx,:]
        Ce = cenario[eve][idx,:]
        Cs = Cb - Ce
        Cs[Cs < 0] = 0
        
        outage_pt[idx] = np.sum(Cs < Rs) / len(Cs)
    
    ax.plot(Pt_dBm, outage_pt, linestyle=ls, marker=mk, color=cor, label=legenda)

# Gera a figura
ax.set_xlabel(r'$P_t$ (dBm)')
ax.set_ylabel('Probabilidade de $Outage$ de Sigilo')
ax.grid(True)
ax.set_xlim(Pt_dBm.min(), Pt_dBm.max())
ax.set_xticks(Pt_dBm)
ax.set_ylim(0.3, 1, 0.1)

ax.set_yticks(np.arange(0.3, 1.1, 0.1))

lineHandles = [Line2D([0], [0], color='black', ls=ls, lw=2, label=key) for key, ls in lineStyleMap.items()]
markerHandles = [Line2D([0], [0], color=colorMap[key], marker=mk, lw=0, markersize=8, label=key)
                 for key, mk in markerMap.items()]
guideLegend = ax.legend(handles=lineHandles + markerHandles, loc='upper center',
                        bbox_to_anchor=(0.5, 1.17), ncol=5, frameon=False)
ax.add_artist(guideLegend)

plt.tight_layout()
plt.subplots_adjust(top=0.82)
plt.savefig("SOP_PT.pdf", dpi=300)
plt.show()
        
########################
# 3.1) SOP x Rs
# Pt_dBm fixo em 20 dBm
########################
plt.figure(figsize=(8, 6))
ax = plt.gca()

pt_simu = 20 #dBm
idx = np.where(Pt_dBm == pt_simu)[0][0]  # encontra o índice correspondente

RsMin = 0.01 ; passo = 0.25; RsMax = 20 + passo

for (bob, eve, legenda) in parCenario:
    # Coisa do gpt pra plotar bonito
    sysType, seType = legenda.split('-')[:2] # ex.: 'MIMO-ME' -> 'MIMO', 'ME'
    ls  = lineStyleMap[sysType]
    mk  = markerMap[seType]
    cor = colorMap[seType]
    
    # Outage
    Cb = cenario[bob][idx,:]
    Ce = cenario[eve][idx,:]
    Cs = Cb - Ce
    Cs[Cs < 0] = 0
    
    Rs = np.arange(RsMin, RsMax, passo)
    outage_cdf = np.array([np.sum(Cs < rs) / len(Cs) for rs in Rs])
    
    # Plota
    ax.plot(Rs, outage_cdf, linestyle=ls, marker=mk, color=cor, label=legenda, markevery = 4)

# Gera a figura 
ax.set_xlabel(r'$R_s$ (b/s/Hz)')
ax.set_ylabel('Probabilidade de $Outage$ de Sigilo')
ax.grid(True)
ax.set_xlim(0, RsMax)
ax.set_ylim(0.3, 1)
ax.set_yticks

ax.set_xticks(np.arange(0, RsMax , 1))
ax.set_yticks(np.arange(0.3, 1.1, 0.1))

lineHandles = [Line2D([0], [0], color='black', ls=ls, lw=2, label=key) for key, ls in lineStyleMap.items()]
markerHandles = [Line2D([0], [0], color=colorMap[key], marker=mk, lw=0, markersize=8, label=key)
                 for key, mk in markerMap.items()]
guideLegend = ax.legend(handles=lineHandles + markerHandles, loc='upper center',
                        bbox_to_anchor=(0.5, 1.17), ncol=5, frameon=False)
ax.add_artist(guideLegend)

plt.tight_layout()
plt.subplots_adjust(top=0.82)
plt.savefig("SOP_RS.pdf", dpi=300)
plt.show()


#%%
# =============================================================================
# 4) EST —  Caso 1 (RB=CB) e Caso 2 (sem CSI)
# =============================================================================

# Caso 1
est1    = np.zeros((len(parCenario),len(Pt_dBm)))
re1     = np.zeros((len(parCenario),len(Pt_dBm)))

# Caso 2
est2    = np.zeros((len(parCenario),len(Pt_dBm)))
rb2     = np.zeros((len(parCenario),len(Pt_dBm)))
re2     = np.zeros((len(parCenario),len(Pt_dBm)))

for idx, (bob,eve,legenda) in enumerate(parCenario):
    capBob = cenario[bob]
    capEve = cenario[eve]
    
    for k in range(len(Pt_dBm)):
        Cb = capBob[k, :]
        Ce = capEve[k, :]
        
        # Caso 1 (Rb = Cb)
        reMax       = max(Cb.max(),Ce.max()) + 0.5
        reValores   = np.arange(0, reMax + 0.02, 0.01)
        
        # Os = P(Ce > Re); 1 - Os = P(Ce <= Re)
        Os = np.array([np.sum(re < Ce) / len(Ce) for re in reValores])
        umMenosOs = 1.0 - Os
        
        # Para cada Cb, escolhe um Re* que maximiza (Cb - Re) * (1 - Os)
        est1_lista  = []
        re1_lista   = []
        
        for iCb in Cb:            
            verificado = reValores < iCb 
            if not np.any(verificado):
                est1_lista.append(0.0)
                re1_lista.append(0.0)
                continue
            
            ganhos = (iCb - reValores[verificado]) * umMenosOs[verificado]
            idxMax = np.argmax(ganhos)
            est1_lista.append(ganhos[idxMax])
            re1_lista.append(reValores[verificado][idxMax])
        
        est1[idx, k]    = np.mean(est1_lista)
        re1[idx, k]     = np.mean(re1_lista)
        
        # Caso 2 (Sem CSI)
        rbMax   = Cb.max() + 0.05
        reMax2  = max(rbMax,Cb.max()) + 0.05
        
        rbValores   = np.arange(0.0, rbMax + 0.02, 0.01)
        reValores2  = np.arange(0.0, reMax2 + 0.02, 0.01)
        
        # Or = P(Cb < Rb); 1 - Or = P(Cb >= Rb)
        Or = np.array([np.sum(rb > Cb) / len(Cb) for rb in rbValores])
        umMenosOr = 1.0 - Or
        
        # Os = P(Ce > Re); 1 - Os = P(Ce <= Re)
        Os = np.array([np.sum(re < Ce) / len(Ce) for re in reValores2])
        umMenosOs2 = 1.0 - Os
        
        # Busca o par (RB, RE) que maximiza (RB - RE) * (1 - Or) * (1 - Os)
        melhorValor = 0.0; melhorRb = 0.0; melhorRe = 0.0;
        
        for i, rb in enumerate(rbValores):
            reValidos = reValores2[reValores2 < rb]
            if reValidos.size == 0:
                continue
            
            # Calcula os valores da métrica
            ganhos = (rb - reValidos) * umMenosOr[i] * umMenosOs2[reValores2 < rb]
            idxMelhor = np.argmax(ganhos)
            valor = ganhos[idxMelhor]
            
            # Atualiza se for o melhor até agora
            if valor > melhorValor:
                melhorValor = valor
                melhorRb = rb
                melhorRe = reValidos[idxMelhor]
            
        est2[idx,k] = melhorValor
        rb2[idx,k]  = melhorRb
        re2[idx,k]  = melhorRe

# ---------- Figura A — EST1 (RB=CB) × Pt ----------
plt.figure(figsize=(8, 6))
ax = plt.gca()
for pIdx, (_, _, labelStr) in enumerate(parCenario):
    sysType, seType = labelStr.split('-')[:2]
    ls  = lineStyleMap[sysType]
    mk  = markerMap[seType]
    cor = colorMap[seType]
    ax.plot(Pt_dBm, est1[pIdx, :], linestyle=ls, marker=mk, color=cor, label=labelStr)
ax.set_xlabel(r'$P_T$ (dBm)')
ax.set_ylabel(r'$\bar{\Psi}_{1}^{*}$ (b/s/Hz)')
ax.grid(True)
ax.set_xlim(Pt_dBm.min(), Pt_dBm.max())
ax.set_xticks(Pt_dBm)
lineHandles = [Line2D([0], [0], color='black', ls=ls, lw=2, label=key) for key, ls in lineStyleMap.items()]
markerHandles = [Line2D([0], [0], color=colorMap[key], marker=mk, lw=0, markersize=8, label=key)
                 for key, mk in markerMap.items()]
guideLegend = ax.legend(handles=lineHandles + markerHandles, loc='upper center',
                        bbox_to_anchor=(0.5, 1.17), ncol=5, frameon=False)
ax.add_artist(guideLegend)
#ax.legend(loc='best')
plt.tight_layout()
plt.subplots_adjust(top=0.82)
#plt.yticks(np.arange(0,3.75,0.25))
plt.ylim(0,3.5,0.5)
plt.savefig('EST_1.pdf', dpi=300)
plt.show()

# ---------- Figura B — EST2 (sem CSI) × Pt ----------
plt.figure(figsize=(8, 6))
ax = plt.gca()
for pIdx, (_, _, labelStr) in enumerate(parCenario):
    sysType, seType = labelStr.split('-')[:2]
    ls  = lineStyleMap[sysType]
    mk  = markerMap[seType]
    cor = colorMap[seType]
    ax.plot(Pt_dBm, est2[pIdx, :], linestyle=ls, marker=mk, color=cor, label=labelStr)
ax.set_xlabel(r'$P_T$ (dBm)')
ax.set_ylabel(r'$\Psi_{2}^{*}$ (b/s/Hz)')
ax.grid(True)
ax.set_xlim(Pt_dBm.min(), Pt_dBm.max())
ax.set_xticks(Pt_dBm)
lineHandles = [Line2D([0], [0], color='black', ls=ls, lw=2, label=key) for key, ls in lineStyleMap.items()]
markerHandles = [Line2D([0], [0], color=colorMap[key], marker=mk, lw=0, markersize=8, label=key)
                 for key, mk in markerMap.items()]
guideLegend = ax.legend(handles=lineHandles + markerHandles, loc='upper center',
                        bbox_to_anchor=(0.5, 1.17), ncol=5, frameon=False)
ax.add_artist(guideLegend)
#ax.legend(loc='best')
plt.tight_layout()
plt.subplots_adjust(top=0.82)
plt.ylim(0,3.5,0.5)
plt.savefig('EST_2.pdf', dpi=300)
plt.show()

# ---------- Figura C — RB* × Pt ----------
plt.figure(figsize=(8, 6))
ax = plt.gca()
for pIdx, (_, _, labelStr) in enumerate(parCenario):
    sysType, seType = labelStr.split('-')[:2]
    ls  = lineStyleMap[sysType]
    mk  = markerMap[seType]
    cor = colorMap[seType]
    ax.plot(Pt_dBm, rb2[pIdx, :], linestyle=ls, marker=mk, color=cor, label=labelStr)
ax.set_xlabel(r'$P_T$ (dBm)')
ax.set_ylabel(r'$R_{B,2}^{*}$ (b/s/Hz)')
ax.grid(True)
ax.set_xlim(Pt_dBm.min(), Pt_dBm.max())
ax.set_xticks(Pt_dBm)
lineHandles = [Line2D([0], [0], color='black', ls=ls, lw=2, label=key) for key, ls in lineStyleMap.items()]
markerHandles = [Line2D([0], [0], color=colorMap[key], marker=mk, lw=0, markersize=8, label=key)
                 for key, mk in markerMap.items()]
guideLegend = ax.legend(handles=lineHandles + markerHandles, loc='upper center',
                        bbox_to_anchor=(0.5, 1.17), ncol=5, frameon=False)
ax.add_artist(guideLegend)
#ax.legend(loc='best')
plt.tight_layout()
plt.subplots_adjust(top=0.82)
plt.ylim(0, 14)
plt.savefig('RB_2.pdf', dpi=300)
plt.show()

# ---------- Figura D — RE* × Pt ----------
plt.figure(figsize=(8, 6))
ax = plt.gca()
for pIdx, (_, _, labelStr) in enumerate(parCenario):
    sysType, seType = labelStr.split('-')[:2]
    ls  = lineStyleMap[sysType]
    mk  = markerMap[seType]
    cor = colorMap[seType]
    ax.plot(Pt_dBm, re2[pIdx, :], linestyle=ls, marker=mk, color=cor, label=labelStr)
ax.set_xlabel(r'$P_T$ (dBm)')
ax.set_ylabel(r'$R_{E,2}^{*}$ (b/s/Hz)')
ax.grid(True)
ax.set_xlim(Pt_dBm.min(), Pt_dBm.max())
ax.set_xticks(Pt_dBm)
lineHandles = [Line2D([0], [0], color='black', ls=ls, lw=2, label=key) for key, ls in lineStyleMap.items()]
markerHandles = [Line2D([0], [0], color=colorMap[key], marker=mk, lw=0, markersize=8, label=key)
                 for key, mk in markerMap.items()]
guideLegend = ax.legend(handles=lineHandles + markerHandles, loc='upper center',
                        bbox_to_anchor=(0.5, 1.17), ncol=5, frameon=False)
ax.add_artist(guideLegend)
#ax.legend(loc='best')
plt.tight_layout()
plt.subplots_adjust(top=0.82)
plt.ylim(0, 14)
plt.savefig('RE_2.pdf', dpi=300)
plt.show()

# ---------- Figura E — RE ótimo MÉDIO (Caso 1) × Pt ----------
plt.figure(figsize=(8, 6))
ax = plt.gca()
for pIdx, (_, _, labelStr) in enumerate(parCenario):
    sysType, seType = labelStr.split('-')[:2]
    ls  = lineStyleMap[sysType]
    mk  = markerMap[seType]
    cor = colorMap[seType]
    ax.plot(Pt_dBm, re1[pIdx, :], linestyle=ls, marker=mk, color=cor, label=labelStr)
ax.set_xlabel(r'$P_T$ (dBm)')
ax.set_ylabel(r'$\bar{R}_{E,1}^{*}$ (b/s/Hz)')
ax.grid(True)
ax.set_xlim(Pt_dBm.min(), Pt_dBm.max())
ax.set_xticks(Pt_dBm)
lineHandles = [Line2D([0], [0], color='black', ls=ls, lw=2, label=key) for key, ls in lineStyleMap.items()]
markerHandles = [Line2D([0], [0], color=colorMap[key], marker=mk, lw=0, markersize=8, label=key)
                 for key, mk in markerMap.items()]
guideLegend = ax.legend(handles=lineHandles + markerHandles, loc='upper center',
                        bbox_to_anchor=(0.5, 1.17), ncol=5, frameon=False)
ax.add_artist(guideLegend)
#ax.legend(loc='best')
plt.tight_layout()
plt.subplots_adjust(top=0.82)
plt.ylim(0, 14)
plt.savefig('RE_1.pdf', dpi=300)
plt.show()

#%%
#%%
# =============================================================================
# 5) CDF da capacidade - SISO, MISO e MIMO (Bob)
#     Estilo aprimorado para publicação
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib as mpl

# ====== ESTILO VISUAL GLOBAL ======
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.5,
    "figure.dpi": 300
})

plt.figure(figsize=(7, 3))
ax = plt.gca()

# ====== Escolha do P_t ======
pt_simu = 20  # dBm
idx_pt = np.where(Pt_dBm == pt_simu)[0][0]

# ====== Capacidade de cada cenário ======
cenarios_cap = {
    'SISO': matrizCap_siso_bob[idx_pt, :],
    'MISO': matrizCap_miso_bob[idx_pt, :],
    'MIMO': matrizCap_mimo_bob[idx_pt, :]
}

# ====== Paleta e estilos personalizados ======
vibrant_colors = ["#42ADB3", "#497D80", "#28DDE5", "#3B4C4C", "#2A3333"]
line_styles = ['-', '--', '-.']
markers = ['o', 's', '^']

estilos = {
    'SISO': dict(color=vibrant_colors[1], linestyle=line_styles[0]),
    'MISO': dict(color=vibrant_colors[0], linestyle=line_styles[1]),
    'MIMO': dict(color=vibrant_colors[4], linestyle=line_styles[2])
}

# ====== Plot ======
for nome in ['SISO', 'MISO', 'MIMO']:
    dados = np.sort(cenarios_cap[nome])
    cdf = np.arange(1, len(dados) + 1) / len(dados)
    ax.plot(dados, 1-cdf, label=nome, **estilos[nome])

# ====== Eixos e grades ======
ax.set_xlim(0, 20)
ax.set_ylim(0, 1.0)
ax.set_xticks(np.arange(0, 22, 2))
ax.set_yticks(np.arange(0, 1.1, 0.1))

# ====== Ajustes visuais ======
ax.set_xlabel(r'Capacidade $C$ (b/s/Hz)', labelpad=6)
ax.set_ylabel(r'CDF complementar', labelpad=6)

ax.set_xlim(left=0)
ax.set_ylim(0, 1)

# grid leve + bordas discretas
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.6)
    spine.set_color("#444")

# legenda com moldura translúcida
leg = ax.legend(loc='best', frameon=True)
leg.get_frame().set_alpha(0.8)
leg.get_frame().set_edgecolor("#999")

# título opcional
# ax.set_title(r'$P_t = 20$ dBm', fontsize=12, pad=8)

plt.tight_layout(pad=1.5)
plt.savefig('CDF_capacidades_Pt20dBm_beautiful.pdf', dpi=600, bbox_inches='tight')
plt.show()


#%%
# ===========================================================
# CFR média + variância (sombreado)
# Porta 1 (Rx1) e Porta 2 (Rx2)
# Identidade visual do TCC
# ===========================================================

eps = 1e-12

# Config visual
plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 13,
    'axes.titlesize': 13,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
})

vibrant_colors = ["#42ADB3", "#497D80", "#28DDE5", "#3B4C4C", "#2A3333"]

# ===========================================================
# Função para extrair CFR da porta desejada
# ===========================================================
def get_cfr_porta(H_bob, porta_rx):
    # Norm L2 sobre portas Tx -> shape: [total_ch, N]
    return np.linalg.norm(H_bob[:, porta_rx, :, :], axis=1)

# Porta 1 e Porta 2
H_p1 = get_cfr_porta(H_bob, porta_rx=0)
H_p2 = get_cfr_porta(H_bob, porta_rx=1)

total_ch, N = H_p1.shape
freq_MHz = np.linspace(0, 100, N)

# ===========================================================
# Estatísticas
# ===========================================================
# Porta 1
mean_p1 = np.mean(H_p1, axis=0)
std_p1  = np.std(H_p1, axis=0)

mean_p1_db = 20*np.log10(mean_p1 + eps)
std_p1_db  = 20*np.log10(mean_p1 + std_p1 + eps) - mean_p1_db
upper_p1 = mean_p1_db + std_p1_db
lower_p1 = mean_p1_db - std_p1_db

# Porta 2
mean_p2 = np.mean(H_p2, axis=0)
std_p2  = np.std(H_p2, axis=0)

mean_p2_db = 20*np.log10(mean_p2 + eps)
std_p2_db  = 20*np.log10(mean_p2 + std_p2 + eps) - mean_p2_db
upper_p2 = mean_p2_db + std_p2_db
lower_p2 = mean_p2_db - std_p2_db

# ===========================================================
# FIGURA 2A — Porta 1
# ===========================================================
plt.figure(figsize=(8,3))
ax = plt.gca()

ax.plot(freq_MHz, mean_p1_db,
        color=vibrant_colors[0],
        linewidth=1.6,
        label='Média')

ax.fill_between(freq_MHz, lower_p1, upper_p1,
                color=vibrant_colors[0],
                alpha=0.25,
                label='Desvio padrão')

ax.set_xlabel('Frequência [MHz]')
ax.set_ylabel('Magnitude [dB]')
ax.set_xlim(0,100)
ax.set_yticks(range(0, -61, -20))        # ticks de -10 em -10

ax.grid(True, linestyle=':', linewidth=0.6)
ax.legend(loc='lower left', fontsize=10)

plt.tight_layout()
plt.savefig("CFR_media_variancia_porta1.pdf", dpi=300, bbox_inches='tight')
plt.show()

# ===========================================================
# FIGURA 2B — Porta 2
# ===========================================================
plt.figure(figsize=(8,3))
ax = plt.gca()

ax.plot(freq_MHz, mean_p2_db,
        color=vibrant_colors[2],
        linewidth=1.6,
        label='Média')

ax.fill_between(freq_MHz, lower_p2, upper_p2,
                color=vibrant_colors[2],
                alpha=0.25,
                label='Desvio padrão')

ax.set_xlabel('Frequência [MHz]')
ax.set_ylabel('Magnitude [dB]')
ax.set_xlim(0,100)
ax.set_yticks(range(0, -61, -20))        # ticks de -10 em -10

ax.grid(True, linestyle=':', linewidth=0.6)
ax.legend(loc='lower left', fontsize=10)

plt.tight_layout()
plt.savefig("CFR_media_variancia_porta2.pdf", dpi=300, bbox_inches='tight')
plt.show()

#%%
# =============================================================================
# PSD do ruído PLC - Ajuste fino dos eixos
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.io import loadmat

# ====== Carrega PSD ======
psd_med = loadmat('PLC_PSD.mat')
psd     = psd_med['psd_plc'][:, 0]   # vetor PSD (em W/Hz ou dBm/Hz)
N = len(psd)

# ====== Frequência (0 a 100 MHz) ======
freq_MHz = np.linspace(0, 100, N)

# ====== Converter para dBm/Hz se necessário ======
psd_dbm_hz = 10 * np.log10(psd + 1e-20) + 30

# ====== Estilo global ======
vibrant_colors = ["#42ADB3", "#497D80", "#28DDE5", "#3B4C4C", "#2A3333"]
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.5,
    "figure.dpi": 400,
    "axes.spines.top": False,
    "axes.spines.right": False
})

# ====== Plot ======
plt.figure(figsize=(8.5, 3.6))
ax = plt.gca()

ax.plot(freq_MHz, psd_dbm_hz,
        color=vibrant_colors[3],
        linewidth=1.3,
        label='PSD do ruído PLC')

# ====== Ajuste dos eixos ======
# ====== Eixos ajustados ======
ax.set_xlim(0, 100)
ax.set_xticks(np.arange(0, 110, 10))
ax.set_ylim(-70, 0)  # <-- eixo Y fixo
ax.set_yticks(np.arange(-70, 10, 10))

# ====== Layout e legendas ======
ax.set_xlabel('Frequência [MHz]')
ax.set_ylabel('PSD [dBm/Hz]')
ax.legend(loc='best', frameon=True)
ax.grid(True, which='both', linestyle=':', linewidth=0.6, alpha=0.7)

for spine in ax.spines.values():
    spine.set_linewidth(0.6)
    spine.set_color("#555")

plt.tight_layout(pad=1.2)
plt.savefig('PSD_ruido_PLC_eixosAjustados.pdf', dpi=600, bbox_inches='tight')
plt.show()



