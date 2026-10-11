"""Pipeline: integra iGovTIC-JUD (2021-2026) e Pesquisa IA do CNJ (2022-2024),
constroi escores COBIT de governanca digital por orgao-ano e roda as analises QP1-QP4
e o modelo preditivo exploratorio. Uso: python -I analise/pipeline.py <raiz_do_projeto>
Restricao de escopo: apenas perguntas da Area de TIC (1-28); nenhum item do Diagnostico Complementar.
"""
import re, sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import cohen_kappa_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
IG = ROOT / "Base Dados" / "iGovTIC-JUD"
IA = ROOT / "Base Dados" / "Painel de Projetos de IA no Poder Judiciário"
OUT = ROOT / "analise" / "output"; OUT.mkdir(parents=True, exist_ok=True)
RNG = 42

def norm(s):
    s = re.sub(r"[\s\-\.]", "", str(s).strip().upper())
    return {"TJDF": "TJDFT"}.get(s, s)

# ---------------- 1. iGovTIC-JUD ----------------
SHEETS = {2021: "igovtic2021", 2022: "relatorio", 2023: "relatorio", 2024: "Base"}
# Peso maximo da pergunta no ano (Manuais; confere com o maximo observado)
W = {2021: {3: 30, 4: 15, 6: 25, 8: 20, 13: 10, 16: 40, 17: 60, 23: 40, 24: 15, 25: 15},
     2022: {3: 30, 4: 15, 6: 25, 8: 20, 13: 10, 16: 40, 17: 60, 23: 40, 24: 15, 25: 15},
     2023: {3: 38, 4: 22, 6: 10, 8: 25, 13: 10, 16: 50, 17: 50, 23: 50, 24: 15, 25: 10},
     2024: {3: 38, 4: 22, 6: 10, 8: 25, 13: 10, 16: 50, 17: 50, 23: 50, 24: 15, 25: 10}}

cad = pd.read_excel(IG / "iGovTIC-JUD 2026 - Resultado Geral - Limpo.xlsx", sheet_name="base")
cad["sig"] = cad["SIGLA"].map(norm); cad["UF"] = cad["UF"].astype(str).str.strip()
cad = cad.rename(columns={"Segmento ": "segmento"})[["sig", "UF", "segmento", "Porte"] + [f"iGOVTIC_{y}" for y in range(2021, 2025)]]

rows = []
for y, sh in SHEETS.items():
    df = pd.read_excel(IG / f"iGovTIC-JUD {y} - Resultado Geral - Limpo.xlsx", sheet_name=sh)
    df["sig"] = df["SIGLA"].map(norm)
    q = lambda n: pd.to_numeric(df[f"Pergunta {n:02d}"], errors="coerce").fillna(0) / W[y][n]
    b = lambda k: (pd.to_numeric(df[f"s1p03r01{k}"], errors="coerce").fillna(0) > 0).astype(float)
    r = pd.DataFrame({"sig": df["sig"], "ano": y,
        "EDM01": q(4),
        "APO02": pd.concat([q(6), q(8), b("f")], axis=1).mean(axis=1),
        "APO07": pd.concat([q(16), q(17), b("c")], axis=1).mean(axis=1),
        "APO12": q(24),
        "APO12_plano": b("e"),
        "APO13": pd.concat([q(23), q(25)], axis=1).mean(axis=1),
        "P13_1": (pd.to_numeric(df["s1p13r01a"], errors="coerce").fillna(0) > 0).astype(int)})
    rows.append(r)
gov = pd.concat(rows, ignore_index=True)
idx_long = cad.melt(id_vars=["sig"], value_vars=[f"iGOVTIC_{y}" for y in range(2021, 2025)],
                    var_name="ano", value_name="indice")
idx_long["ano"] = idx_long["ano"].str[-4:].astype(int)
gov = gov.merge(idx_long, on=["sig", "ano"], how="left").merge(cad[["sig", "UF", "segmento", "Porte"]], on="sig", how="left")
COBIT = ["EDM01", "APO02", "APO07", "APO12", "APO13"]
gov["indice"] = pd.to_numeric(gov["indice"], errors="coerce")

# ---------------- 2. Pesquisa IA ----------------
c22 = pd.read_csv(IA / "2022-Projetos-IA.csv", sep=None, engine="python")
c22["sig"] = c22["Tribunal"].map(norm)
ia22 = c22.groupby("sig").agg(tem=("Tem projeto de IA?", lambda s: int((s == "Sim").any())),
                              n=("Projeto", lambda s: s.notna().sum())).reset_index()
ia22.loc[ia22.tem == 0, "n"] = 0
r23 = pd.read_excel(IA / "2023-Respostas-Pesquisa-IA.xlsx")
r23["sig"] = r23["Tribunal"].map(norm)
cp = [c for c in r23.columns if "possui Projeto" in c][0]
ce = [c for c in r23.columns if "estágio" in c.lower()][0]
ORD23 = {"Não iniciado": 1, "Estágio Inicial": 1, "Em andamento": 2,
         "Finalizado mas ainda não implementado pelo Tribunal/ Conselho": 3, "Finalizado e em produção": 4}
r23["est"] = r23[ce].map(ORD23)
ia23 = r23.groupby("sig").agg(tem=(cp, lambda s: np.nan if (s == "Não sei responder").all() else int((s == "Sim").any())),
                              n=("est", lambda s: s.notna().sum()), est_max=("est", "max")).reset_index()
p24 = pd.read_excel(IA / "2024-Projetos-IA.xlsx")
p24["sig"] = p24["Tribunal"].map(norm)
ORD24 = {"Concepção (ideação e planejamento)": 1, "Em desenvolvimento": 2, "Em teste": 2,
         "Finalizado, aguardando autorização para entrar em produção / uso": 3,
         "Finalizado, mas não está em produção": 3, "Finalizado e em produção / uso": 4}
p24["est"] = p24["Estágio de Desenvolvimento do Projeto"].map(ORD24)
ia24 = p24.groupby("sig").agg(n=("est", "size"), est_max=("est", "max"),
                              n_prod=("est", lambda s: int((s == 4).sum()))).reset_index()
ia24["tem"] = 1
ia = pd.concat([ia22.assign(ano=2022), ia23.assign(ano=2023), ia24.assign(ano=2024)], ignore_index=True)

# Linkage report
link = {}
for y, d in ((2022, ia22), (2023, ia23), (2024, ia24)):
    s = set(d.sig); g = set(gov.loc[gov.ano == y, "sig"])
    link[y] = {"ia_orgaos": len(s), "casados": len(s & g), "nao_casados": sorted(s - g)}
print("LINKAGE", link)

# ---------------- 3. Painel ----------------
P = gov.merge(ia, on=["sig", "ano"], how="left")
P.loc[P.ano == 2024, "tem"] = P.loc[P.ano == 2024, "tem"].fillna(np.nan)  # ausencia 2024 != sem IA
lag = gov[["sig", "ano"] + COBIT + ["indice"]].copy(); lag["ano"] += 1
P = P.merge(lag, on=["sig", "ano"], how="left", suffixes=("", "_lag"))
for c in [c for c in P.columns if c.endswith("_lag")] + ["tem", "n", "est_max"]:
    P[c] = pd.to_numeric(P[c], errors="coerce")
P["seg_g"] = P["segmento"].replace({"Superior": "Superiores/Conselhos/Militar", "Conselhos": "Superiores/Conselhos/Militar",
                                    "Militar": "Superiores/Conselhos/Militar"})
P.to_csv(OUT / "painel_orgao_ano.csv", index=False, encoding="utf-8")

res = {"linkage": link}
# ---------------- QP1 ----------------
q1 = (P[P.ano.isin([2022, 2023])].dropna(subset=["tem"]).groupby(["ano", "segmento"])["tem"]
      .agg(["mean", "size"]).reset_index())
q1.to_csv(OUT / "qp1_adocao_segmento.csv", index=False)
res["qp1_total"] = P[P.ano.isin([2022, 2023])].dropna(subset=["tem"]).groupby("ano")["tem"].agg(["mean", "sum", "size"]).round(3).to_dict()
res["qp1_2024"] = {"orgaos_com_projeto": int(ia24.shape[0]), "projetos": int(p24.shape[0]),
                   "projetos_em_producao": int((p24.est == 4).sum()),
                   "mediana_proj_por_orgao": float(ia24.n.median())}
uf = P[P.ano == 2023].dropna(subset=["tem"]).groupby("UF")["tem"].mean().sort_values()
uf.to_csv(OUT / "qp1_adocao_uf_2023.csv")
res["mt"] = P[(P.UF == "MT")][["sig", "ano", "tem", "n", "indice"] + COBIT].round(3).to_dict("records")

# ---------------- QP2 / QP3 ----------------
tab = []
for y in (2022, 2023):
    d = P[(P.ano == y)].dropna(subset=["tem"])
    for v in ["indice"] + COBIT:
        for suf in ("", "_lag"):
            col = v + suf
            if col not in d: continue
            dd = d.dropna(subset=[col])
            x1, x0 = dd.loc[dd.tem == 1, col], dd.loc[dd.tem == 0, col]
            u = stats.mannwhitneyu(x1, x0, alternative="two-sided")
            r_rb = 2 * u.statistic / (len(x1) * len(x0)) - 1  # correlacao rank-biserial
            tab.append({"ano": y, "variavel": v, "medida": "t-1" if suf else "t", "n1": len(x1), "n0": len(x0),
                        "med_adot": round(x1.median(), 3), "med_nao": round(x0.median(), 3),
                        "p_MW": round(u.pvalue, 4), "r_rb": round(r_rb, 3)})
qp2 = pd.DataFrame(tab); qp2.to_csv(OUT / "qp2_mannwhitney.csv", index=False)

# Logistica por construto, controle de segmento, dados empilhados 2022+2023 com efeito de ano
L = P[P.ano.isin([2022, 2023])].dropna(subset=["tem"]).copy()
L["tem"] = L["tem"].astype(int)
logit = []
for v in ["indice"] + COBIT:
    for suf in ("", "_lag"):
        col = v + suf; dd = L.dropna(subset=[col]).copy()
        dd["z"] = (dd[col] - dd[col].mean()) / dd[col].std()
        try:
            try:
                m = smf.logit("tem ~ z + C(seg_g) + C(ano)", data=dd).fit(disp=0, cov_type="cluster",
                                                                         cov_kwds={"groups": pd.factorize(dd.sig)[0]})
            except Exception:
                m = smf.logit("tem ~ z + C(seg_g) + C(ano)", data=dd).fit(disp=0, method="bfgs", maxiter=2000,
                                                                         cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.sig)[0]})
            ci = m.conf_int().loc["z"]
            logit.append({"variavel": v, "medida": "t-1" if suf else "t", "n": int(m.nobs),
                          "OR_por_DP": round(np.exp(m.params["z"]), 3), "IC95_inf": round(np.exp(ci[0]), 3),
                          "IC95_sup": round(np.exp(ci[1]), 3), "p": round(m.pvalues["z"], 4)})
        except Exception as e:
            logit.append({"variavel": v, "medida": "t-1" if suf else "t", "erro": str(e)[:80]})
qp3 = pd.DataFrame(logit); qp3.to_csv(OUT / "qp3_logit.csv", index=False)

# Intensidade 2024 entre adotantes (Spearman)
A = P[(P.ano == 2024) & P.n.notna()]
res["qp2_intensidade_2024"] = {v: dict(zip(("rho", "p"), np.round(stats.spearmanr(A[v], A["n"], nan_policy="omit"), 4)))
                               for v in ["indice"] + COBIT}

# ---------------- QP4: P13.1 declarado x Pesquisa IA ----------------
q4 = []
for y in (2022, 2023):
    d = P[(P.ano == y)].dropna(subset=["tem"])
    ct = pd.crosstab(d.P13_1, d.tem)
    q4.append({"ano": y, "n": len(d), "kappa": round(cohen_kappa_score(d.P13_1, d.tem.astype(int)), 3),
               "concordancia": round((d.P13_1 == d.tem).mean(), 3), "tabela": ct.to_dict()})
q4.append({"ano": 2024, "P13_1_sim_pct": round(gov.loc[gov.ano == 2024, "P13_1"].mean(), 3)})
q4.append({"ano": 2026, "P13_1_sim": "94/94 (100%)"})
res["qp4"] = q4

# ---------------- Direcao reversa ----------------
c26 = pd.read_excel(IG / "iGovTIC-JUD 2026 - Resultado Geral - Limpo.xlsx", sheet_name="base")
c26["sig"] = c26["SIGLA"].map(norm)
i26 = [c for c in c26.columns if str(c).strip() in ("iGovTIC-Jud 2026", "iGOVTIC_2026")]
if i26:
    R = P[P.ano == 2023][["sig", "tem", "indice"]].dropna().merge(c26[["sig", i26[0]]], on="sig")
    R["delta"] = pd.to_numeric(R[i26[0]], errors="coerce") - R["indice"]
    u = stats.mannwhitneyu(R.loc[R.tem == 1, "delta"], R.loc[R.tem == 0, "delta"])
    res["reversa_delta_2023_2026"] = {"col": str(i26[0]), "med_adot": round(R.loc[R.tem == 1, "delta"].median(), 2),
                                      "med_nao": round(R.loc[R.tem == 0, "delta"].median(), 2), "p_MW": round(u.pvalue, 4)}

# ---------------- Modelo preditivo exploratorio ----------------
M = P[P.ano == 2023].dropna(subset=["tem"] + COBIT).copy(); yv = M.tem.astype(int).values
X = M[COBIT + ["seg_g"]]
pre = ColumnTransformer([("num", StandardScaler(), COBIT), ("cat", OneHotEncoder(handle_unknown="ignore"), ["seg_g"])])
pre_seg = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["seg_g"])])
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=RNG)
models = {"Maioria (baseline)": make_pipeline(pre, DummyClassifier(strategy="most_frequent")),
          "Logística só segmento": make_pipeline(pre_seg, LogisticRegression(max_iter=2000)),
          "Logística L2 (COBIT+segmento)": make_pipeline(pre, LogisticRegression(C=1.0, max_iter=2000)),
          "Árvore de decisão (prof. 3)": make_pipeline(pre, DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=RNG))}
pm = []
for k, m in models.items():
    s = cross_validate(m, X, yv, cv=cv, scoring=["roc_auc", "f1", "balanced_accuracy"])
    pm.append({"modelo": k, "AUC": round(s["test_roc_auc"].mean(), 3), "AUC_dp": round(s["test_roc_auc"].std(), 3),
               "F1": round(s["test_f1"].mean(), 3), "Acc_bal": round(s["test_balanced_accuracy"].mean(), 3)})
pm = pd.DataFrame(pm); pm.to_csv(OUT / "modelo_preditivo.csv", index=False)
lr = models["Logística L2 (COBIT+segmento)"].fit(X, yv)
coef = dict(zip(COBIT, np.round(lr[-1].coef_[0][:len(COBIT)], 3)))
tree = models["Árvore de decisão (prof. 3)"].fit(X, yv)
imp = dict(zip(list(pre.fit(X).get_feature_names_out()), np.round(tree[-1].feature_importances_, 3)))
res["modelo"] = {"n": int(len(M)), "prevalencia": round(yv.mean(), 3), "coef_logistica": coef,
                 "importancia_arvore": {k: v for k, v in imp.items() if v > 0}}

# ---------------- Figuras ----------------
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(6, 2.8))
pv = q1.pivot(index="segmento", columns="ano", values="mean").sort_values(2023)
pv.plot.barh(ax=ax, color=["#bbbbbb", "#444444"], width=0.75)
ax.set_xlabel("Proporção de órgãos com projeto de IA"); ax.set_ylabel(""); ax.set_xlim(0, 1)
ax.legend(title="Ano", frameon=False); fig.tight_layout(); fig.savefig(OUT / "fig_adocao_segmento.pdf"); fig.savefig(OUT / "fig_adocao_segmento.png", dpi=200)

fig, ax = plt.subplots(figsize=(6, 2.6))
u2 = P[P.ano == 2023].dropna(subset=["tem"]).groupby("UF").agg(ad=("tem", "mean"), g=("indice", "mean")).sort_values("ad")
ax.bar(u2.index, u2.ad, color=["#c0392b" if i == "MT" else "#777777" for i in u2.index])
ax.set_ylabel("Adoção de IA (2023)"); ax.tick_params(axis="x", rotation=90, labelsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig_adocao_uf.pdf"); fig.savefig(OUT / "fig_adocao_uf.png", dpi=200)

f = qp3.dropna(subset=["OR_por_DP"])
fig, ax = plt.subplots(figsize=(5.5, 3))
for i, (lab, g) in enumerate(f.groupby("medida")):
    yy = np.arange(len(g)) + (0.15 if lab == "t-1" else -0.15)
    ax.errorbar(g.OR_por_DP, yy, xerr=[g.OR_por_DP - g.IC95_inf, g.IC95_sup - g.OR_por_DP], fmt="o" if lab == "t" else "s",
                color="#000" if lab == "t" else "#888", label=f"governança em {lab}", capsize=2)
ax.set_yticks(np.arange(len(f[f.medida == "t"]))); ax.set_yticklabels(f[f.medida == "t"].variavel.replace({"indice": "Índice iGovTIC"}))
ax.axvline(1, ls="--", c="#999"); ax.set_xscale("log"); ax.set_xlabel("Odds ratio por 1 DP (IC 95%)")
ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(OUT / "fig_forest_or.pdf"); fig.savefig(OUT / "fig_forest_or.png", dpi=200)

json.dump(res, open(OUT / "resultados.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(qp2.to_string()); print(qp3.to_string()); print(pm.to_string()); print(q1.to_string())
