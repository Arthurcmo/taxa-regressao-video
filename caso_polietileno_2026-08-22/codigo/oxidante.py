"""
Vazão de oxidante do ensaio, lida do registro do banco de testes.

Lê uma planilha Excel do sistema de aquisição do banco e grava um CSV pequeno com
t_s (ignição em 0) e mdot_ox_g_s, que o resto do código usa. Da planilha, interessam a
coluna de tempo relativo (segundos, com o zero na ignição) e a de vazão. Números
gravados como texto com vírgula decimal ("0,1") ou com anotação grudada são convertidos
aqui. Ler o .xls exige xlrd; ler o CSV, não.

    python codigo/oxidante.py --xls dados_oxidante/registro_banco.xls --csv dados_oxidante/vazao_oxidante.csv
"""
from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
XLS_PADRAO = RAIZ / "dados_oxidante" / "registro_banco.xls"
CSV_PADRAO = RAIZ / "dados_oxidante" / "vazao_oxidante.csv"

_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


def _num(v) -> float:
    """Numero de uma celula que pode ser float, texto com virgula, ou texto
    com anotacao grudada ('aqui por favor0,1'). Devolve NaN se nao houver."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = _NUM.findall(v.strip())
        if m:
            return float(m[-1].replace(",", "."))
    return float("nan")


def ler_xls(caminho: Path = XLS_PADRAO):
    """Le a planilha e devolve (t_s, mdot_g_s) alinhados, ignicao em t=0."""
    import xlrd  # so aqui, para o CSV poder ser lido sem a dependencia

    livro = xlrd.open_workbook(str(caminho))
    aba = livro.sheet_by_index(0)
    cab = [str(aba.cell_value(0, c)).strip() for c in range(aba.ncols)]

    def coluna(*nomes):
        for i, nome in enumerate(cab):
            baixo = nome.lower()
            if any(n in baixo for n in nomes):
                return i
        raise KeyError(f"coluna {nomes} nao encontrada em {cab}")

    ct = coluna("tempo relativo", "tempo")
    cv = coluna("vaz")
    t = np.array([_num(aba.cell_value(r, ct)) for r in range(1, aba.nrows)])
    v = np.array([_num(aba.cell_value(r, cv)) for r in range(1, aba.nrows)])
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    ordem = np.argsort(t, kind="stable")
    return t[ordem], v[ordem]


def escrever_csv(t, v, caminho: Path = CSV_PADRAO):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "mdot_ox_g_s"])
        for tt, vv in zip(t, v):
            w.writerow([f"{tt:.3f}", f"{vv:.4f}"])


def carregar(caminho_csv: Path = CSV_PADRAO, caminho_xls: Path = XLS_PADRAO):
    """Fonte unica de vazao para o resto do codigo. Usa o CSV se existir
    (caminho da nuvem, sem xlrd/sem video); senao le o .xls e o gera."""
    if Path(caminho_csv).exists():
        t, v = [], []
        with open(caminho_csv) as f:
            for row in csv.DictReader(f):
                t.append(float(row["t_s"]))
                v.append(float(row["mdot_ox_g_s"]))
        return np.array(t), np.array(v)
    t, v = ler_xls(caminho_xls)
    escrever_csv(t, v, caminho_csv)
    return t, v


def resumo(t, v) -> dict:
    dt = float(np.median(np.diff(t))) if len(t) > 1 else float("nan")
    aceso = v > 0.05
    janela = ([float(t[aceso][0]), float(t[aceso][-1])] if aceso.any()
              else [float("nan"), float("nan")])
    durante = (t >= 0) & aceso
    return dict(
        n=int(len(t)),
        t0_s=float(t.min()), t1_s=float(t.max()), dt_s=dt,
        vazao_min_g_s=float(v.min()), vazao_max_g_s=float(v.max()),
        vazao_media_durante_queima_g_s=float(v[durante].mean()) if durante.any() else float("nan"),
        janela_vazao_s=janela,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xls", default=str(XLS_PADRAO))
    ap.add_argument("--csv", default=str(CSV_PADRAO))
    a = ap.parse_args()
    t, v = ler_xls(Path(a.xls))
    escrever_csv(t, v, Path(a.csv))
    r = resumo(t, v)
    print(f"lido {a.xls}")
    print(f"gerado {a.csv}  ({r['n']} amostras, dt={r['dt_s']:.3f}s)")
    print(f"t de {r['t0_s']:.1f} a {r['t1_s']:.1f} s (ignicao em 0)")
    print(f"vazao de oxidante: {r['vazao_min_g_s']:.2f} a "
          f"{r['vazao_max_g_s']:.2f} g/s; "
          f"media na queima {r['vazao_media_durante_queima_g_s']:.2f} g/s")
    print(f"vazao > 0,05 g/s de {r['janela_vazao_s'][0]:.1f} a "
          f"{r['janela_vazao_s'][1]:.1f} s")


if __name__ == "__main__":
    main()
