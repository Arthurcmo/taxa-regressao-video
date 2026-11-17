"""Valores do ensaio analisado, lidos de configuracao_pe.json (seção "ensaio").

Todo valor que depende do ensaio (vídeo, escala, geometria, base de tempo, janelas e
registro de vazão) vem daqui. Para analisar outro ensaio, copie
configuracao_exemplo.json para configuracao_pe.json, na pasta do caso, e preencha a
seção "ensaio" com os dados do seu ensaio. O README explica cada campo.
"""
from __future__ import annotations

import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
_ARQ = RAIZ / "configuracao_pe.json"
if not _ARQ.exists():
    raise FileNotFoundError(
        f"Falta {_ARQ}. Copie configuracao_exemplo.json para configuracao_pe.json e preencha "
        "a seção 'ensaio' com os dados do seu ensaio (veja o README).")
_E = json.loads(_ARQ.read_text(encoding="utf-8"))["ensaio"]

VIDEO = RAIZ / _E["video"]                              # vídeo do ensaio
FPS_CAMERA = float(_E["fps_camera"])                    # quadros por segundo da câmera
QUADROS_POR_AMOSTRA = int(_E["quadros_por_amostra"])    # quadros únicos por amostra do cubo
CAM = QUADROS_POR_AMOSTRA / FPS_CAMERA                  # s de câmera por amostra
ROI = [int(v) for v in _E["roi"]]                       # recorte [y0, y1, x0, x1] em px
MM_POR_PX = float(_E["mm_por_px"])                      # escala da imagem
LARGURA_PORTA_MM = float(_E["largura_porta_mm"])        # largura da porta, para dentro da imagem
FATOR_TEMPO_REAL = float(_E["fator_tempo_real"])        # tempo real / tempo marcado pela câmera
AMOSTRA_IGNICAO = int(_E["amostra_ignicao"])            # amostra da ignição (contagem a partir de 0)
AMOSTRA_INICIO_MEDIDA = int(_E["amostra_inicio_medida"])  # primeira amostra em que a superfície só recua
AMOSTRA_FIM_CHAMA = int(_E["amostra_fim_chama"])        # amostra em que a chama apaga
JANELA_ANALISE_S = tuple(float(v) for v in _E["janela_analise_s"])  # s desde a ignição
BLOCOS_TESTE = {k: tuple(int(x) for x in v) for k, v in _E["blocos_teste"].items()}
MARCACOES = RAIZ / _E["marcacoes"]                      # guia.csv com as marcações manuais

_R = _E["registro_vazao"]
VAZAO_CSV = RAIZ / _R["vazao_csv"]                      # colunas: t (s do registro), vazão
CANAIS_CSV = RAIZ / _R["canais_csv"]                    # canais do banco, com cabeçalho
T_IGNICAO_REGISTRO = float(_R["t_ignicao_s"])           # instante da ignição no relógio do registro
DENSIDADE_OXIDANTE = float(_R["densidade_oxidante_g_ml"])  # converte a vazão de mL/s para g/s
