"""
chamacera — detecção, rotulagem e automação para vídeos de slab burner
(chama + grão de parafina) em propulsão híbrida.

Classes semânticas (fixas em todo o pacote):
    0 = fundo   (câmara, janela, fora do domínio)
    1 = chama   (zona auto-luminosa)
    2 = cera    (combustível sólido/líquido remanescente)
"""

__version__ = "0.1.0"

FUNDO, CHAMA, CERA = 0, 1, 2
CLASSES = {FUNDO: "fundo", CHAMA: "chama", CERA: "cera"}
N_CLASSES = 3

# paleta BGR para visualização (OpenCV)
PALETTE_BGR = ((0, 0, 0), (62, 168, 255), (170, 188, 196))
