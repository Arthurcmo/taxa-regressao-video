"""Termos padronizados aplicados a todo texto de figura na hora de salvar.

Voz impessoal ("marcações manuais", nunca "do autor") e nada de "sessão" de pedidos. As
chaves dos resultados gravados continuam com os nomes antigos; só o texto desenhado muda.

Basta importar este módulo (figuras_tg2.py já o importa): ele envolve Figure.savefig.
"""
from __future__ import annotations

import matplotlib.figure as _mf
import matplotlib.text as _mt

TROCAS = (
    ("Marcações do autor", "Marcações manuais"),
    ("marcações do autor", "marcações manuais"),
    ("marcação do autor", "marcação manual"),
    ("pontos do autor", "pontos marcados"),
    ("face superior do autor", "face superior marcada"),
    ("suas marcações", "marcações manuais"),
    ("ImageNet e sessão", "ImageNet e marcações espaçadas"),
    ("ajuste fino e sessão", "ajuste fino e marcações espaçadas"),
    ("+ sessão", "+ marcações espaçadas"),
    ("sessão de marcação", "marcações espaçadas"),
    ("camada de pirólise", "faixa escura junto ao sólido"),
)

trocados: list[tuple[str, str]] = []


def aplicar_em(fig) -> None:
    for t in fig.findobj(_mt.Text):
        s = t.get_text()
        if not s:
            continue
        n = s
        for a, b in TROCAS:
            n = n.replace(a, b)
        if n != s:
            trocados.append((s, n))
            t.set_text(n)


if not getattr(_mf.Figure.savefig, "_textos_tg", False):
    _orig = _mf.Figure.savefig

    def _savefig(self, *args, **kwargs):
        aplicar_em(self)
        return _orig(self, *args, **kwargs)

    _savefig._textos_tg = True
    _mf.Figure.savefig = _savefig
