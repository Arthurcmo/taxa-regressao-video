#!/bin/bash
# Clique duas vezes neste arquivo para abrir o marcador.
# Ele abre uma janela para escolher a sessão (lista de quadros) e o perfil.
AQUI="$(cd "$(dirname "$0")" && pwd)"
CASO="$AQUI/../caso_polietileno_2026-08-22"
cd "$CASO" || { echo "não achei a pasta do caso: $CASO"; read -r; exit 1; }
PY="${TG_PY:-$( [ -x "$CASO/.venv/bin/python" ] && echo "$CASO/.venv/bin/python" || ( [ -x "$AQUI/../.venv/bin/python" ] && echo "$AQUI/../.venv/bin/python" || echo python3 ) )}"
"$PY" "$AQUI/abrir_marcador.py"
echo
echo "Pode deixar esta janela do Terminal aberta enquanto marca (ela mostra avisos do programa)."
