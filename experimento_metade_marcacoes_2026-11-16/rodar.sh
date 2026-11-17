#!/bin/zsh
# Treino com parte dos quadros marcados: 3 grupos x 3 sementes x 2 faces, um treino por vez.
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/caso_polietileno_2026-08-22"
PY=${TG_PY:-python3}
export PYTHONPATH=codigo PYTHONWARNINGS=ignore
for g in A B Q; do
  for s in 0 1 2; do
    for f in baixo cima; do
      $PY codigo/metade_marcacoes.py treinar --grupo $g --face $f --seed $s || echo "FALHOU $g $f $s"
    done
  done
done
echo FIM
