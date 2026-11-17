#!/bin/zsh
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/caso_polietileno_2026-08-22"
for s in 0 1 2; do
  echo "=== semente $s $(date +%H:%M)"
  PYTHONPATH=codigo ${TG_PY:-python3} -W ignore codigo/treino_v2.py v4 --seed $s || { echo "ERRO semente $s"; exit 1; }
done
echo "fim $(date +%H:%M)"
