# Taxa de regressão de combustível híbrido medida em vídeo

Este código mede a taxa de regressão de um combustível sólido a partir do vídeo de uma queima em queimador bidimensional, isto é, em uma câmara com janela que mostra o grão de lado. Em cada quadro, ele localiza as duas faces do grão e acompanha o recuo delas no tempo. Daí saem a taxa de regressão, o fluxo de oxidante e a lei de regressão do ensaio.

A posição das faces sai por três caminhos:

1. um detector clássico de vale, com decodificação temporal por programação dinâmica;
2. redes em forma de U treinadas com marcações manuais (S1 e o ajuste fino 2b);
3. uma rede treinada sem nenhuma marcação manual, só com o detector clássico e regras físicas (SM1 v4).

O código foi escrito para analisar o vídeo de um ensaio de polietileno de alta densidade com peróxido de hidrogênio, e os parâmetros dos métodos foram ajustados nesse vídeo. **Este repositório não traz nenhum dado desse ensaio**: nem o vídeo, nem o registro do banco, nem as marcações, nem os resultados. Para usar o código, coloque os dados do seu ensaio nos lugares indicados abaixo.

## Onde colocar os dados do seu ensaio

Tudo fica dentro de `caso_polietileno_2026-08-22/`. Os caminhos abaixo são relativos a essa pasta.

| Dado | Onde colocar | Formato |
|---|---|---|
| Configuração | `configuracao_pe.json` | copie `configuracao_exemplo.json` e preencha a seção `ensaio` (tabela abaixo) |
| Vídeo da queima | `video/` | o arquivo indicado em `ensaio.video` |
| Vazão de oxidante | `dados_oxidante/vazao_oxidante.csv` | CSV com cabeçalho e duas colunas: tempo no relógio do registro (s) e vazão em mL/s. O `codigo/oxidante.py` gera esse CSV a partir de uma planilha Excel do banco |
| Canais do banco | `dados_oxidante/canais.csv` | CSV com cabeçalho e as colunas `t_planilha_s`, `vazao_mL_s`, `P_leito`, `T_leito2_C` e `T_entrada_camara_C` (relógio do registro, vazão, pressão e temperaturas do leito catalítico e da entrada da câmara) |
| Sessões de marcação | `sessao_<nome>/quadros.json` | `{"quadros": [{"k": 180, "face": "baixo", "uso": "treino"}, ...]}`, com `k` a amostra, `face` igual a `baixo` ou `cima` e `uso` igual a `treino` ou `teste_<bloco>` |
| Marcações manuais | `resultados/revisao_humana/perfis/<perfil>/guia.csv` | gravadas pelo marcador. Colunas `tipo,k,t_s,face,x_px,y_roi_px,t_video_s`; os pontos são as linhas com `tipo` igual a `ancora` |

### Campos de `configuracao_pe.json`, seção `ensaio`

| Campo | O que é |
|---|---|
| `video` | caminho do vídeo |
| `fps_camera` | quadros por segundo da câmera |
| `quadros_por_amostra` | quadros únicos da câmera por amostra do cubo |
| `roi` | recorte da imagem, `[y0, y1, x0, x1]` em px, com o grão inteiro e as duas faces |
| `mm_por_px` | escala da imagem, de preferência pelo comprimento conhecido do grão |
| `largura_porta_mm` | largura da porta para dentro da imagem, usada em G_ox = ṁ_ox / (L H) |
| `fator_tempo_real` | tempo real dividido pelo tempo marcado pela câmera; o `codigo/relogio_e_corte.py` mede esse fator por eventos de vazão que aparecem no vídeo e no registro |
| `amostra_ignicao` | amostra da ignição, contada a partir de 0 |
| `amostra_inicio_medida` | primeira amostra em que a superfície só recua |
| `amostra_fim_chama` | amostra em que a chama apaga |
| `janela_analise_s` | início e fim da janela de análise, em s desde a ignição; o `relogio_e_corte.py` sugere o início pelo sistema de oxidante |
| `blocos_teste` | blocos de amostras reservados para testar as redes |
| `marcacoes` | `guia.csv` das marcações manuais usadas no treino |
| `registro_vazao` | arquivos do registro, instante da ignição no relógio do registro e densidade do oxidante (g/mL) |

### O que ainda está ajustado ao vídeo original

Alguns valores do código foram escolhidos olhando o vídeo para o qual ele foi escrito e precisam ser revistos para outro ensaio:

- o número de amostras do vídeo (963) e de colunas de 20 px (93), fixos em vários scripts;
- as janelas de tempo em que o `relogio_e_corte.py` procura os eventos de vazão (`JANELA_VIDEO` e `LER_VIDEO`);
- a largura do vale, o passo máximo por quadro e as penalidades do detector e da decodificação (`detectar_interfaces.py`);
- os pesos das perdas, o número de passos e as faixas em volta das marcações no treino das redes;
- os quadros escolhidos para as figuras.

## Estrutura

```
README.md, requirements.txt
MARCADOR/                                programa de marcação manual (abre caso_polietileno_2026-08-22/codigo/interface.py)
caso_polietileno_2026-08-22/
    codigo/                              todo o processamento
    configuracao_exemplo.json            modelo da configuração do ensaio
chamacera_2026-08-16/codigo/chamacera/   pacote da rede: rede em forma de U, regressão de fronteira por coluna, conjunto de dados
experimento_recompensa_lei_2026-11-06/   pasta de saída das redes e das análises de taxa (os scripts a preenchem)
experimento_metade_marcacoes_2026-11-16/ pasta de saída do treino com parte dos quadros marcados
```

As pastas com data no nome ficaram assim de propósito. Os scripts acham uns aos outros e acham os dados e as saídas por esses nomes, e trocá-los quebraria o código. O pacote da rede ficou em `chamacera_2026-08-16` pelo mesmo motivo.

## Como instalar

Python 3.13:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

O treino foi feito num computador com processador Apple M2, pela interface Metal Performance Shaders do PyTorch.

## Como rodar

Os scripts rodam de dentro de `caso_polietileno_2026-08-22/`, com `PYTHONPATH=codigo`. O início de cada arquivo diz o que ele faz, o que lê e como chamá-lo. Com `BASE_TEMPO=real`, as taxas saem no tempo real (tempo marcado pela câmera vezes `fator_tempo_real`).

```bash
cd caso_polietileno_2026-08-22
export PYTHONPATH=codigo BASE_TEMPO=real
python codigo/oxidante.py --xls dados_oxidante/registro_banco.xls   # vazão do registro (se vier em Excel)
python codigo/ler_contador.py                    # contador gravado em cada quadro e quadros únicos
python codigo/extrair_cubo.py                    # uma amostra a cada N quadros únicos, em _tmp/
python codigo/relogio_e_corte.py                 # fator de tempo e critérios do início da janela
python codigo/experimento_prioridades12.py classic   # detector clássico de vale e decodificação
python ../MARCADOR/abrir_marcador.py             # marcações manuais
python codigo/experimento_s_sessao.py rotulos    # rótulos da S1 a partir das marcações
python codigo/experimento_s_sessao.py treinar --variante S1 --face baixo
python codigo/experimento_s_sessao.py treinar --variante S1 --face cima
python codigo/recompensa_lei.py treinar nenhum 4 400   # ajuste fino 2b
python codigo/comparacao_geral.py                # todas as linhas na mesma régua
python codigo/taxa_real_cap4.py                  # taxa, intervalos e lei de regressão
```

A marcação manual abre pelo arquivo `MARCADOR/Abrir marcador.command` ou por `python MARCADOR/abrir_marcador.py`. O `MARCADOR/LEIA-ME.txt` explica o uso.

## O que cada script faz

Os scripts estão em `caso_polietileno_2026-08-22/codigo/`.

| Etapa | Script |
|---|---|
| Valores do ensaio (lidos da configuração) | `ensaio.py` |
| Vazão de oxidante no registro do banco | `oxidante.py`, `registro_vazao.py` |
| Fator de tempo e corte do aquecimento | `relogio_e_corte.py` |
| Escala e geometria do grão | `metrologia_pe.py` |
| Leitura do vídeo, contador e tempo de cada quadro | `ler_contador.py`, `extrair_cubo.py`, `extrair_carimbo.py` |
| Detector de vale e decodificação temporal | `detectar_interfaces.py`, `experimento_prioridades12.py` |
| Programa de marcação | `MARCADOR/abrir_marcador.py`, `interface.py`, `nucleo.py`, `graficos_marcacao.py` |
| Rede S1, treinada com as marcações a partir da ImageNet | `experimento_s_sessao.py`, `experimento_p2_v2.py` |
| 2b, ajuste fino com faixa de 4 px nas marcações | `recompensa_lei.py` (alvo `nenhum`) |
| Recompensa de forma | `recompensa_oculta.py` |
| SM1 v4 e SMF v4, sem marcação | `sem_marcacao.py`, `rotulos_v3_sem_marcacao.py`, `treino_v2.py` |
| Penalidades físicas (parede, moldura e recuo) | `treino_v2.py` |
| Erro de posição nos blocos de teste e comparação entre linhas | `avaliar_linhas.py`, `comparacao_geral.py`, `figuras_comparacao_cap4.py` |
| Taxa no tempo real, intervalos e lei de regressão | `taxa_real_cap4.py`, `resultados_tg2.py`, `janelas_taxa.py`, `lei_potencia.py` |
| Degraus de vazão e expoente | `secoes_e_degraus.py`, `degraus_tg.py` |
| Perfil da taxa ao longo do grão | `perfil_axial.py` |
| Conferência da linha sem marcação | `conferencia_sem_marcacao.py` |
| Figuras sobre o vídeo e curvas de aprendizado | `figuras_tg2.py`, `figuras_tg2_resultados.py`, `figuras_tg2_sem_marcacao.py`, `fig_curvas_aprendizado.py`, `fig_validacao_zoom.py`, `figuras_sementes.py`, `figuras_galeria_queima.py` |
| Treino com parte dos quadros marcados | `metade_marcacoes.py`, `metade_marcacoes_figuras.py` |

Os outros arquivos de `codigo/` são módulos auxiliares que os da tabela importam.

## Origem

O código foi desenvolvido por Arthur Cardozo Munhoz de Oliveira no Trabalho de Graduação do Instituto Tecnológico de Aeronáutica (ITA), em 2026, com orientação do Prof. Dr. Leonardo Henrique Gouvêa.
