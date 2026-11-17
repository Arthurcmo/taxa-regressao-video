"""
Passagem unica pelo video: monta o cubo espaco-temporal (t, y, x_bin, BGR)
e grava quadros de referencia em frames/.

IMPORTANTE -- o eixo de tempo. O arquivo tem 9624 quadros a 29,97 fps, mas
1925 deles (exatamente 20,0 %) sao COPIA do anterior: e a assinatura de uma
conversao de 24 para 30 quadros por segundo feita na recodificacao. Quadro
repetido nao e medida, e ruido correlacionado que estraga qualquer derivada.

Entao este script NAO indexa por tempo do conteiner. Ele le
`resultados/quadros_camera.json` (produzido por `ler_contador.py`, que decodifica
o contador que a propria camera gravou na imagem) e usa so os quadros unicos,
numerados pelo contador da camera. O eixo passa a ser QUADRO DE CAMERA, que e
uma grandeza da aquisicao e nao do arquivo.

O cubo e a base de tudo que vem depois: kimograma, deteccao do traco preto,
a(t) e taxa de regressao. Fica em _tmp/ porque e regeneravel em ~30 s.
"""
import argparse, json, time
from pathlib import Path
import cv2, numpy as np

import ensaio as E

RAIZ = Path(__file__).resolve().parent.parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(E.VIDEO))
    ap.add_argument("--passo", type=int, default=8,
                    help="1 amostra a cada N quadros UNICOS de camera")
    ap.add_argument("--camera", default=str(RAIZ / "resultados/quadros_camera.json"))
    ap.add_argument("--fps-camera", type=float, default=E.FPS_CAMERA,
                    help="taxa nominal da camera (o relogio gravado e de 30 fps)")
    ap.add_argument("--bin-x", type=int, default=20, help="largura do bin em x [px]")
    ap.add_argument("--roi", type=int, nargs=4, default=E.ROI,
                    metavar=("Y0", "Y1", "X0", "X1"))
    ap.add_argument("--dt-quadro", type=float, default=4.0, help="salvar 1 quadro a cada dt [s]")
    ap.add_argument("--out", default=str(RAIZ / "_tmp"))
    a = ap.parse_args()

    Y0, Y1, X0, X1 = a.roi
    cam = json.loads(Path(a.camera).read_text())
    unico = np.asarray(cam["unico"], bool)
    contador = np.asarray(cam["contador"], int)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = X1 - X0
    nb = W // a.bin_x
    X1 = X0 + nb * a.bin_x                       # corta a sobra
    H = Y1 - Y0

    dirf = RAIZ / "frames"; dirf.mkdir(exist_ok=True)
    for velho in dirf.glob("*.jpg"):
        velho.unlink()
    passo_q = max(1, int(round(a.dt_quadro * a.fps_camera)))

    cubo, qcam, idxs = [], [], []
    i = k_unico = 0; t0 = time.time()
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i < len(unico) and unico[i]:
            if k_unico % a.passo == 0:
                ok, f = cap.retrieve()
                if not ok:
                    break
                roi = f[Y0:Y1, X0:X1].astype(np.float32)
                # media em blocos de bin_x colunas -> (H, nb, 3)
                b = roi.reshape(H, nb, a.bin_x, 3).mean(2)
                cubo.append(np.clip(b, 0, 255).astype(np.uint8))
                qcam.append(int(contador[i])); idxs.append(i)
                if contador[i] % passo_q < a.passo:
                    cv2.imwrite(str(dirf / f"cam{contador[i]:06d}.jpg"), f,
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
            k_unico += 1
        i += 1
    cap.release()

    C = np.stack(cubo)                            # (N, H, nb, 3) BGR
    out = Path(a.out); out.mkdir(exist_ok=True)
    np.save(out / "cubo_bgr.npy", C)
    meta = dict(video=a.video, fps_arquivo=float(fps), n_quadros_arquivo=n,
                n_unicos=int(unico.sum()), passo=a.passo,
                fps_camera=a.fps_camera, bin_x=a.bin_x, roi=[Y0, Y1, X0, X1],
                forma=list(C.shape), dq=a.passo,
                quadro_camera=qcam, quadro_arquivo=idxs)
    (out / "cubo_meta.json").write_text(json.dumps(meta))
    print(f"cubo {C.shape}  1 amostra a cada {a.passo} quadros de camera "
          f"({time.time()-t0:.0f}s)")
    print(f"quadros do arquivo {n}, unicos {int(unico.sum())}, "
          f"amostrados {len(qcam)}  (camera {qcam[0]} a {qcam[-1]})")
    print(f"ROI y[{Y0},{Y1}) x[{X0},{X1})  bins={nb}")

if __name__ == "__main__":
    main()
