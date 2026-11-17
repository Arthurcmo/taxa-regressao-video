"""Experimento separado: detector clássico, referência visual e SurfaceNet.

Não sobrescreve resultados anteriores. O bloco k=180..480 é reservado;
nenhum alvo desse bloco entra no treino ou na seleção de checkpoint.
"""
from pathlib import Path
import ensaio as E
import argparse
import hashlib
import json
import time
import sys
import csv

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experimento_prioridades12_2026-10-09'
CODE = ROOT.parent / 'chamacera_2026-08-16/codigo'
sys.path.insert(0, str(CODE))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def prepare():
    OUT.mkdir(exist_ok=True)
    cube = np.load(ROOT / '_tmp/cubo_bgr.npy', mmap_mode='r')
    meta = json.loads((ROOT / '_tmp/cubo_meta.json').read_text())
    labels = np.load(ROOT / 'resultados/rotulos_humanos_face_inferior_v1.npz')
    # Bloco de teste de 80 s e embargo de 8 s de cada lado.
    eligible = [int(k) for k in labels['k'] if k < 150 or k > 510]
    val = [90, 660, 810]
    train = [k for k in eligible if k not in val]
    test = list(range(180, 481, 30))
    protocol = dict(schema=1, created='2026-10-09', train=train, validation=val,
                    test_annotations=test, test_block=[180, 480], embargo=[150, 510],
                    columns=list(range(18, 91)), face='inferior',
                    scale_mm_px=E.MM_POR_PX, fps=E.FPS_CAMERA, dq=E.QUADROS_POR_AMOSTRA,
                    calibration='configuracao_pe.json vigente; sem recalibracao',
                    source_labels='rotulos_humanos_face_inferior_v1: revisão visual assistida anterior',
                    reference_author='Codex; não é anotação humana independente',
                    classic=dict(dmax=2, lam=0.5, mono=True, bin_x=20,
                                 definition='centro do vale escuro multiescala + Viterbi'),
                    training=dict(seed=0, epochs=60, size=[592,96], base=24, depth=4,
                                  lr=0.0003, checkpoint_initial=None),
                    temporal_filter=dict(method='mediana móvel centrada', size=9,
                                         time_span_s=8*E.CAM),
                    rate='inclinação OLS da média axial nos 11 instantes reservados; extremos também reportados',
                    limits=['mesmo ensaio; sem teste entre ensaios', 'um treino, uma semente',
                            'rótulos de treino legados assistidos; teste remarcado sem previsões',
                            'validação visual por especialista pendente'],
                    hashes={str(p.relative_to(ROOT)):digest(p) for p in [
                        ROOT/'_tmp/cubo_bgr.npy', ROOT/'_tmp/cubo_meta.json',
                        ROOT/'resultados/rotulos_humanos_face_inferior_v1.npz',
                        ROOT/'configuracao_pe.json', ROOT/'codigo/detectar_interfaces.py',
                        CODE/'scripts/treinar_superficie.py', CODE/'chamacera/surfnet.py'] if p.is_relative_to(ROOT) and p.exists()})
    for p in [CODE/'scripts/treinar_superficie.py', CODE/'chamacera/surfnet.py',
              CODE/'chamacera/surfdata.py', CODE/'chamacera/models.py']:
        if p.exists():
            protocol['hashes'][str(p)] = digest(p)
    target = OUT/'protocolo.json'
    if target.exists() and json.loads(target.read_text()) != protocol:
        raise ValueError('Protocolo já congelado; não modificar silenciosamente')
    save_json(target, protocol)
    run = OUT/'run_bloco'
    imgdir = run/'dataset/img'
    imgdir.mkdir(parents=True, exist_ok=True)
    (run/'model').mkdir(exist_ok=True)
    target_y = np.zeros((len(cube),1,cube.shape[2]),np.float32)
    weight = np.zeros_like(target_y)
    for i,k in enumerate(labels['k']):
        if int(k) not in eligible:
            continue
        target_y[k,0] = labels['y'][i,1]
        weight[k,0,18:91] = labels['peso'][i,1,18:91]
        assert cv2.imwrite(str(imgdir/f'{k:06d}.png'), cube[k])
    np.save(run/'alvo.npy',target_y)
    np.save(run/'peso.npy',weight)
    config = run/'model/config.json'
    if not (run/'model_surf/best.pt').exists():
        save_json(config, dict(train_ids=[f'{k:06d}' for k in train], val_ids=[f'{k:06d}' for k in val]))
    save_json(run/'superficie.json',dict(altura_roi=cube.shape[1],largura_roi=cube.shape[2],n_surf=1))
    # Imagens de referência sem contornos de detector, rede ou rótulos anteriores.
    rawdir = OUT/'referencia_sem_previsoes'
    rawdir.mkdir(exist_ok=True)
    y0,y1,x0,x1 = meta['roi']
    for k in test:
        im = cv2.imread(str(ROOT/f'frames/cam{k*8:06d}.jpg'))
        assert im is not None
        roi = im[y0:y1,x0:x1].copy()
        assert cv2.imwrite(str(rawdir/f'k{k:04d}.png'),roi)
        # Painel em coordenadas nativas: grade para leitura de pontos, sem previsão.
        grid = roi.copy()
        for x in range(100,1860,200):
            cv2.line(grid,(x,0),(x,592),(120,120,120),1)
            cv2.putText(grid,str(x),(x+4,20),0,.55,(30,255,30),1,cv2.LINE_AA)
        for y in range(50,593,50):
            cv2.line(grid,(0,y),(1859,y),(100,100,100),1)
            cv2.putText(grid,str(y),(5,y-4),0,.55,(30,255,30),1,cv2.LINE_AA)
        cv2.imwrite(str(rawdir/f'grade_k{k:04d}.png'),grid)
    print(json.dumps(dict(train=train,validation=val,test=test),indent=2),flush=True)


def classic():
    from detectar_interfaces import detectar, quadros_com_flare
    cube=np.load(ROOT/'_tmp/cubo_bgr.npy',mmap_mode='r')
    start=time.perf_counter()
    # BGR -> luminância; nenhum alvo ou checkpoint é lido por esta etapa.
    gray=np.stack([cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) for frame in cube]).astype(np.float32)
    flare,_,_=quadros_com_flare(cube)
    result=detectar(gray,dmax=2,lam=.5,flare=flare,mono=True)
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/'classico.npz',**result,flare=flare)
    save_json(OUT/'tempo_classico.json',dict(seconds=time.perf_counter()-start,n_frames=len(cube)))
    print('Detector clássico concluído.',flush=True)


def infer():
    import torch
    from scipy.ndimage import median_filter
    from chamacera.surfnet import SurfaceNet,soft_argmax_col
    torch.set_num_threads(4)
    model=OUT/'run_bloco/model_surf'
    cfg=json.loads((model/'config.json').read_text())
    net=SurfaceNet(in_ch=3,n_surf=1,base=cfg['base'],depth=cfg['depth'],p_drop=cfg['p_drop'],seg_head=False)
    net.load_state_dict(torch.load(model/'best.pt',map_location='cpu',weights_only=True))
    net.eval()
    cube=np.load(ROOT/'_tmp/cubo_bgr.npy',mmap_mode='r')
    # Guarda margem para o filtro; só k=180..480 entra nas métricas finais.
    ks=np.arange(176,485)
    out=[]
    start=time.perf_counter()
    oh,ow=cfg['size']
    with torch.no_grad():
        for i in range(0,len(ks),4):
            ims=np.stack([cv2.resize(cube[k],(ow,oh)).astype(np.float32)/255 for k in ks[i:i+4]])
            y,_=soft_argmax_col(net(torch.from_numpy(ims).permute(0,3,1,2))[0])
            for row in y.numpy()[:,0]*cube.shape[1]/oh:
                out.append(np.interp(np.arange(cube.shape[2]),np.linspace(0,cube.shape[2]-1,ow),row))
            if i%80==0: print(f'Inferência {i}/{len(ks)}',flush=True)
    raw=np.asarray(out)
    smooth=median_filter(raw,size=(9,1),mode='nearest')
    np.savez_compressed(OUT/'rede.npz',k=ks,raw=raw,smooth=smooth)
    save_json(OUT/'tempo_rede.json',dict(seconds=time.perf_counter()-start,n_frames=len(ks),
                                       checkpoint_sha256=digest(model/'best.pt')))


def reference():
    data=json.loads((OUT/'referencia_visual.json').read_text())
    xs=np.asarray(data['x'])
    cols=np.arange(18,91)
    xc=20*(cols+.5)
    ys=[]
    dest=OUT/'referencia_conferida'
    dest.mkdir(exist_ok=True)
    for frame in data['quadros']:
        k=frame['k']
        y=np.interp(xc,xs,frame['y'])
        ys.append(y)
        im=cv2.imread(str(OUT/f'referencia_sem_previsoes/k{k:04d}.png'))
        pts=np.column_stack([xs,frame['y']]).astype(np.int32)
        cv2.polylines(im,[pts],False,(255,100,255),2,cv2.LINE_AA)
        for point in pts:
            cv2.circle(im,tuple(point),5,(255,255,255),-1)
        cv2.putText(im,f'k={k} | referencia visual do assistente (sem previsoes)',(20,35),0,.8,(255,255,255),2)
        cv2.imwrite(str(dest/f'k{k:04d}.png'),im)
    np.savez_compressed(OUT/'referencia.npz',k=[f['k'] for f in data['quadros']],cols=cols,y=ys)
    save_json(OUT/'referencia_manifesto.json',dict(reference_sha256=digest(OUT/'referencia_visual.json'),
               raw_images={f'k{f["k"]:04d}.png':digest(OUT/f'referencia_sem_previsoes/k{f["k"]:04d}.png')
                           for f in data['quadros']},n_control_points=len(xs)*len(ys),n_frames=len(ys)))


def rate(t, y, scale):
    """Taxa de uma face; não há fator 1/2 nem ancoragem em referência."""
    if y.ndim!=2 or len(t)!=len(y) or len(t)<2 or np.any(np.diff(t)<=0):
        raise ValueError('Tempos e contornos incompatíveis')
    if not np.isfinite(y).all():
        raise ValueError('Não preencher lacunas silenciosamente')
    position=y.mean(axis=1)*scale
    centered=t-t.mean()
    slope=float(centered@position/(centered@centered))
    residual=position-(position.mean()+slope*centered)
    den=float(np.sum((position-position.mean())**2))
    return dict(rate_ols_mm_s=slope,
                rate_endpoints_mm_s=float((position[-1]-position[0])/(t[-1]-t[0])),
                r2=1-float(residual@residual)/den if den>0 else None)


def evaluate():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=json.loads((OUT/'protocolo.json').read_text())
    ref=np.load(OUT/'referencia.npz')
    cfg=json.loads((OUT/'run_bloco/model_surf/config.json').read_text())
    tr=set(map(int,cfg['train_ids']));va=set(map(int,cfg['val_ids']))
    forbidden=set(range(150,511))
    assert not tr&va and not (tr|va)&forbidden
    assert cfg['checkpoint_inicial'] is None and cfg['n_surf']==1
    assert np.load(OUT/'run_bloco/peso.npy')[150:511].sum()==0
    assert digest(OUT/'referencia_visual.json')==json.loads((OUT/'referencia_manifesto.json').read_text())['reference_sha256']
    ks=ref['k'];cols=ref['cols'];t=ks*p['dq']/p['fps']
    dl=np.load(OUT/'rede.npz');cl=np.load(OUT/'classico.npz')
    idx=np.searchsorted(dl['k'],ks)
    assert np.array_equal(dl['k'][idx],ks)
    arrays={'referencia_visual':ref['y'],'classico':cl['y_inf'][ks][:,cols],
            'rede_bruta':dl['raw'][idx][:,cols], 'rede_mediana':dl['smooth'][idx][:,cols]}
    scale=p['scale_mm_px']
    results={}
    rr=rate(t,ref['y'],scale)
    for name,y in arrays.items():
        error=y-ref['y']
        result=rate(t,y,scale)
        result.update(n_frames=len(ks),n_grid_points=int(y.size),n_manual_control_points=99,
                      mae_px=float(np.abs(error).mean()),p95_px=float(np.percentile(np.abs(error),95)),
                      bias_px=float(error.mean()),mae_mm=float(np.abs(error).mean()*scale),
                      p95_mm=float(np.percentile(np.abs(error),95)*scale),
                      prediction_coverage=float(np.isfinite(y).mean()),
                      rate_error_mm_s=result['rate_ols_mm_s']-rr['rate_ols_mm_s'])
        result['rate_abs_error_mm_s']=abs(result['rate_error_mm_s'])
        result['rate_relative_error_pct']=100*result['rate_error_mm_s']/abs(rr['rate_ols_mm_s'])
        results[name]=result
    # Métricas só nos 99 pontos originais: não confundir interpolação com novas marcações.
    original=json.loads((OUT/'referencia_visual.json').read_text())
    manual_x=np.asarray(original['x'])
    grid_x=20*(cols+.5)
    manual_y=np.asarray([f['y'] for f in original['quadros']])
    for name,y in arrays.items():
        pred_at_points=np.stack([np.interp(manual_x,grid_x,row) for row in y])
        results[name]['control_point_mae_px']=float(np.abs(pred_at_points-manual_y).mean()) if name!='referencia_visual' else 0.0
    legacy=np.load(ROOT/'resultados/rotulos_humanos_face_inferior_v1.npz')
    legacy_idx=np.searchsorted(legacy['k'],ks)
    assert np.array_equal(legacy['k'][legacy_idx],ks)
    legacy_y=legacy['y'][legacy_idx,1][:,cols]
    legacy_error=legacy_y-ref['y']
    legacy_audit=dict(mae_px=float(np.abs(legacy_error).mean()),bias_px=float(legacy_error.mean()),
                      mean_bias_by_frame_px=legacy_error.mean(1).tolist(),**rate(t,legacy_y,scale),
                      purpose='Auditoria posterior da referência legada; não altera protocolo, alvos ou checkpoint.')
    summary=dict(protocol_sha256=digest(OUT/'protocolo.json'),
                 reference_sha256=digest(OUT/'referencia_visual.json'),
                 checkpoint_sha256=digest(OUT/'run_bloco/model_surf/best.pt'),
                 interval_video_s=t[[0,-1]].tolist(), interval_after_ignition_s=(t[[0,-1]]-352/30).tolist(),
                 results=results, training_best=json.loads((OUT/'run_bloco/model_surf/best.json').read_text()),
                 legacy_reference_audit=legacy_audit,
                 checks=dict(no_test_or_embargo_labels_in_training=True,no_initial_checkpoint=True,
                             no_reference_anchoring=True,same_frames_columns_scale=True),
                 warning='Concordância com referência visual do assistente no mesmo ensaio; não é exatidão metrológica nem validação humana independente.')
    save_json(OUT/'metricas.json',summary)
    with open(OUT/'comparacao.csv','w') as f:
        writer=csv.DictWriter(f,fieldnames=['metodo']+list(results['classico']))
        writer.writeheader()
        for name,row in results.items():writer.writerow(dict(metodo=name,**row))
    with open(OUT/'series_comparaveis.csv','w') as f:
        writer=csv.writer(f);writer.writerow(['k','t_video_s','t_ignicao_s']+[f'{name}_y_px' for name in arrays])
        for i,k in enumerate(ks):writer.writerow([k,t[i],t[i]-352/30]+[float(y[i].mean()) for y in arrays.values()])
    np.savez_compressed(OUT/'comparacao_contornos.npz',k=ks,columns=cols,**arrays)
    colors={'referencia_visual':'#252525','classico':'#cc6700','rede_bruta':'#2365ac','rede_mediana':'#008577'}
    names={'referencia_visual':'Referência visual do assistente','classico':'Visão clássica',
           'rede_bruta':'SurfaceNet bruta','rede_mediana':'SurfaceNet + mediana temporal'}
    fig,ax=plt.subplots(2,1,figsize=(9,7),layout='constrained')
    for name,y in arrays.items():
        ax[0].plot(t-352/30,y.mean(1),'.-',label=names[name],color=colors[name],lw=1.6)
        if name!='referencia_visual':
            ax[1].plot(t-352/30,np.abs(y-ref['y']).mean(1),'.-',label=names[name],color=colors[name])
    ax[0].set_ylabel('Posição média da face inferior (px da ROI)')
    ax[0].legend(fontsize=9);ax[0].set_title('Mesmo bloco de teste, mesmos 73 pontos axiais')
    ax[1].set_ylabel('Erro absoluto médio de contorno (px)')
    ax[1].set_xlabel('Tempo desde a ignição (s)')
    for a in ax:a.grid(alpha=.2)
    fig.savefig(OUT/'comparacao.png',dpi=180);fig.savefig(OUT/'comparacao.pdf');plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(11,9),layout='constrained')
    for ax,k in zip(axes,[180,270,390,480]):
        i=int(np.flatnonzero(ks==k)[0])
        im=cv2.imread(str(OUT/f'referencia_sem_previsoes/k{k:04d}.png'))
        ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        for name in ['referencia_visual','classico','rede_bruta']:
            ax.plot(grid_x,arrays[name][i],color=colors[name],lw=1.5,label=names[name])
        ax.set_xlim(320,1850);ax.set_ylim(570,270)
        ax.set_title(f'k={k} | t={t[i]-352/30:.2f} s desde a ignição',fontsize=10)
        ax.set_ylabel('y (px)')
    axes[0].legend(fontsize=8,ncol=3,loc='upper right');axes[-1].set_xlabel('x (px da ROI)')
    fig.savefig(OUT/'contornos_comparados.png',dpi=160);fig.savefig(OUT/'contornos_comparados.pdf');plt.close(fig)
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['prepare','classic','infer','reference','evaluate'])
    args=ap.parse_args()
    globals()[args.stage]()
