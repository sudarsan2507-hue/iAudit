"""Build the self-contained, self-checking Colab notebook (no uploads needed).

Everything the notebook needs is embedded or downloaded by it, and every input is
verified against the MD5 of the file this laptop actually used:
  - IXI T1s from NITRC (original host is 403), eNKI T1s from FCP-INDI S3
  - DeepBrainNet weights via antspynet (figshare), BrainAgeNeXt code (GitHub) + weights (HF)
  - reference results from this laptop's dbn_embed run, used as a pre-flight reproduction test

Output: E:\\iAuditData\\colab_bundle\\iaudit_colab_auto.ipynb
"""
import hashlib
import json
import re
import urllib.request
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

ROOT = Path(r"E:\iAuditData")
OUT = ROOT / "colab_bundle"
REPO = Path(r"D:\Projects\iAuditAge")
UA = {"User-Agent": "Mozilla/5.0"}
REF_SCANS = ["IXI002", "IXI012", "IXI013"]
BANG_REF_IXI002 = {"as_is": 37.2, "flip_LR": 35.5, "flip_AP": 40.1, "flip_SI": 42.7}  # bang_orient.log


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_ref():
    m = pd.read_csv(ROOT / "dbn_embed" / "manifest.csv")
    m = m[m["site"] != "IOP"].reset_index(drop=True)   # IOP excluded entirely (user decision)

    with urllib.request.urlopen(urllib.request.Request(
            "https://www.nitrc.org/ir/data/projects/ixi/experiments?format=json", headers=UA), timeout=120) as r:
        exps = json.loads(r.read())["ResultSet"]["Result"]
    nitrc = {}
    for e in exps:
        mm = re.fullmatch(r"(\d+)_(\w+)", e["label"])
        if mm:
            nitrc.setdefault((int(mm.group(1)), mm.group(2)), []).append(e["ID"])

    enki_keys = {}
    for line in (REPO / "results" / "_enki_fetch_checkpoint.jsonl").read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            if "nii_key" in rec:
                enki_keys["eNKI_" + rec["participant_id"]] = rec["nii_key"]

    rows = []
    for i, r in enumerate(m.itertuples(), 1):
        local = Path(PureWindowsPath(r.filepath))
        fname = local.name
        if r.dataset == "IXI":
            ids = nitrc.get((int(r.scan_id[3:]), r.site), [])
            if len(ids) != 1:
                raise SystemExit(f"{r.scan_id}: {len(ids)} NITRC sessions")
            url = f"https://www.nitrc.org/ir/data/experiments/{ids[0]}/scans/T1/resources/NIfTI/files/{fname}"
        else:
            url = "https://fcp-indi.s3.amazonaws.com/" + enki_keys[r.scan_id]
        rows.append(dict(scan_id=r.scan_id, dataset=r.dataset, site=r.site,
                         chronological_age=round(r.chronological_age, 4), sex=r.sex,
                         original_pred=round(r.original_pred, 4), fname=fname, url=url,
                         md5=md5(local), size=local.stat().st_size))
        if i % 100 == 0:
            print(f"  hashed {i}/{len(m)}", flush=True)
    man = pd.DataFrame(rows)

    dbn_ref = {}
    for s in REF_SCANS:
        d = np.load(ROOT / "dbn_embed" / f"{s}.npz")
        dbn_ref[s] = {"pred": round(float(d["predicted_age"]), 4),
                      "gap_mean": [round(float(x), 5) for x in d["gap"].astype(np.float32).mean(0)]}

    ref = {
        "antsxnet": {p.name: md5(p) for p in (ROOT / "keras_cache" / "ANTsXNet").iterdir()},
        "bang_weights": {f"BrainAge_{i}.pth": md5(ROOT / "brainagenext" / f"BrainAge_{i}.pth") for i in range(1, 6)},
        "bang_code_md5": md5(ROOT / "brainagenext" / "BrainAge_estimation.py"),
        "dbn_ref": dbn_ref,
        "bang_ref": {"IXI002": BANG_REF_IXI002},
    }
    return man, ref


CELLS = []


def md(s):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n")})


def build_cells(man_csv, ref_json):
    md(r"""
# iAuditAge - automatic feature extraction (self-checking)

**How to run - this is all you do:**
1. *Runtime > Change runtime type >* **T4 GPU**, then *Runtime > Run all*. Allow Google Drive access when asked.
2. If Colab disconnects (session limit, GPU quota, idle timeout), just *Run all* again. Every finished scan
   is saved to `MyDrive/iAuditAge/outputs/` and is never redone.

Scans: IXI Guys + HH (train, 5-fold CV) and eNKI (external test). IXI-IOP is excluded.
Each scan goes through DeepBrainNet and then straight into BrainAgeNeXt, so both advance together.
(It also works on a CPU runtime: it then does DeepBrainNet only and asks for a T4 at the end.)

**Safety checks - the run stops itself, logs why, and releases the Colab machine if any of these fail:**
- every downloaded file must match the MD5 of the file used on the laptop
- pre-flight: one scan is re-run fresh each session and must reproduce the laptop's result (+/-1 y, features r>=0.98)
- every scan's DeepBrainNet age is compared to the laptop's original prediction (drift watchdog)
- BrainAgeNeXt must reproduce the laptop's orientation test on IXI002, and its error must stay plausible
- 3 failed scans in a row, or more than 5% failed, stops the run
- every saved file is re-loaded and shape/NaN-checked before it counts as done

Log: `MyDrive/iAuditAge/outputs/run_log.txt`. If it stopped, the reason is in `outputs/STOPPED.txt`.

Reference run (T4, Oct 2026): all 531 scans, no safety stops, ~150 s/scan for DeepBrainNet (CPU-bound
ANTs registration) and ~5 s/scan for BrainAgeNeXt. Results: `results/transfer_head_mae.csv` in the repo.
""")

    code(r"""
# ===== Settings (defaults are the safe choice; normally change nothing) =====
AUTO_RELEASE_RUNTIME = True   # free the Colab machine when finished or when a safety check stops the run
DRIVE_FOLDER = 'iAuditAge'
""")

    code(r"""
# ===== 1. Install (pinned to the versions validated on the laptop) =====
import sys, os, subprocess, importlib.metadata as im

class Stop(Exception):
    pass

WATCH = ('numpy', 'scipy', 'pandas')
loaded = [p for p in WATCH if p in sys.modules]
before = {p: im.version(p) for p in WATCH}
tfv = '.'.join(im.version('tensorflow').split('.')[:2])
pkgs = ['antspyx==0.6.3', 'antspynet==0.3.2', f'tf_keras~={tfv}.0', 'monai==1.6.1', 'torchio==1.2.1',
        'git+https://github.com/FrancescoLR/MedNeXt.git@463813b1c7352edb487f43b1eecebe507429a195']
print('installing (about 2-4 min) ...', flush=True)
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *pkgs], capture_output=True, text=True)
if r.returncode:
    print(r.stdout[-2000:], r.stderr[-4000:])
    raise Stop('pip install failed - nothing was run. Send this output.')
env = dict(os.environ, TF_USE_LEGACY_KERAS='1', TF_CPP_MIN_LOG_LEVEL='3')
chk = subprocess.run([sys.executable, '-c',
                      'import ants, antspynet, tf_keras, monai, torchio, nnunet_mednext; print("imports ok")'],
                     capture_output=True, text=True, env=env)
print(chk.stdout.strip() or chk.stderr[-3000:])
if chk.returncode:
    raise Stop('installed packages do not import - nothing was run. Send this output.')
changed = [p for p in loaded if im.version(p) != before[p]]
if changed:
    print(f'{changed} changed after already being loaded -> restarting the runtime once. Click Run all again.')
    os.kill(os.getpid(), 9)
""")

    code("# ===== 2. Embedded data (do not edit): scan list with download URLs + MD5s, laptop reference results =====\n"
         f"MANIFEST_CSV = {json.dumps(man_csv)}\n"
         f"REF_JSON = {json.dumps(ref_json)}\n")

    code(r"""
# ===== 3. Drive, logging, safety helpers =====
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import io, json, time, shutil, hashlib, pathlib, datetime
import numpy as np, pandas as pd, requests
from google.colab import drive, runtime
drive.mount('/content/drive')

DRIVE = pathlib.Path('/content/drive/MyDrive') / DRIVE_FOLDER
OUT = DRIVE / 'outputs'
DBN_OUT, BANG_OUT, CACHE = OUT / 'dbn_embed', OUT / 'bang_embed', DRIVE / 'ANTsXNet'
WORK = pathlib.Path('/content/work')
for d in (DBN_OUT, BANG_OUT, CACHE, WORK / 't1', WORK / 'bang'):
    d.mkdir(parents=True, exist_ok=True)
LOG = OUT / 'run_log.txt'

manifest = pd.read_csv(io.StringIO(MANIFEST_CSV))
REF = json.loads(REF_JSON)
ROW = {r.scan_id: r for r in manifest.itertuples()}

def log(msg):
    line = f'{datetime.datetime.now():%m-%d %H:%M:%S}  {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

def release():
    if AUTO_RELEASE_RUNTIME:
        log('releasing the Colab runtime (no more compute is used)')
        try:
            drive.flush_and_unmount()
        finally:
            runtime.unassign()

def stop(reason, ok=False):
    log(('PAUSED: ' if ok else 'STOPPED: ') + reason)
    if not ok:
        (OUT / 'STOPPED.txt').write_text(f'{datetime.datetime.now()}\n{reason}\n')
    release()
    raise Stop(reason)

def md5sum(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def fetch(url, dest, md5=None, tries=5):
    # download to .part, verify MD5, then rename; retries with backoff
    dest = pathlib.Path(dest)
    if dest.exists() and (md5 is None or md5sum(dest) == md5):
        return dest
    last = None
    for i in range(tries):
        try:
            tmp = dest.with_name(dest.name + '.part')
            with requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True, timeout=(30, 300)) as r:
                r.raise_for_status()
                with open(tmp, 'wb') as f:
                    for b in r.iter_content(1 << 20):
                        f.write(b)
            if md5 and md5sum(tmp) != md5:
                raise IOError(f'MD5 mismatch for {dest.name} (server file differs from the laptop copy)')
            os.replace(tmp, dest)
            return dest
        except Exception as e:
            last = e
            time.sleep(10 * (i + 1))
    raise RuntimeError(f'download failed after {tries} tries: {url}  ({last!r})')

def put(src, dst):
    # copy to Drive under a temp name, then rename: a disconnect never leaves a half-written file
    tmp = dst.with_name(dst.name + '.part')
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)

def dbn_ok(p):
    try:
        d = np.load(p)
        return (d['gap'].shape == (80, 512) and d['dense'].shape == (80, 1024)
                and np.isfinite(d['gap']).all() and np.isfinite(d['dense']).all()
                and 0 < float(d['predicted_age']) < 120)
    except Exception:
        return False

def bang_ok(p):
    try:
        d = np.load(p)
        return d['feat'].shape == (5, 512) and np.isfinite(d['feat']).all() and np.isfinite(d['preds']).all()
    except Exception:
        return False

import torch
GPU = torch.cuda.is_available()
if (OUT / 'STOPPED.txt').exists():
    log('note: the previous session stopped with: ' + (OUT / 'STOPPED.txt').read_text().strip().replace('\n', ' | '))
    (OUT / 'STOPPED.txt').unlink()
free_gb = shutil.disk_usage('/content/drive/MyDrive').free / 1e9
log(f'session start  runtime={"T4 GPU" if GPU else "CPU"}  drive_free={free_gb:.1f} GB  scans={len(manifest)}')
if free_gb < 5:
    stop(f'Google Drive has only {free_gb:.1f} GB free; about 4 GB is needed for outputs.')
""")

    code(r"""
# ===== 4. Load models + pre-flight reproduction checks =====
from concurrent.futures import ThreadPoolExecutor
N = len(manifest)
D1 = {s for s in manifest.scan_id if (DBN_OUT / f'{s}.npz').exists()}    # DeepBrainNet done
D2 = {s for s in manifest.scan_id if (BANG_OUT / f'{s}.npz').exists()}   # BrainAgeNeXt done
log(f'DeepBrainNet {len(D1)}/{N} done   BrainAgeNeXt {len(D2)}/{N} done')
NEED1, NEED2 = len(D1) < N, len(D2) < N
BANG_READY = GPU and NEED2

def t1_of(r):
    return fetch(r.url, WORK / 't1' / r.fname, r.md5)

if NEED1:
    import tensorflow as tf
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)    # leave GPU memory for BrainAgeNeXt
    import ants, tf_keras as keras
    from antspynet.utilities import preprocess_brain_image, set_antsxnet_cache_directory, get_pretrained_network
    set_antsxnet_cache_directory(str(CACHE))
    h5 = get_pretrained_network('brainAgeDeepBrainNet')
    if md5sum(h5) != REF['antsxnet']['brainAgeDeepBrainNet.h5']:
        stop('DeepBrainNet weights downloaded by antspynet differ from the laptop copy.')
    model = keras.models.load_model(h5)
    feat_model = keras.Model(model.input, [model.get_layer('global_average_pooling2d_2').output,
                                           model.get_layer('dense_3').output, model.output])
    SLICES = list(range(45, 125))

    def dbn_scan(r, t1_path, save=True):
        prep = preprocess_brain_image(ants.image_read(str(t1_path)), truncate_intensity=(0.01, 0.99),
                                      brain_extraction_modality='t1', template='croppedMni152',
                                      template_transform_type='antsRegistrationSyNQuickRepro[a]',
                                      do_bias_correction=True, do_denoising=True, verbose=False)
        img = prep['preprocessed_image'] * prep['brain_mask']
        img = (img - img.min()) / (img.max() - img.min())
        batch = np.zeros((len(SLICES), *img.shape[0:2], 3), dtype=np.float32)
        for j, idx in enumerate(SLICES):
            batch[j] = ants.slice_image(img, axis=2, idx=idx).numpy()[..., None]
        gap, dense, sp = feat_model.predict(batch, verbose=0)
        pred = float(np.median(sp[:, 0]))
        if save:
            lp, ln = WORK / f'{r.scan_id}_prep.nii.gz', WORK / f'{r.scan_id}.npz'
            ants.image_write(img, str(lp))
            np.savez_compressed(ln, gap=gap.astype(np.float16), dense=dense.astype(np.float16),
                                slice_pred=sp[:, 0].astype(np.float32), predicted_age=np.float32(pred))
            if not dbn_ok(ln):
                raise ValueError('saved features failed the shape/NaN check')
            put(lp, DBN_OUT / lp.name)
            put(ln, DBN_OUT / ln.name)          # npz last: it is what marks the scan as done
            lp.unlink(); ln.unlink()
            D1.add(r.scan_id)
        return pred, gap.astype(np.float32).mean(0)

    def check_ref(sid, pred, gmean):
        ref = REF['dbn_ref'][sid]
        dp = abs(pred - ref['pred'])
        c = float(np.corrcoef(gmean, np.array(ref['gap_mean']))[0, 1])
        log(f'  reproduce {sid}: colab={pred:.2f} laptop={ref["pred"]:.2f} (diff {dp:.2f} y)  feature r={c:.4f}')
        if dp > 1.0 or c < 0.98:
            stop(f'pre-flight failed on {sid}: Colab does not reproduce the laptop result '
                 f'(diff {dp:.2f} y, feature r={c:.3f}). Nothing else was run.')

    # re-run IXI002 fresh every session (Colab's preinstalled packages can change between sessions)
    log('pre-flight: re-running IXI002 through DeepBrainNet and comparing with the laptop ...')
    r0 = ROW['IXI002']
    pred0, g0 = dbn_scan(r0, t1_of(r0), save='IXI002' not in D1)
    check_ref('IXI002', pred0, g0)
    for f in CACHE.iterdir():
        if f.name in REF['antsxnet'] and md5sum(f) != REF['antsxnet'][f.name]:
            stop(f'antspynet file {f.name} differs from the laptop copy.')
    log('DeepBrainNet pre-flight passed')

VARIANTS = {'as_is': None, 'flip_LR': 2, 'flip_AP': 3, 'flip_SI': 4}   # tensor is (1, 1, X, Y, Z)
LOCK = OUT / 'bang_orientation.json'
ORIENT = json.loads(LOCK.read_text())['orientation'] if LOCK.exists() else None

if BANG_READY:
    B = WORK / 'bang'
    fetch('https://raw.githubusercontent.com/FrancescoLR/BrainAgeNeXt/main/BrainAge_estimation.py',
          B / 'BrainAge_estimation.py', REF['bang_code_md5'])
    for name, h in REF['bang_weights'].items():
        fetch(f'https://huggingface.co/FrancescoLR/BrainAgeNeXt/resolve/main/{name}', B / name, h)
    sys.path.insert(0, str(B))
    from BrainAge_estimation import MedNeXtEncReg, prepare_transforms   # official file, MD5-verified
    bang = []
    for i in range(1, 6):
        m = MedNeXtEncReg().cuda()
        m.load_state_dict(torch.load(B / f'BrainAge_{i}.pth', map_location='cuda'))
        bang.append(m.eval())
    tfm = prepare_transforms()

    def bang_run(prep_path, flip_dim):
        x = torch.as_tensor(np.asarray(tfm({'image': str(prep_path)})['image']))[None].cuda()
        if flip_dim is not None:
            x = torch.flip(x, dims=[flip_dim])
        feats, preds = [], []
        with torch.no_grad():
            for m in bang:
                f = torch.flatten(m.global_avg_pool(m.mednextv1(x)), 1)   # same path as forward()
                feats.append(f[0].float().cpu().numpy())
                preds.append(float(m.regression_fc(f).squeeze()))
        return np.stack(feats), np.array(preds, dtype=np.float32)

    log('pre-flight: BrainAgeNeXt on IXI002, all 4 orientations, vs the laptop ...')
    for k, ref in REF['bang_ref']['IXI002'].items():
        got = float(np.median(bang_run(DBN_OUT / 'IXI002_prep.nii.gz', VARIANTS[k])[1]))
        log(f'  reproduce BrainAgeNeXt IXI002 {k}: colab={got:.1f} laptop={ref:.1f}')
        if abs(got - ref) > 1.5:
            stop(f'BrainAgeNeXt on Colab does not reproduce the laptop ({k}: {got:.1f} vs {ref:.1f}).')
    log('BrainAgeNeXt pre-flight passed' + (f'; orientation locked earlier: {ORIENT}' if ORIENT else ''))
""")

    code(r"""
# ===== 5. Main loop: each scan -> DeepBrainNet -> BrainAgeNeXt =====
class Guard:
    def __init__(self, name):
        self.name, self.fail, self.row = name, 0, 0
    def ok(self):
        self.row = 0
    def failed(self, sid, e):
        self.fail += 1
        self.row += 1
        log(f'  {sid} {self.name} FAILED: {e!r}')
        if self.row >= 3:
            stop(f'3 scans in a row failed in {self.name} (last: {sid}: {e!r}).')
        if self.fail > max(5, 0.05 * N):
            stop(f'{self.fail} scans failed in {self.name} (>5%).')

g1, g2, diffs, errs = Guard('DeepBrainNet'), Guard('BrainAgeNeXt'), [], []
GH = [r.scan_id for r in manifest.itertuples() if r.site in ('Guys', 'HH')]

def do_dbn(r, t1):
    pred, g = dbn_scan(r, t1)
    if r.scan_id in REF['dbn_ref']:
        check_ref(r.scan_id, pred, g)
    d = abs(pred - r.original_pred)
    diffs.append(d)
    if d > 8:
        stop(f'{r.scan_id}: DeepBrainNet {pred:.1f} vs laptop original {r.original_pred:.1f} '
             f'({d:.1f} y apart) - preprocessing has drifted.')
    if len(diffs) >= 10 and np.median(diffs[-10:]) > 2.0:
        stop(f'median gap to the laptop predictions over the last 10 scans is '
             f'{np.median(diffs[-10:]):.2f} y (>2) - preprocessing has drifted.')
    return pred

def choose_orientation():
    # chosen once on 20 Guys/HH scans, then locked in a file so every session uses the same one
    global ORIENT
    sample = [ROW[s] for s in GH if s in D1][:20]
    err = {k: [] for k in VARIANTS}
    for r in sample:
        for k, d in VARIANTS.items():
            err[k].append(abs(float(np.median(bang_run(DBN_OUT / f'{r.scan_id}_prep.nii.gz', d)[1]))
                              - r.chronological_age))
    maes = {k: float(np.mean(v)) for k, v in err.items()}
    log('orientation check (MAE, 20 Guys/HH scans): ' + '  '.join(f'{k}={v:.2f}' for k, v in maes.items()))
    best = min(maes, key=maes.get)
    if maes[best] > 10:
        stop(f'best BrainAgeNeXt orientation still has MAE {maes[best]:.1f} y - input space is wrong, '
             'not worth running on every scan.')
    LOCK.write_text(json.dumps({'orientation': best, 'mae': maes}))
    ORIENT = best
    log(f'orientation locked: {ORIENT}')

def do_bang(r):
    feats, preds = bang_run(DBN_OUT / f'{r.scan_id}_prep.nii.gz', VARIANTS[ORIENT])
    ln = WORK / f'{r.scan_id}_bang.npz'
    np.savez_compressed(ln, feat=feats.astype(np.float16), preds=preds,
                        predicted_age=np.float32(np.median(preds)), orientation=ORIENT)
    if not bang_ok(ln):
        raise ValueError('saved features failed the shape/NaN check')
    put(ln, BANG_OUT / f'{r.scan_id}.npz')
    ln.unlink()
    D2.add(r.scan_id)
    errs.append(abs(float(np.median(preds)) - r.chronological_age))
    if len(errs) >= 20 and np.mean(errs[-20:]) > 12:
        stop(f'BrainAgeNeXt MAE over the last 20 scans is {np.mean(errs[-20:]):.1f} y (>12) - stopping.')
    return float(np.median(preds))

def try_bang(r):
    if not BANG_READY or r.scan_id not in D1 or r.scan_id in D2:
        return None
    if ORIENT is None:
        if sum(s in D1 for s in GH) < 20:
            return None                       # not enough scans yet to pick the orientation
        choose_orientation()
    try:
        p = do_bang(r)
        g2.ok()
        return p
    except Stop:
        raise
    except Exception as e:
        g2.failed(r.scan_id, e)

todo = [r for r in manifest.itertuples() if r.scan_id not in D1 or (BANG_READY and r.scan_id not in D2)]
log(f'{len(todo)} scans to process this session')
pool, futs, t0 = ThreadPoolExecutor(max_workers=3), {}, time.time()
for i, r in enumerate(todo):
    for nxt in todo[i:i + 4]:                 # download the next scans while this one runs
        if nxt.scan_id not in D1 and nxt.scan_id not in futs:
            futs[nxt.scan_id] = pool.submit(t1_of, nxt)
    p1 = p2 = None
    if r.scan_id not in D1:
        try:
            t1 = futs.pop(r.scan_id).result()
            p1 = do_dbn(r, t1)
            t1.unlink(missing_ok=True)
            g1.ok()
        except Stop:
            raise
        except Exception as e:
            g1.failed(r.scan_id, e)
            continue
    p2 = try_bang(r)
    if (i + 1) % 5 == 0:
        rate = (time.time() - t0) / (i + 1)
        log(f'working: DeepBrainNet {len(D1)}/{N}  BrainAgeNeXt {len(D2)}/{N}  '
            f'(last {r.scan_id}: age {r.chronological_age:.1f}'
            + (f', DBN {p1:.1f}' if p1 is not None else '') + (f', BANG {p2:.1f}' if p2 is not None else '')
            + f')  {rate:.0f} s/scan, ~{(len(todo) - i - 1) * rate / 3600:.1f} h left')
pool.shutdown(wait=False)

if BANG_READY:                                # scans whose DeepBrainNet finished before the orientation lock
    backlog = [ROW[s] for s in manifest.scan_id if s in D1 and s not in D2]
    if backlog:
        log(f'BrainAgeNeXt backlog: {len(backlog)} scans')
    for k, r in enumerate(backlog, 1):
        try_bang(r)
        if k % 5 == 0:
            log(f'working: BrainAgeNeXt {len(D2)}/{N}')

log(f'finished this pass: DeepBrainNet {len(D1)}/{N}  BrainAgeNeXt {len(D2)}/{N}')
if N - len(D1) > 0.05 * N:
    stop(f'{N - len(D1)} scans still missing DeepBrainNet features; Run all again to retry them.')
if not GPU:
    stop('DeepBrainNet is complete. Switch to Runtime > Change runtime type > T4 GPU, then Run all.', ok=True)
if N - len(D2) > 0.05 * N:
    stop(f'{N - len(D2)} scans still missing BrainAgeNeXt features; Run all again to retry them.')
""")

    md(r"""
## Stage 3 - transfer-learning heads
Ridge heads are trained only on IXI Guys+HH (5-fold out-of-fold MAE reported); eNKI is never used to fit them.

**Caveat - pretraining overlap.** BrainAgeNeXt's own training set includes openBHB (which contains IXI)
and NKI-RS (the same Rockland sample as eNKI). Its sub-4 y errors on these cohorts are therefore not a
held-out result. A clean test needs a cohort BrainAgeNeXt only used for testing (e.g. SALD, DLBS).
""")

    code(r"""
# ===== 6. Stage 3 - transfer-learning heads (train on IXI Guys+HH only) =====
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict

recs = []
for r in manifest.itertuples():
    a, b = DBN_OUT / f'{r.scan_id}.npz', BANG_OUT / f'{r.scan_id}.npz'
    if not (a.exists() and b.exists() and dbn_ok(a) and bang_ok(b)):
        continue
    a, b = np.load(a), np.load(b)
    recs.append(dict(scan_id=r.scan_id, site=r.site, age=r.chronological_age,
                     dbn_raw=float(a['predicted_age']), bang_raw=float(b['predicted_age']),
                     gap=a['gap'].astype(np.float32).mean(0), dense=a['dense'].astype(np.float32).mean(0),
                     bangf=b['feat'].astype(np.float32).mean(0)))
df = pd.DataFrame(recs)
log(f'stage 3: {len(df)} scans with both feature sets  ' + str(df.groupby('site').size().to_dict()))

tr = df['site'].isin(['Guys', 'HH']).values
mae = lambda p, y: float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))
X = lambda cols, m: np.hstack([np.vstack(df.loc[m, c].to_numpy()) for c in cols])
table = []
for name in ('dbn_raw', 'bang_raw'):
    row = {'model': f'{name} (no head)', 'Guys+HH': mae(df.loc[tr, name], df.loc[tr, 'age'])}
    for s in ('eNKI',):
        k = (df.site == s).values
        row[s] = mae(df.loc[k, name], df.loc[k, 'age'])
    table.append(row)
for name, cols in {'DBN gap512': ['gap'], 'DBN dense1024': ['dense'], 'BANG 512': ['bangf'],
                   'DBN gap + BANG': ['gap', 'bangf']}.items():
    head = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 20)))
    oof = cross_val_predict(head, X(cols, tr), df.loc[tr, 'age'], cv=KFold(5, shuffle=True, random_state=0))
    head.fit(X(cols, tr), df.loc[tr, 'age'])
    df.loc[tr, f'pred {name}'] = oof
    row = {'model': f'{name} + ridge', 'Guys+HH': mae(oof, df.loc[tr, 'age'])}
    for s in ('eNKI',):
        k = (df.site == s).values
        p = head.predict(X(cols, k))
        df.loc[k, f'pred {name}'] = p
        row[s] = mae(p, df.loc[k, 'age'])
    table.append(row)
res = pd.DataFrame(table).set_index('model').round(2)
res.to_csv(OUT / 'transfer_head_mae.csv')
df.drop(columns=['gap', 'dense', 'bangf']).to_csv(OUT / 'transfer_head_predictions.csv', index=False)
log('MAE in years (Guys+HH = 5-fold out-of-fold; eNKI never used to train the heads):\n' + res.to_string())
log('caveat: BrainAgeNeXt was pretrained on openBHB (includes IXI) and NKI-RS - its rows are not held-out results')
(OUT / 'DONE.txt').write_text(f'{datetime.datetime.now()}\n' + res.to_string() + '\n')
log('ALL DONE. Results: outputs/transfer_head_mae.csv and outputs/transfer_head_predictions.csv')
release()
""")


def main():
    man, ref = build_ref()
    man_csv = man.to_csv(index=False)
    ref_json = json.dumps(ref, separators=(",", ":"))
    build_cells(man_csv, ref_json)
    nb = {"nbformat": 4, "nbformat_minor": 0,
          "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "cells": CELLS}
    for c in nb["cells"]:
        c["source"] = c["source"].splitlines(keepends=True)
    p = OUT / "iaudit_colab_auto.ipynb"
    p.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    json.loads(p.read_text(encoding="utf-8"))
    print(f"notebook -> {p}  ({p.stat().st_size/1e3:.0f} KB, {len(man)} scans, "
          f"{man.url.str.contains('nitrc').sum()} NITRC + {man.url.str.contains('fcp-indi').sum()} S3 URLs)")


if __name__ == "__main__":
    main()
