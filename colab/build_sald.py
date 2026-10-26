"""Build the SALD held-out test notebook (Colab T4), self-checking like iaudit_colab_auto.ipynb.

SALD (Southwest University Adult Lifespan Dataset) was only a *test* set in the BrainAgeNeXt
paper, so it is a clean held-out cohort for the MAE < 4 question. 200 scans are selected,
age-balanced across 6 bins with a fixed seed.

- Scans: FCP-INDI S3, verified in Colab against S3's multipart ETag (8 MiB parts; checked
  locally against a downloaded scan). Ages/sex: INDI sub_information.xlsx.
- Pre-flight: IXI002 is re-run each session and must reproduce the laptop (same check and
  reference values as the IXI notebook).
- BrainAgeNeXt orientation is NOT re-chosen on SALD: it stays locked to the IXI run's result.
- Heads: ridge heads are trained on the IXI Guys+HH features already on Drive (outputs/ of the
  IXI run) and applied unchanged to SALD.

Output: E:\\iAuditData\\colab_bundle\\iaudit_colab_sald.ipynb
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_auto  # noqa: E402  (reuse its settings/install cells and embedded IXI data)

PROBE = Path(r"E:\iAuditData\sald_probe")
XLSX_URL = "http://fcon_1000.projects.nitrc.org/indi/retro/SALD/sub_information.xlsx"
S3 = "https://fcp-indi.s3.amazonaws.com/"
N_TARGET, SEED = 200, 42
BINS = [19, 30, 40, 50, 60, 70, 81]
ORIENT_LOCKED = json.loads((Path(r"D:\Projects\iAuditAge\results") / "bang_orientation.json").read_text())


def select_sald():
    PROBE.mkdir(exist_ok=True)
    xlsx = PROBE / "sub_information.xlsx"
    if not xlsx.exists():
        xlsx.write_bytes(requests.get(XLSX_URL, timeout=120).content)
    demo = pd.read_excel(xlsx)
    demo = demo[demo["T1Img"] == 1].copy()
    demo["sex"] = demo["Sex"].astype(str).str.strip().str.upper()
    lst = pd.read_csv(PROBE / "sald_t1_listing.csv")      # from sald_listing.py
    df = demo.merge(lst, left_on="Sub_ID", right_on="sub_id", how="inner")
    assert df["etag"].str.endswith("-2").all() and df["size"].between(8 * 2**20 + 1, 16 * 2**20).all()

    df["bin"] = pd.cut(df["Age"], BINS, right=False)
    avail = df["bin"].value_counts().sort_index()
    quota = {b: 0 for b in avail.index}
    left = N_TARGET
    open_bins = [b for b in avail.index if avail[b] > 0]
    while left > 0 and open_bins:                         # equal share, spill over from small bins
        share = max(1, left // len(open_bins))
        for b in list(open_bins):
            take = min(share, avail[b] - quota[b], left)
            quota[b] += take
            left -= take
            if quota[b] == avail[b]:
                open_bins.remove(b)
            if left == 0:
                break
    rng = np.random.default_rng(SEED)
    picks = [g.iloc[rng.permutation(len(g))[:quota[b]]] for b, g in df.groupby("bin", observed=True)]
    sel = pd.concat(picks).sort_values("Sub_ID")
    print("available per age bin:", avail.to_dict())
    print("selected per age bin: ", sel["bin"].value_counts().sort_index().to_dict(),
          "| sex:", sel["sex"].value_counts().to_dict(), "| n =", len(sel))
    return pd.DataFrame({
        "scan_id": [f"SALD_{s:06d}" for s in sel["Sub_ID"]], "dataset": "SALD", "site": "SALD",
        "chronological_age": sel["Age"].astype(float), "sex": sel["sex"],
        "fname": [k.split("/")[-1] for k in sel["key"]], "url": S3 + sel["key"],
        "etag": sel["etag"], "size": sel["size"]})


CELLS = []


def md(s):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n")})


def build(sald, ixi_csv, ref_json):
    # identical settings + install cells to the IXI notebook that ran cleanly
    build_auto.CELLS.clear()
    build_auto.build_cells(ixi_csv, ref_json)
    by_head = {c["source"].split("\n")[0]: c for c in build_auto.CELLS if c["cell_type"] == "code"}
    settings = by_head["# ===== Settings (defaults are the safe choice; normally change nothing) ====="]
    install = by_head["# ===== 1. Install (pinned to the versions validated on the laptop) ====="]

    ixi = pd.read_csv(io.StringIO(ixi_csv))
    ixi002 = json.loads(ixi[ixi["scan_id"] == "IXI002"].iloc[0].to_json())   # plain JSON types
    ixi_train = ixi[ixi["site"].isin(["Guys", "HH"])][["scan_id", "chronological_age"]]

    md(rf"""
# iAuditAge - SALD held-out test (self-checking)

**Why SALD:** the BrainAgeNeXt paper used SALD only as a *test* set, so unlike IXI and eNKI it is
not in BrainAgeNeXt's training data. This notebook measures both CNNs on {len(sald)} SALD scans
(age-balanced, ages {int(sald.chronological_age.min())}-{int(sald.chronological_age.max())}).

**How to run:** *Runtime > Change runtime type >* **T4 GPU** (CPU also works, slower), then
*Runtime > Run all* and click *Allow* on the two Google sign-in prompts.
- **From a second Google account:** first share the `iAuditAge` folder from the account that owns it
  (Drive > right-click > Share > Editor). This notebook then adds the My Drive shortcut itself, and
  stops without processing anything if the folder it finds is not the project folder.
- If Colab disconnects, *Run all* again - every finished scan is saved to
  `MyDrive/iAuditAge/sald_outputs/` and never redone.
- No GPU available? BrainAgeNeXt runs on CPU instead (~3-4 min/scan) - same results, slower.
- At the end the results zip is saved to `sald_outputs/iaudit_sald_outputs.zip` and downloaded
  automatically, then the runtime is released.

**Nothing is tuned on SALD:**
- BrainAgeNeXt orientation stays locked to the IXI run's choice (**{ORIENT_LOCKED['orientation']}**).
- Ridge heads are trained only on the IXI Guys+HH features from the earlier run
  (`MyDrive/iAuditAge/outputs/`) and applied to SALD unchanged.

**Safety checks** (the run stops itself, logs why, and frees the runtime):
- every scan is verified against S3's checksum (multipart ETag); IXI002 against its MD5
- pre-flight: IXI002 is re-run each session and must reproduce the laptop (DeepBrainNet +/-1 y,
  features r>=0.98; BrainAgeNeXt +/-1.5 y in all 4 orientations)
- watchdogs: 3 failures in a row or >5% failed; prediction collapse (near-constant output);
  implausible running error (DeepBrainNet > 20 y, BrainAgeNeXt > 15 y over 20 scans)
- every saved file is re-loaded and shape/NaN-checked before it counts as done

The last cell (export) is only needed if the automatic download did not start.
""")
    code(settings["source"])
    code(install["source"])
    code(r"""
# ===== 1b. Find the iAuditAge folder in this account's My Drive; if it was only shared with this
#            account, add a My Drive shortcut to it (Drive API), so the paths below work unchanged =====
from google.colab import auth
from googleapiclient.discovery import build as gbuild
auth.authenticate_user()
svc = gbuild('drive', 'v3')

def drive_find(query):
    return svc.files().list(q=query, fields='files(id,name,mimeType,owners(emailAddress))', pageSize=50,
                            supportsAllDrives=True, includeItemsFromAllDrives=True).execute().get('files', [])

in_root = drive_find(f"name = '{DRIVE_FOLDER}' and 'root' in parents and trashed = false")
if in_root:
    kinds = ['shortcut' if f['mimeType'].endswith('shortcut') else 'folder' for f in in_root]
    print(f"'{DRIVE_FOLDER}' already in My Drive ({', '.join(kinds)}) - using it")
else:
    shared = drive_find(f"name = '{DRIVE_FOLDER}' and sharedWithMe = true and trashed = false "
                        "and mimeType = 'application/vnd.google-apps.folder'")
    if len(shared) != 1:
        raise Stop(f"found {len(shared)} folders named '{DRIVE_FOLDER}' shared with this account (need exactly 1). "
                   "In the account that owns the data: Drive > right-click iAuditAge > Share > add this "
                   "account as Editor, then Run all again.")
    svc.files().create(body={'name': DRIVE_FOLDER, 'mimeType': 'application/vnd.google-apps.shortcut',
                             'shortcutDetails': {'targetId': shared[0]['id']}, 'parents': ['root']},
                       fields='id').execute()
    owner = shared[0].get('owners', [{}])[0].get('emailAddress', '?')
    print(f"added a My Drive shortcut to the '{DRIVE_FOLDER}' folder shared by {owner}")
""")
    code("# ===== 2. Embedded data (do not edit) =====\n"
         f"SALD_CSV = {json.dumps(sald.to_csv(index=False))}\n"
         f"IXI_TRAIN_CSV = {json.dumps(ixi_train.to_csv(index=False))}\n"
         f"IXI002 = {json.dumps(ixi002)}\n"
         f"REF_JSON = {json.dumps(ref_json)}\n"
         f"ORIENT = {json.dumps(ORIENT_LOCKED['orientation'])}   # locked on IXI, never re-chosen here\n")

    code(r"""
# ===== 3. Drive, logging, safety helpers =====
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import io, json, time, shutil, hashlib, pathlib, datetime
import numpy as np, pandas as pd, requests
from google.colab import drive, runtime
drive.mount('/content/drive')

DRIVE = pathlib.Path('/content/drive/MyDrive') / DRIVE_FOLDER
OUT = DRIVE / 'sald_outputs'
PREV = DRIVE / 'outputs'                      # IXI run: features used to train the heads
DBN_OUT, BANG_OUT, CACHE = OUT / 'dbn_embed', OUT / 'bang_embed', DRIVE / 'ANTsXNet'
WORK = pathlib.Path('/content/work')
for d in (DBN_OUT, BANG_OUT, CACHE, WORK / 't1', WORK / 'bang'):
    d.mkdir(parents=True, exist_ok=True)
LOG = OUT / 'run_log.txt'

manifest = pd.read_csv(io.StringIO(SALD_CSV))
REF = json.loads(REF_JSON)
N = len(manifest)

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

def s3_etag(p, part=8 * 1024 * 1024):
    # S3 multipart ETag: md5 of the concatenated per-part md5 digests, plus '-<parts>'
    digests = []
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(part), b''):
            digests.append(hashlib.md5(b).digest())
    return hashlib.md5(b''.join(digests)).hexdigest() + f'-{len(digests)}'

def fetch(url, dest, check, tries=5):
    # download to .part, verify with check(path) -> bool, then rename; retries with backoff
    dest = pathlib.Path(dest)
    if dest.exists() and check(dest):
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
            if not check(tmp):
                raise IOError(f'checksum mismatch for {dest.name}')
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
DEV = 'cuda' if GPU else 'cpu'
if not (PREV / 'bang_embed').exists() and not any(DBN_OUT.glob('*.npz')):
    raise Stop(f"{DRIVE} does not look like the project folder (no outputs/ from the IXI run, no SALD progress). "
               "If this account has its own empty 'iAuditAge' folder, rename or remove it so the shared one "
               "is used, then Run all again. Nothing was processed.")
if (OUT / 'STOPPED.txt').exists():
    log('note: the previous session stopped with: ' + (OUT / 'STOPPED.txt').read_text().strip().replace('\n', ' | '))
    (OUT / 'STOPPED.txt').unlink()
free_gb = shutil.disk_usage('/content/drive/MyDrive').free / 1e9
log(f'SALD session start  runtime={"T4 GPU" if GPU else "CPU"}  drive_free={free_gb:.1f} GB  scans={N}')
if free_gb < 2:
    stop(f'Google Drive has only {free_gb:.1f} GB free; about 1.5 GB is needed for outputs.')
""")

    code(r"""
# ===== 4. Load models + pre-flight reproduction checks (IXI002, same references as the IXI run) =====
from concurrent.futures import ThreadPoolExecutor
D1 = {s for s in manifest.scan_id if (DBN_OUT / f'{s}.npz').exists()}
D2 = {s for s in manifest.scan_id if (BANG_OUT / f'{s}.npz').exists()}
log(f'DeepBrainNet {len(D1)}/{N} done   BrainAgeNeXt {len(D2)}/{N} done')
NEED1, NEED2 = len(D1) < N, len(D2) < N
BANG_READY = NEED2                          # runs on the T4 if available, otherwise on CPU (slower, same result)
if NEED2 and not GPU:
    log(f'no GPU in this session: BrainAgeNeXt will run on CPU (~3-4 min/scan for {N - len(D2)} scans)')

def t1_of(r):
    return fetch(r.url, WORK / 't1' / r.fname, lambda p: s3_etag(p) == r.etag)

import tensorflow as tf
for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)    # leave GPU memory for BrainAgeNeXt
import ants, tf_keras as keras
from antspynet.utilities import preprocess_brain_image, set_antsxnet_cache_directory, get_pretrained_network
set_antsxnet_cache_directory(str(CACHE))
h5 = get_pretrained_network('brainAgeDeepBrainNet')
if md5sum(h5) != REF['antsxnet']['brainAgeDeepBrainNet.h5']:
    stop('DeepBrainNet weights differ from the laptop copy.')
model = keras.models.load_model(h5)
feat_model = keras.Model(model.input, [model.get_layer('global_average_pooling2d_2').output,
                                       model.get_layer('dense_3').output, model.output])
SLICES = list(range(45, 125))

def dbn_scan(scan_id, t1_path, save=True):
    # returns (pred, gap-mean features, local prep path); with save=True also writes to Drive
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
    lp = WORK / f'{scan_id}_prep.nii.gz'
    ants.image_write(img, str(lp))
    if save:
        ln = WORK / f'{scan_id}.npz'
        np.savez_compressed(ln, gap=gap.astype(np.float16), dense=dense.astype(np.float16),
                            slice_pred=sp[:, 0].astype(np.float32), predicted_age=np.float32(pred))
        if not dbn_ok(ln):
            raise ValueError('saved features failed the shape/NaN check')
        put(lp, DBN_OUT / lp.name)
        put(ln, DBN_OUT / ln.name)          # npz last: it is what marks the scan as done
        ln.unlink()
        D1.add(scan_id)
    return pred, gap.astype(np.float32).mean(0), lp

log('pre-flight: re-running IXI002 through DeepBrainNet and comparing with the laptop ...')
ixi_t1 = fetch(IXI002['url'], WORK / 't1' / IXI002['fname'], lambda p: md5sum(p) == IXI002['md5'])
pred0, g0, prep0 = dbn_scan('IXI002', ixi_t1, save=False)
ref = REF['dbn_ref']['IXI002']
dp, c = abs(pred0 - ref['pred']), float(np.corrcoef(g0, np.array(ref['gap_mean']))[0, 1])
log(f'  reproduce IXI002: colab={pred0:.2f} laptop={ref["pred"]:.2f} (diff {dp:.2f} y)  feature r={c:.4f}')
if dp > 1.0 or c < 0.98:
    stop(f'pre-flight failed: DeepBrainNet does not reproduce the laptop (diff {dp:.2f} y, r={c:.3f}).')
for f in CACHE.iterdir():
    if f.name in REF['antsxnet'] and md5sum(f) != REF['antsxnet'][f.name]:
        stop(f'antspynet file {f.name} differs from the laptop copy.')
log('DeepBrainNet pre-flight passed')

VARIANTS = {'as_is': None, 'flip_LR': 2, 'flip_AP': 3, 'flip_SI': 4}   # tensor is (1, 1, X, Y, Z)
if BANG_READY:
    B = WORK / 'bang'
    fetch('https://raw.githubusercontent.com/FrancescoLR/BrainAgeNeXt/main/BrainAge_estimation.py',
          B / 'BrainAge_estimation.py', lambda p: md5sum(p) == REF['bang_code_md5'])
    for name, h in REF['bang_weights'].items():
        fetch(f'https://huggingface.co/FrancescoLR/BrainAgeNeXt/resolve/main/{name}', B / name,
              lambda p, h=h: md5sum(p) == h)
    sys.path.insert(0, str(B))
    from BrainAge_estimation import MedNeXtEncReg, prepare_transforms   # official file, MD5-verified
    bang = []
    for i in range(1, 6):
        m = MedNeXtEncReg().to(DEV)
        m.load_state_dict(torch.load(B / f'BrainAge_{i}.pth', map_location=DEV))
        bang.append(m.eval())
    tfm = prepare_transforms()

    def bang_run(prep_path, flip_dim):
        x = torch.as_tensor(np.asarray(tfm({'image': str(prep_path)})['image']))[None].to(DEV)
        if flip_dim is not None:
            x = torch.flip(x, dims=[flip_dim])
        feats, preds = [], []
        with torch.no_grad():
            for m in bang:
                f = torch.flatten(m.global_avg_pool(m.mednextv1(x)), 1)   # same path as forward()
                feats.append(f[0].float().cpu().numpy())
                preds.append(float(m.regression_fc(f).squeeze()))
        return np.stack(feats), np.array(preds, dtype=np.float32)

    checks = REF['bang_ref']['IXI002'] if GPU else {ORIENT: REF['bang_ref']['IXI002'][ORIENT]}  # CPU: skip 3 extra runs
    log(f'pre-flight: BrainAgeNeXt on IXI002 ({len(checks)} orientation(s), {DEV}) vs the laptop ...')
    for k, refv in checks.items():
        got = float(np.median(bang_run(prep0, VARIANTS[k])[1]))
        log(f'  reproduce BrainAgeNeXt IXI002 {k}: colab={got:.1f} laptop={refv:.1f}')
        if abs(got - refv) > 1.5:
            stop(f'BrainAgeNeXt does not reproduce the laptop ({k}: {got:.1f} vs {refv:.1f}).')
    log(f'BrainAgeNeXt pre-flight passed; orientation locked from the IXI run: {ORIENT}')
prep0.unlink(missing_ok=True)
""")

    code(r"""
# ===== 5. Main loop: each SALD scan -> DeepBrainNet -> BrainAgeNeXt =====
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

g1, g2 = Guard('DeepBrainNet'), Guard('BrainAgeNeXt')
p1s, e1s, e2s = [], [], []

def watch(preds, errs, name, max_mae):
    # no laptop reference exists for SALD: catch broken preprocessing instead of real-but-large errors
    if len(errs) >= 20 and np.mean(errs[-20:]) > max_mae:
        stop(f'{name} MAE over the last 20 SALD scans is {np.mean(errs[-20:]):.1f} y (> {max_mae}) - '
             'more likely broken preprocessing than a real effect; check before continuing.')
    if len(preds) >= 20 and np.std(preds[-20:]) < 2:
        stop(f'{name} predictions have collapsed (SD {np.std(preds[-20:]):.2f} y over the last 20 scans).')

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
    return float(np.median(preds))

b2s = []
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
            p1, _, lp = dbn_scan(r.scan_id, t1)
            lp.unlink(missing_ok=True)
            t1.unlink(missing_ok=True)
            g1.ok()
            p1s.append(p1)
            e1s.append(abs(p1 - r.chronological_age))
            watch(p1s, e1s, 'DeepBrainNet', 20)
        except Stop:
            raise
        except Exception as e:
            g1.failed(r.scan_id, e)
            continue
    if BANG_READY and r.scan_id in D1 and r.scan_id not in D2:
        try:
            p2 = do_bang(r)
            g2.ok()
            b2s.append(p2)
            e2s.append(abs(p2 - r.chronological_age))
            watch(b2s, e2s, 'BrainAgeNeXt', 15)
        except Stop:
            raise
        except Exception as e:
            g2.failed(r.scan_id, e)
    if (i + 1) % 5 == 0:
        rate = (time.time() - t0) / (i + 1)
        log(f'working: DeepBrainNet {len(D1)}/{N}  BrainAgeNeXt {len(D2)}/{N}  '
            f'(last {r.scan_id}: age {r.chronological_age:.0f}'
            + (f', DBN {p1:.1f}' if p1 is not None else '') + (f', BANG {p2:.1f}' if p2 is not None else '')
            + f')  {rate:.0f} s/scan, ~{(len(todo) - i - 1) * rate / 3600:.1f} h left'
            + (f'  running MAE DBN {np.mean(e1s):.2f} / BANG {np.mean(e2s):.2f}' if e1s and e2s else ''))
pool.shutdown(wait=False)

log(f'finished this pass: DeepBrainNet {len(D1)}/{N}  BrainAgeNeXt {len(D2)}/{N}')
if N - len(D1) > 0.05 * N:
    stop(f'{N - len(D1)} scans still missing DeepBrainNet features; Run all again to retry them.')
if N - len(D2) > 0.05 * N:
    stop(f'{N - len(D2)} scans still missing BrainAgeNeXt features; Run all again to retry them.')
""")

    md(r"""
## Stage 3 - SALD results
Raw MAE of both CNNs on SALD, plus ridge heads trained **only on IXI Guys+HH** (features from the
earlier IXI run on Drive) and applied to SALD unchanged.
""")

    code(r"""
# ===== 6. Stage 3 - evaluate on SALD; IXI-trained heads applied unchanged =====
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def feats(dbn_path, bang_path):
    a, b = np.load(dbn_path), np.load(bang_path)
    return dict(dbn_raw=float(a['predicted_age']), bang_raw=float(b['predicted_age']),
                bang_sd=float(np.std(b['preds'])), gap=a['gap'].astype(np.float32).mean(0),
                dense=a['dense'].astype(np.float32).mean(0), bangf=b['feat'].astype(np.float32).mean(0),
                **{f'bang_model{i + 1}': float(v) for i, v in enumerate(b['preds'])})

recs = []
for r in manifest.itertuples():
    a, b = DBN_OUT / f'{r.scan_id}.npz', BANG_OUT / f'{r.scan_id}.npz'
    if a.exists() and b.exists() and dbn_ok(a) and bang_ok(b):
        recs.append(dict(scan_id=r.scan_id, age=r.chronological_age, sex=r.sex, **feats(a, b)))
sald = pd.DataFrame(recs)
log(f'stage 3: {len(sald)} SALD scans with both feature sets')

mae = lambda p, y: float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))
def row(name, p):
    p, y = np.asarray(p), sald['age'].to_numpy()
    return {'model': name, 'MAE': mae(p, y), 'mean_error': float(np.mean(p - y)),
            'r': float(np.corrcoef(p, y)[0, 1]), 'n': len(y)}
table = [row('DeepBrainNet raw', sald['dbn_raw']), row('BrainAgeNeXt raw', sald['bang_raw'])]

train = pd.read_csv(io.StringIO(IXI_TRAIN_CSV))
tr = []
for r in train.itertuples():
    a, b = PREV / 'dbn_embed' / f'{r.scan_id}.npz', PREV / 'bang_embed' / f'{r.scan_id}.npz'
    if a.exists() and b.exists():
        tr.append(dict(age=r.chronological_age, **feats(a, b)))
tr = pd.DataFrame(tr)
log(f'IXI training features found on Drive: {len(tr)}/{len(train)}')
if len(tr) >= 450:
    X = lambda df, cols: np.hstack([np.vstack(df[c].to_numpy()) for c in cols])
    for name, cols in {'DBN dense1024 + ridge': ['dense'], 'BANG 512 + ridge': ['bangf'],
                       'DBN gap + BANG + ridge': ['gap', 'bangf']}.items():
        head = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 20)))
        head.fit(X(tr, cols), tr['age'])
        sald[f'pred {name}'] = head.predict(X(sald, cols))
        table.append(row(name + ' (trained on IXI)', sald[f'pred {name}']))
else:
    log('not enough IXI features on Drive (outputs/ of the IXI run) - heads skipped, raw results only')

res = pd.DataFrame(table).set_index('model').round(3)
res.to_csv(OUT / 'sald_mae.csv')
sald['age_bin'] = pd.cut(sald['age'], [19, 30, 40, 50, 60, 70, 81], right=False).astype(str)
by_bin = sald.groupby('age_bin').apply(lambda g: pd.Series({
    'n': len(g), 'DBN': mae(g['dbn_raw'], g['age']), 'BANG': mae(g['bang_raw'], g['age'])})).round(2)
sald.drop(columns=['gap', 'dense', 'bangf']).to_csv(OUT / 'sald_predictions.csv', index=False)
log('SALD MAE (held out for BrainAgeNeXt; heads never saw SALD):\n' + res.to_string())
log('MAE by age bin:\n' + by_bin.to_string())
(OUT / 'DONE.txt').write_text(f'{datetime.datetime.now()}\n' + res.to_string() + '\n')

# export: zip saved to Drive (sald_outputs/iaudit_sald_outputs.zip) and downloaded to this browser
import zipfile
from google.colab import files
zpath = WORK / 'iaudit_sald_outputs.zip'
with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as zf:
    for name in ['sald_predictions.csv', 'sald_mae.csv', 'run_log.txt', 'DONE.txt']:
        if (OUT / name).exists():
            zf.write(OUT / name, name)
    for f in sorted(BANG_OUT.glob('*.npz')):
        zf.write(f, f'bang_embed/{f.name}')
put(zpath, OUT / zpath.name)
log(f'ALL DONE. Results zip saved to Drive: {OUT / zpath.name}  ({zpath.stat().st_size / 1e6:.1f} MB)')
try:
    files.download(str(zpath))
    time.sleep(20)                          # let the browser download start before the runtime is released
except Exception as e:
    log(f'browser download did not start ({e!r}) - take the zip from Drive instead')
release()
""")

    md(r"""
## Export for the GitHub repo
Run **only this cell** after the run has finished (a CPU runtime is enough). It downloads
`iaudit_sald_outputs.zip`.
""")

    code(r"""
# ===== 7. Export SALD results for the repo (run ONLY this cell; CPU runtime is enough) =====
import pathlib, zipfile
import pandas as pd
from google.colab import drive, files
drive.mount('/content/drive')
OUT = pathlib.Path('/content/drive/MyDrive/iAuditAge/sald_outputs')
NEED = ['sald_predictions.csv', 'sald_mae.csv', 'run_log.txt']
missing = [f for f in NEED if not (OUT / f).exists()]
if missing:
    raise SystemExit(f'not found in {OUT}: {missing} - has the run finished?')
print(pd.read_csv(OUT / 'sald_mae.csv').to_string(index=False))
bang = sorted((OUT / 'bang_embed').glob('*.npz'))
zpath = pathlib.Path('/content/iaudit_sald_outputs.zip')
with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as zf:
    for name in NEED + ['DONE.txt']:
        if (OUT / name).exists():
            zf.write(OUT / name, name)
    for f in bang:
        zf.write(f, f'bang_embed/{f.name}')
print(f'{zpath.name}: {zpath.stat().st_size / 1e6:.1f} MB ({len(bang)} BrainAgeNeXt feature files)')
files.download(str(zpath))
""")


def main():
    ixi_csv, ref_json = build_auto.embedded_data(HERE / "iaudit_colab_auto.ipynb")
    sald = select_sald()
    build(sald, ixi_csv, ref_json)
    nb = {"nbformat": 4, "nbformat_minor": 0,
          "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "cells": CELLS}
    for c in nb["cells"]:
        c["source"] = c["source"].splitlines(keepends=True)
    p = HERE / "iaudit_colab_sald.ipynb"
    p.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    json.loads(p.read_text(encoding="utf-8"))
    print(f"notebook -> {p}  ({p.stat().st_size / 1e3:.0f} KB, {len(sald)} SALD scans)")


if __name__ == "__main__":
    main()
