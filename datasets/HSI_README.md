# HSI FAP: cube-space PGD

Datasets: HoustonU, Indian_pines, KSC, PaviaC, PaviaU, Salinas.
Use the lowercase YAML names in configs/datasets. Default root:
/home/vivek/HSI_DATA; --root overrides it. PaviaC currently has no MAT files.

## Pipeline

Whole raw HSI cube -> one-time global min-max scaling to [0,1]
-> add one shared PGD delta -> zero-padded patch slicing -> trainable B-to-3 adapter -> bicubic resize to
CLIP resolution -> CLIP mean/std normalization -> frozen CLIP with trainable FAP prompts.
The adapter is saved and restored with the FAP model checkpoint. When best-val
checkpoint selection is enabled, the scalar adversarial validation accuracy
selects the checkpoint; clean and adversarial metrics are both logged.

Reference: physics_spectral_attack.pdf, Sections 1, 5 (Eq. 12), 6 and 7.
This implements its conventional PGD-on-cube baseline, not the proposed
physics-parameterized F1/F2/F3 attack. The FAP outer loss is retained:
CE(clean) + LAMBDA_1 * cosine-weighted KL; ADV_TERM=ce instead uses CE(adv).

The whole scene is scaled once before PGD using
X01 = (Xraw - Xraw.min()) / (Xraw.max() - Xraw.min()). Negative raw values are
handled by the subtraction; a constant cube maps to zero. The same scaled cube
is used by clean and adversarial paths. Scaling is never refitted to X01+delta.
No per-band standardization or spectral L2 normalization is applied.

ATTACK.HSI_PGD.EPS and ALPHA are configured in 1/255 units: config value 1
means 1/255 of a [0,1]-cube unit, converted once in attack_cube_pgd. LOWER=0 and
UPPER=1 keep X01+delta in [0,1]. Both perturbation budgets and gradients are in
the scaled cube space; the scene's original shape and all bands are preserved.

Gradients from patch microbatches accumulate directly into the shared scaled
cube perturbation before each PGD update. The adapter retains its existing
BatchNorm, but has no sigmoid. After resizing, CLIP channel mean/std
normalization is applied in the shared model forward for clean and adversarial
training/evaluation and every PGD step. Gradients flow through this operation.
CLIP feature normalization is unchanged.

ATTACK.HSI_PGD.NORM accepts l_inf or l_2 (one global cube L2 radius).
The attack function defaults to train_trades=False (cross-entropy, requires
labels). FAP uses train_trades=True for training (TRADES KL, paper Eq. 17) and
train_trades=False for validation/testing (label-based cross-entropy). For the training attack, clean predictions are detached and cached once per
attack; labels are not used in that training attack. Testing uses ground-truth labels. Random initialization is always enabled,
matching the original attack/pgd.py; there is no random-start argument or config
option. This avoids the zero KL gradient at the exact clean input.
TRAIN_ITER, TEST_ITER, ADV_TERM and LAMBDA_1 are also under ATTACK.HSI_PGD.
Both EPS and ALPHA default to 1 in configs/scripts (effective value 1/255).
Microbatch size is
ATTACK.HSI_PGD.BATCH_SIZE, independent of the outer training batch size.

During training a fresh whole-cube perturbation is optimized against ONLY the
current training centers, using current adapter and prompt weights.
Independent patch augmentation is bypassed in the FAP cube path. During each
evaluation one full cube perturbation is optimized against ALL centers of that
evaluation split, accumulating gradients across microbatches before every PGD
step. All evaluated patches then use that same attacked cube. This is a
white-box, label-based, scene-level evaluation. It can be expensive on large
scenes: reducing microbatch size reduces CLIP activation memory, but the cube,
delta, and cube gradients still require full-scene memory.

Attack steps freeze defender weights and BatchNorm statistics without blocking
input gradients, then restore modes and trainability for the outer update.
Zero padding is added during patchification and cannot itself be perturbed.

## Run

From the FAP directory, using a training environment with the project dependencies:

```bash
python train.py --trainer FAP --seed 1 \
  --dataset-config-file configs/datasets/indian_pines.yaml \
  --config-file configs/trainers/FAP/hsi_vit_b32.yaml \
  --output-dir output/indian_pines/base_seed1 \
  DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES base

python train.py --trainer FAP --seed 1 --eval-only \
  --dataset-config-file configs/datasets/indian_pines.yaml \
  --config-file configs/trainers/FAP/hsi_vit_b32.yaml \
  --model-dir output/indian_pines/base_seed1 --load-epoch 20 \
  --output-dir output/indian_pines/new_seed1 \
  DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES new
```

Use SUBSAMPLE_CLASSES=all for conventional few-shot training/evaluation.
The scripts scripts/Adv/fap/few_shot.sh and base2new.sh use the HSI config
and /home/vivek/HSI_DATA by default. DATA_ROOT, OUTPUT_DIR and PYTHON environment
variables can override their paths/interpreter. The cross_dataset.sh script
still contains legacy RGB dataset choices.

## Fixed saved splits (CoOp/MaPLe-style sampling)

Prepare partitions once, before training, from the FAP directory:

```bash
python split.py --dataset indian_pines --seed 42
# Or all available datasets (missing data folders are skipped):
python split.py --all --root /home/vivek/HSI_DATA --seed 42
```

This saves <DATASET.ROOT>/<dataset folder>/split_<dataset name>.json, e.g.
/home/vivek/HSI_DATA/Indian_pines/split_Indian_pines.json. DATASET.HSI.SPLIT_FILE selects
the filename (or an absolute path). Files contain the full train/val/test pixel
partitions, not duplicated image patches. Each entry is [row, column, zero-based
class label, classname]. Metadata records the split seed, ratios, stride,
class order, ground-truth shape and checksum. Existing files are validated and
kept; replacing one requires an explicit --overwrite. Missing splits fail with
a command to generate them; training never silently regenerates a partition.

The master partition uses the existing 60% train / 20% validation / 20% test
ratios per class. Background 0 is excluded. The split seed (--seed for split.py,
default 42) fixes the partition independently of the training seed. These are
locally generated HSI partitions, not official CoOp/MaPLe RGB benchmark files.

During training, --seed controls only the experiment's few-shot draw (and model
randomness). The full partition and test membership never change with that seed.
Training draws up to NUM_SHOTS per class without replacement from the saved
training partition. Validation draws up to min(NUM_SHOTS,4) from the saved
validation partition. Base/new modes use the first ceil(C/2) classes and the
remaining classes. New-class evaluation uses only the held-out new-class test
entries, with empty train/val lists. Indian Pines at 16 shots: all classes
252/64/2061 train/val/test; base classes 128/32/859; new classes 0/0/1202.

Run training after preparing the splits, using a fresh output directory so old
checkpoints from the previous splitting/preprocessing do not skip training:

```bash
OUTPUT_DIR=/home/vivek/FAP/output/fixed_split \
CUDA_VISIBLE_DEVICES=0 bash scripts/Adv/fap/few_shot.sh indian_pines 16 0

OUTPUT_DIR=/home/vivek/FAP/output/fixed_split \
CUDA_VISIBLE_DEVICES=0 bash scripts/Adv/fap/base2new.sh indian_pines 0
```

13 x 13 patches are extracted after location selection, with zero padding.
Neighboring train/test patches may overlap spatially: these partitions are
pixel-disjoint, not spatially disjoint patch partitions. Whole-scene min-max
scaling uses the full unlabeled cube. The bundled Dassl DataManager supports the
custom full-band wrapper and empty training lists in new-class evaluation.

DATALOADER.TEST.BATCH_SIZE and ATTACK.HSI_PGD.TEST_BATCH_SIZE are both 128.
Testing uses 50 PGD steps. The outer training and training PGD batch sizes are
4, training runs for 20 epochs, and DATALOADER.NUM_WORKERS is 8.

Checks (use your Python environment with the project dependencies):

```bash
python tests/test_hsi_splits.py
python tests/test_hsi_pgd.py
```

## Evaluation metrics

Clean and adversarial result blocks both report OA (overall accuracy, percent),
AA (mean per-class recall over classes present in ground truth, percent), and
Cohen's Kappa (unscaled, -1 to 1). Metrics are also recorded under the clean and
adversarial PGD-N scalar tags. The legacy accuracy key remains equal to OA for
checkpoint selection. Kappa is undefined (NaN) if chance agreement is 1.
