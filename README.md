# Deforestation Monitor: Machadinho d'Oeste, Rondônia (Brazil)

Detects newly cleared forest between 2022 and 2024 from free Sentinel-2 satellite imagery, using a U-Net forest segmentation model and multi-date change detection, and shows the results on an interactive map.

**Live app:** `[(https://deforestation-monitor-hkl3dpoyfd3dnfffuy2vze.streamlit.app/)`



---

## What it does

1. Downloads cloud-masked Sentinel-2 dry-season composites (June to September) for a 20 × 20 km area in Rondônia for 2022 and 2024.
2. Segments every pixel as forest / non-forest with a U-Net.
3. Compares the two forest-probability maps and keeps only changes that survive several filters.
4. Exports the detected clearings as polygons (GeoJSON) with area and confidence, and displays them in a Streamlit map with a before/after imagery toggle and a confidence slider.

```
Sentinel-2 L2A  ->  cloud mask + median composite  ->  U-Net (forest probability)
                                                              |
         Streamlit map  <-  GeoJSON polygons  <-  change detection + filters
```

## Data

| Item | Source |
|---|---|
| Imagery | Sentinel-2 L2A via Microsoft Planetary Computer (STAC), bands B02, B03, B04, B08, B11 at 10 m |
| Cloud masking | Scene Classification Layer (SCL), median composite of June to September scenes |
| Training labels | ESA WorldCover 2021, tree-cover class (noisy labels, see limitations) |
| Independent check | Hansen Global Forest Change, GFC-2025-v1.13 loss year |

Study area: about 20 × 20 km around Machadinho d'Oeste (-9.45, -62.00), UTM zone 20S. Image size is 1999 × 2012 pixels.

## Method

**Preprocessing.** Reflectance is `(DN - 1000) / 10000`, because Sentinel-2 data processed after January 2022 carries a 1000-unit offset. Each band is then standardized using statistics from 2022 only, and the same statistics are applied to 2024 so both years look alike to the model.

**Model.** U-Net with a ImageNet-pretrained ResNet34 encoder (`segmentation-models-pytorch`), 5 input channels, trained with Dice + BCE loss, Adam (lr 1e-3, cosine schedule), batch size 16, 30 epochs. Training uses 400 random 256 × 256 crops per epoch with flips and 90° rotations. The top 512 pixel rows are held out for validation (geographic split), the rest is used for training.

**Inference.** Sliding window with 64 px overlap, Hann-weighted blending to remove seams, and horizontal-flip test-time augmentation.

**Change detection.** A pixel is a candidate if forest probability was above 0.6 in 2022 and below 0.4 in 2024. Candidates then go through:
- morphological opening (removes slivers narrower than about 50 m)
- a 0.5 ha minimum area
- NDVI confirmation: the patch's mean NDVI must drop between the two dates

The app exposes the NDVI-drop threshold as a slider (default 0.30).

## Results

**Segmentation (agreement with WorldCover labels):**

| Metric | Value |
|---|---|
| Best validation IoU during training (14 held-out tiles) | 0.838 (F1 0.912) |
| Full-image IoU on the held-out validation rows | 0.862 |
| Forest fraction, 2022 / 2024 | 0.457 / 0.439 |

These scores measure how well the model reproduces WorldCover, not how accurate the forest map is against reality.

**Change detection:**

| Stage | Result |
|---|---|
| Raw forest loss (2022 forest and 2024 non-forest) | 1,160 ha |
| After hysteresis, opening, min area and NDVI filter | 155 patches, 471 ha |

**Check against Hansen forest loss, 2022 to 2024** (30 m tolerance, patches counted as matching if at least 30% overlaps):

| Setting | Patch precision | Patch recall |
|---|---|---|
| NDVI drop ≥ 0.10 (all 155 patches) | 0.50 | 0.38 |
| NDVI drop ≥ 0.30, tuning rows (43 patches) | 0.70 | 0.21 |
| NDVI drop ≥ 0.30, held-out rows (16 patches) | 0.62 | 0.20 |

In plain terms: roughly 60 to 70% of the stricter detections overlap Hansen-recorded loss, and about one fifth of Hansen's loss patches are recovered. The held-out sample is very small, so treat these figures as rough.

Other checks:
- Shifting Hansen by up to 100 m in any direction never improved agreement, so the two maps are aligned.
- At 30 m tolerance, agreement with Hansen is 0.34, against a chance baseline of 0.02 (shifted-reference test).

## Limitations

- **Most detections are on previously disturbed land.** About 86% of patches (134 of 155) have at least 10% of their area with earlier Hansen loss. The model often treats regrowth, pasture with scattered trees and ragged forest edges as "forest", and small seasonal differences between composites then register as loss. Only 21 patches lie on land with little earlier loss (62% of those overlap Hansen).
- **The reference is imperfect.** Hansen records only the first loss at each pixel and works at 30 m, so it can miss re-clearing and small clearings.
- **Noisy training labels.** WorldCover is itself a classifier output.
- **Thresholds were chosen after looking at the data.** The NDVI-drop setting was picked from sweeps on this one area, so the held-out numbers are not fully clean.
- **One area, two dates, one season.** No claim is made about other regions.
- **Not live tracking.** The pipeline is run manually. It is not scheduled, and Sentinel-2 cloud cover and revisit time would make true real-time detection impossible anyway.
- This is a change-detection prototype, not a validated measurement of deforestation.

## Repository structure

```
deforestation-monitor/
├── app/
│   ├── app.py               # Streamlit app
│   ├── changes.geojson      # detected patches
│   ├── rgb_2022.png, rgb_2024.png
│   ├── meta.json
│   └── requirements.txt
├── src/
│   ├── download.py          # STAC search, cloud-masked composites
│   ├── labels.py            # WorldCover forest labels
│   ├── preprocess.py        # reflectance, normalization
│   ├── dataset.py           # random-crop training dataset
│   ├── model.py             # U-Net
│   ├── train.py             # training loop, metrics
│   ├── predict.py           # sliding-window inference
│   ├── change_detection.py  # filters, vectorization
│   └── evaluate.py          # comparison with Hansen
└── notebooks/               # Colab notebook that runs the pipeline
```

## Running it

**View the app locally:**

```bash
pip install -r app/requirements.txt
python -m streamlit run app/app.py
```

**Reproduce the pipeline** (a GPU runtime such as a Colab T4 is recommended for training):

```bash
pip install rasterio geopandas pystac-client planetary-computer odc-stac \
    segmentation-models-pytorch scikit-image huggingface_hub
```

Then run the steps in `notebooks/`: download composites, build labels, train, predict, detect change, export. Model weights are not stored in this repo (size). Train your own, or host them on Hugging Face.

## Possible next steps

- Confirm detections with a 2025 composite and an earlier stability check, so flickering regrowth is rejected.
- Hand-label a sample of patches and evaluate against an independent source such as INPE PRODES.
- Train on more areas and years, and add Sentinel-1 radar.
- Separate primary-forest loss from loss of secondary vegetation using land history.
- Schedule the pipeline (for example with GitHub Actions) for periodic updates.

## Acknowledgements

Sentinel-2 data: Copernicus / ESA, accessed via Microsoft Planetary Computer. Labels: ESA WorldCover. Reference: Hansen et al., *High-Resolution Global Maps of 21st-Century Forest Cover Change*, Science 2013 (University of Maryland / Google / USGS / NASA).
