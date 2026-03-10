# admet-multitask-demo 

Following on from my previous post about [using a a foundation model for ADMET modeling](https://github.com/DKchemistry/pretrain-vs-scratch-admet), I also wanted to explore multi-task learning. Like last time, this post was inspired by the commentary in a post by the OpenADMET team ([Lessons Learned from the OpenADMET ExpansionRx Blind Challenge](https://openadmet.ghost.io/lessons-learned-from-the-openadmet-expansionrx-blind-challenge/)). Multitask GNNs are stated as being the undisputed winners of this challenge and are well represented in the leaderboard. My PhD work primarily used single-task DNNs for docking score predictions and we were interested in expanding towards multi-task prediction for auxillary tasks that may be useful in downstream prospective VS. I worked on this for sometime but the set up I explored in my multi-task docking tasks weren't very fruitful at baseline, so integrating the multi-task approaches fell on the backburner (and our ability to learn the two tasks I had selected was limiting, though this may be an architectural issue). I still want to return to that project with some fresh ideas and I think some practical experience with multi-task modelling in this context may help :) 

The pebble team stated: 

> "Models were trained separately for related task groups, specifically: {LogD}, {KSOL}, {MLM, HLM}, {Caco-2 Papp, Caco-2 ER}, {MPPB, MGMB, MBPB}. For some task-group models, data from other properties was included as additional tasks to improve model performance. For example, many of the models were trained to predict LogD and pKa along with their target tasks."

Which gives us some ideas about what tasks may benefit from shared representations. 

In my previous demo, I grabbed data from the Therapeutic Data Commons, here we'll grab the [OpenADMET-ExpansionRx competition data](https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data) hosted on HuggingFace. We are going to grab the "ML-Ready" dataset which excludes out of range qualifiers (e.g. ">" values). The formerly blinded test is now available, so we'll grab that too.

We'll create a conda env for this, to avoid any dependency sort of issues in the chemprop env I've run into before: 

```sh
conda create -n admet-data -y -c conda-forge python=3.11 datasets pandas
```

Now, we have a script to get the Caco-2 data we want from the HuggingFace repo: 

```sh
conda activate admet-data
python scripts/00_make_dataset.py
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
README.md: 4.39kB [00:00, 9.65MB/s]
expansion_data_train.csv: 464kB [00:00, 10.2MB/s]
expansion_data_test.csv: 221kB [00:00, 11.4MB/s]
Generating train split: 100%|██████████████████████████████████████████████████████████████████████| 5326/5326 [00:00<00:00, 118290.80 examples/s]
Generating test split: 100%|███████████████████████████████████████████████████████████████████████| 2282/2282 [00:00<00:00, 103428.77 examples/s]
Wrote:
 - data/raw/caco2_train.csv: (5326, 3)
 - data/raw/caco2_test.csv:  (2282, 3)

Missing labels in TRAIN:
Caco-2 Permeability Papp A>B    3169
Caco-2 Permeability Efflux      3165

Missing labels in TEST:
Caco-2 Permeability Papp A>B    666
Caco-2 Permeability Efflux      666
```

Now we need to decide how we want to clean and split the data. I will be creating a validation set out of the training data, but it's worthwhile to pause and think about what the goal for the repo is (a quick comparison of single-task and multi-task approaches). As far as I understand, it is possible to train on data when only one of the two tasks has an appropiate label ("partial label" multi-task learning), but this going to add complexity in terms of both training (although chemprop v2 does support this) and evaluation. To avoid that for now, we will only keep data that has both labels. In terms of how we split out data, I previously used a scaffold split when trying CheMeleon. However, the more "realistic" perspective of how ADMET data is generated is temporally. This creates it's own set of issues and challenges (does later chemical space reflect earlier chemical space? what if there was a drastic scaffold hop?) which I would naively assume lead to "worse performance" compared to scaffold-splitting on train/val/test, but it may perform better on a real hold out test set. So with that in mind, our cleaning and splitting strategy will work like so: 

1. Drop rows where any of the 3 fields (SMILES, Papp, Efflux) are missing/blank/unparseable.
2. Deduplicate any SMILES, keeping first (data is likely already clean, but doesn't hurt to check.)
3. Assume data that is earlier in the CSV (initial rows) is temporally first, data that is later in the dataset is temporally last. Use an 80/20 train/valid split to make our training and validation data.


```sh
conda activate admet-data
python scripts/01_clean_and_split_temporal.py --val_frac 0.2
```

```sh
=== HF TRAIN raw -> cleaned paired ===
start rows: 5326
dropped (missing/unparseable in required cols): 3170
rows after dropna: 2156
dropped duplicates by SMILES: 0
rows after dedupe: 2156
final missing counts (should be 0):
SMILES                          0
Caco-2 Permeability Papp A>B    0
Caco-2 Permeability Efflux      0

=== HF TEST raw -> cleaned paired ===
start rows: 2282
dropped (missing/unparseable in required cols): 666
rows after dropna: 1616
dropped duplicates by SMILES: 0
rows after dedupe: 1616
final missing counts (should be 0):
SMILES                          0
Caco-2 Permeability Papp A>B    0
Caco-2 Permeability Efflux      0

=== Temporal split summary (based on row order) ===
train rows: 1725
val rows:   431
val is last 431 rows of cleaned train

Wrote:
  data/processed/caco2_train_paired_temporal.csv shape= (1725, 3)
  data/processed/caco2_val_paired_temporal.csv shape= (431, 3)
  data/processed/caco2_test_paired.csv shape= (1616, 3)
```
Now, we set up an ECFP4/RF model in both single-task and multi-task modes that we will run with five seeds to get some statistics. Sklearn takes care of most of the heavy lifting in terms of how the multi-task RF is handled, with us just supplying some initial hyperparameters. 

```sh
conda activate chemprop
python scripts/02_train_rf_baseline.py
```

We can use the summary data in `results/rf_val_metrics_summary.csv` and format a table. 

### Papp (Caco-2 Permeability Papp A>B)

| Model  | MAE (mean ± sd)     | RMSE (mean ± sd)    | R² (mean ± sd)      |
|--------|----------------------|----------------------|----------------------|
| Single | 5.9736 ± 0.0207      | 7.5032 ± 0.0146      | 0.2118 ± 0.0031      |
| Multi  | 5.8836 ± 0.0654      | 7.4140 ± 0.0459      | 0.2304 ± 0.0096      |

### Efflux (Caco-2 Permeability Efflux)

| Model  | MAE (mean ± sd)     | RMSE (mean ± sd)    | R² (mean ± sd)      |
|--------|----------------------|----------------------|----------------------|
| Single | 5.5644 ± 0.0269      | 12.2046 ± 0.0365     | 0.1894 ± 0.0048      |
| Multi  | 5.6612 ± 0.0180      | 12.9520 ± 0.0503     | 0.0870 ± 0.0071      |

We can also look at the deltas between approaches to make the single-task versus multi-task performance a bit more clear. 

### Effect of Multi-task RF vs Single-task RF

| Task   | Effect of Multi-task RF vs Single-task RF |
|--------|-------------------------------------------|
| Papp   | MAE ↓ 0.0900, RMSE ↓ 0.0892, R² ↑ 0.0186   |
| Efflux | MAE ↑ 0.0968, RMSE ↑ 0.7474, R² ↓ 0.1024   |

We see some minor performance gain in Papp by using a multi-task approach here, but a more considerable decrement in performance on Efflux. The standard deviations are quite small as well. Something pretty obvious I did not consider when making this demo is that such a situation like this could occur (one task improves, the other deteriorates) yet still be a boon for predictive ADME regardless, as I could use the multi-task model here to submit Papp results while sticking to the single-task model for Efflux results. There is, of course, more we could explore here, like digging into what failure modes RF is experiencing (comparing the ranges of predictions versus the ranges of the dataset, for example). We could also try to tune our RF model to try and improve these metrics based on such analyses, but for now, we will avoid those rabbit-holes and move on to the multi-task GNNs in chemprop and see how it compares "out of the box" :) 

Here, I want to avoid accidentally peaking at the test results and the CLI (as far as I understand) expects train/val/test, or trainval/test, where it will split trainval in someway for you. As we already made our "temporal" splits, neither option is great for me. One trick we can try here is simply passing the val data twice. 


```sh
export CUDA_VISIBLE_DEVICES="MIG-75c6e677-9d93-5114-97a4-dab667418517"
conda activate chemprop

chemprop train \
  --data-path data/processed/caco2_train_paired_temporal.csv \
            data/processed/caco2_val_paired_temporal.csv \
            data/processed/caco2_val_paired_temporal.csv \
  --task-type regression \
  --smiles-columns "SMILES" \
  --target-columns "Caco-2 Permeability Papp A>B" "Caco-2 Permeability Efflux" \
  --pytorch-seed 0 \
  --accelerator gpu \
  --devices 1 \
  --remove-checkpoints \
  --output-dir results/_scratch/chemprop_val_as_test/seed_0
  ```
This didn't seem to cause any issues in the output, so we will have to stay organized and keep in mind this data is ultimately *not* the real "test data". It might also make our lives easier as we can analyze the validation predictions like we did with the RF code. 

Now, we'll set a simple loop over the seeds and run both models (enjoy some coffee or tea in the meanwhile :))

```sh
conda activate chemprop
./scripts/03_train_chemprop_single.sh && ./scripts/04_train_chemprop_multi.sh
```

To keep myself sane, I will rename the usual outputs (e.g., `results/chemprop_multi/seed_0/model_0/test_predictions.csv`) to `val_predictions.csv` before we start looking at the data. 

```sh
for d in results/chemprop_multi/seed_*/model_0; do
  if [ -f "$d/test_predictions.csv" ]; then
    mv "$d/test_predictions.csv" "$d/val_predictions.csv"
  fi
done
```

```sh
for d in results/chemprop_single/*/seed_*/model_0; do
  if [ -f "$d/test_predictions.csv" ]; then
    mv "$d/test_predictions.csv" "$d/val_predictions.csv"
  fi
done
```

Now we can analyze our output metrics similar to before: 

```sh
python scripts/05_eval_chemprop_val.py
Wrote:
 - results/chemprop_val_metrics_by_seed.csv
 - results/chemprop_val_metrics_summary.csv
 ```

Format the table as before:

 ### Papp (Caco-2 Permeability Papp A>B)

| Model  | MAE (mean ± sd)     | RMSE (mean ± sd)    | R² (mean ± sd)      |
|--------|----------------------|----------------------|----------------------|
| Single | 5.7168 ± 0.1756      | 7.3348 ± 0.0359      | 0.2468 ± 0.0074      |
| Multi  | 5.6983 ± 0.1322      | 7.6789 ± 0.1683      | 0.1742 ± 0.0363      |

### Efflux (Caco-2 Permeability Efflux)

| Model  | MAE (mean ± sd)     | RMSE (mean ± sd)    | R² (mean ± sd)      |
|--------|----------------------|----------------------|----------------------|
| Single | 6.3030 ± 0.3295      | 12.5881 ± 0.1087     | 0.1376 ± 0.0149      |
| Multi  | 6.4079 ± 0.1279      | 12.7580 ± 0.1974     | 0.1141 ± 0.0273      |

And look at the deltas as before: 

### Effect of Multi-task Chemprop vs Single-task Chemprop

| Task   | Effect of Multi-task Chemprop vs Single-task Chemprop |
|--------|--------------------------------------------------------|
| Papp   | MAE ↓ 0.0185, RMSE ↑ 0.3441, R² ↓ 0.0726               |
| Efflux | MAE ↑ 0.1049, RMSE ↑ 0.1699, R² ↓ 0.0235               |

Here, contrary to my intuition, multitask is uniformly worse for Efflux and only improves MAE (slightly) for Papp. In terms of the leaderboards of this competition, [winners were picked on MAE for individual endpoints](https://openadmet.ghost.io/openadmet-expansionrx-blind-challenge/).

> "The endpoints will be judged individually by mean absolute error (MAE), while the overall leaderboard will be judged by the macro-averaged relative absolute error (MA-RAE). For endpoints that are not already on a log scale (e.g, LogD), they will be transformed to a log scale to minimize the impact of outliers on evaluation.  Relative absolute error (RAE) normalizes the MAE to the dynamic range of the test data, making the RAE comparable between endpoints, unlike MAE.

In that context, I would go with chemprop-multi-task for Papp (MAE = 5.6983), and chemprop-single-task for Efflux (MAE = 6.3030). But what about in comparison to the best performers on our "classical" RF baseline? A similar trend of RF-multi-task out performing for Papp (MAE = 5.8836) and RF-single-task out performing for Efflux (MAE = 5.5644). Contrary to my expectations, the RF-single-task approach for Efflux appears better than chemprop-single-task for Efflux. But this was likely a naive expectation to begin with for a few reasons. First, concluding the relative advantage of one approach over the other here without a statistical test is likely inappropriate. The most common statistical test I have seen in kind of work (from both the Novalix paper on the [prior OpenADMET challenge](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5c01982), as well as the [CheMeleon paper](https://arxiv.org/abs/2506.15792)) is the post-hoc Tukey Honestly Significant Difference (HSD), which we may want to implement. Second, even if the tests confirmed what we see from direct MAE comparisons, many leaderboard winners are reported as using multiple model types (pebble used an "Ensemble GNN" for example), as well as external data/proprietary data/HPO, which may ultimately lead to a stronger performance from GNN multi-task models compared to this very basic demo we have here. 

These two issues are related for where I'd like to go in this demo. So far, I have been using a temporal split of the data as I assume it to be "more realistic" for how the true test set is constructed. However, I'd also like to explore alternative modelling approaches from papers/posts I am reading to see if they improve performance. As I want to compare various approaches, I'd like to have reasonable statistical testing between approaches (there is a good paper on this subject [Practically significant method comparison protocols for machine learning in small molecule drug discovery](https://pubs.acs.org/doi/10.1021/acs.jcim.5c01609) by Ash et al). Following their conventions, a CV split of some sort seems ideal (e.g. 5x5). This seems challenging to do using a temporal split, as I have somewhat locked myself into a particular data distribution and can not easily avoid memorizing training data. It could be the case that using a temporal split and optimizing methods *could* lead to legitimate improved performance in either the leaderboard or real-world decision making data, but I find myself having less confindence in that style of data splitting. 

With that in mind, I'd like to shift to a 5x5 CV as described by Ash et al. 

First, we need to do the split itself. The following is going to do a 5x5 CV using scaffold splitting and trying to preserve relatively similar train/validation set sizes. 

(Side note: I recently watched a seminar from [Pat Walter's on building ML models from BindingDB data](https://www.youtube.com/watch?v=R97Qikb7_38&t=13s), wherein he mentioned in Q&A that he isn't a generally a fan of scaffold splitting, recommending those interested to read this blog post, ["Some Thoughts on Splitting Chemical Datasets"](https://practicalcheminformatics.blogspot.com/2024/11/some-thoughts-on-splitting-chemical.html) which also references this work ["Scaffold Splits Overestimate Virtual Screening Performance"](https://link.springer.com/chapter/10.1007/978-3-031-72359-9_5) by Guo et al. I still need to go through these posts/papers in more detail, but I'd like to briefly comment. The core argument seems to be that that scaffold splitting may seperate molecules into different scaffold clusters that are ultimately still quite similar, such as in Figure 1 of Guo et al, leading to a general overestimation of performance. Additionally, the relative performance of one modelling approach versus another, such as an RF vs GNN, also varied as a function of how the data was ultimately split. It's worthwhile to keep this in mind when we later reach any sort of comparison of modelling approaches.)

```sh
python scripts/07_make_scaffold_cv5x5.py
```

That is going to allow us to run the RF models: 

```sh
python scripts/08_train_rf_baseline_cv5x5.py
```

And our two shell scripts for the Chemprop CLI

```sh
scripts/09_train_chemprop_cv5x5_single.sh
```

```sh
scripts/10_train_chemprop_cv5x5_multi.sh
```

I could have set this up differently (I think), but the way I ran it in those shell scripts did not produce a predictions csv as before when I specified `--data-path` in the CLI and then renamed it. So we will run the predictions sequentially now:

```sh
scripts/11_predict_chemprop_cv5x5_all.sh
```

Prediction is very quick, though it is 75 predictions, so it will take a bit regardless!

I think it's possible to have predictions come out via the training CLI so we don't have to do this extra step, but this is learning-by-doing in action :) Next time, we will try to do that. 

We can evaluate the chemprop results next: 

```sh
python scripts/12_eval_chemprop_cv5x5.py
```
This will compute the chemprop performance metrics in a way that we can compare to the random forest metrics, giving us a csv like so: 

```
model_family,variant,task,cv_iter,fold,mae,rmse,r2,n_val
chemprop,multi,papp,0,0,6.5055,8.6319,0.3284,439
chemprop,multi,efflux,0,0,3.8981,8.9422,0.162378,439
chemprop,single_papp,papp,0,0,6.502,8.6969,0.318253,439
chemprop,single_efflux,efflux,0,0,3.7993,8.9794,0.155394,439
```
A couple things about how I chose to do this methods comparison: I am trying my best to follow along with "Practically Significant Method Comparison Protocols for Machine Learning in Small Molecule Drug Discovery". I am quite new to to statistical testing, so I am taking this slowly. 

We'll pick up in here: 

```sh
notebooks/5x5_cv_scaffold_baselines.ipynb
```

It is good that we kept track of both cv_iter and fold, because our evaluation produces 25 paired performance measurements per method (one MAE for each (cv_iter, fold) CV run). Since all methods are evaluated on the same CV runs, the measurements are paired by CV split.

As a global test, we run a repeated-measures ANOVA (rmANOVA) using Pingouin. This tests the null hypothesis that all methods have the same mean MAE on Papp under this 5×5 scaffold CV protocol. We obtain a very small p-value (p_unc = 1.36e−10; still significant under the sphericity-corrected value, p_GG_corr = 1e-6), so we reject the null and conclude that at least one method’s mean MAE differs.

```
| Source |       SS | DF |       MS |         F |        p_unc | p_GG_corr |      ng2 |      eps | sphericity |  W_spher |      p_spher |
| ------ | -------: | -: | -------: | --------: | -----------: | --------: | -------: | -------: | ---------- | -------: | -----------: |
| method | 2.028728 |  3 | 0.676243 | 23.130874 | 1.364919e-10 |  0.000001 | 0.124696 | 0.529715 | False      | 0.137252 | 1.446930e-08 |
| Error  | 2.104956 | 72 | 0.029235 |       NaN |          NaN |       NaN |      NaN |      NaN | NaN        |      NaN |          NaN |
```
To determine which methods differ, we follow up with a post-hoc Tukey HSD test (controlling family-wise error rate across all pairwise method comparisons). In the Tukey table, meandiff is the difference in mean MAE between the two methods (negative values indicate the second method has lower MAE, i.e., better performance when MAE is the metric), and reject=True indicates a statistically significant difference after correction:

```
Multiple Comparison of Means - Tukey HSD, FWER=0.05               
================================================================================
       group1               group2        meandiff p-adj   lower   upper  reject
--------------------------------------------------------------------------------
      CHEMPROP-multi CHEMPROP-single_papp   0.1729 0.3909 -0.1119  0.4577  False
      CHEMPROP-multi             RF-multi   -0.194 0.2887 -0.4789  0.0908  False
      CHEMPROP-multi       RF-single_papp  -0.1425   0.56 -0.4273  0.1423  False
CHEMPROP-single_papp             RF-multi  -0.3669 0.0059 -0.6517 -0.0821   True
CHEMPROP-single_papp       RF-single_papp  -0.3154 0.0239 -0.6002 -0.0306   True
            RF-multi       RF-single_papp   0.0515 0.9649 -0.2333  0.3363  False
--------------------------------------------------------------------------------
```


We can see from plotting the data that RF-multi is the "best" method for Papp, but statistically similar to RF-single_papp and CHEMPROP-multi, with only CHEMPROP-single_papp having worse MAE that is statistically significant. We can color code by a similar convention to Ash et al (blue = best, grey = statistically similar, red = statistically worse) (in the notebook). 

It's a similar story for efflux, but while RF-multi has the lowest MAE and rmANOVA suggests a global effect is present (p_GG_corr = 2.52e−4), the HSD statistical test does not imply that it is statistically distinct from the other methods (at alpha=0.05): 

```
| Source |       SS | DF |       MS |         F |    p_unc | p_GG_corr |      ng2 |      eps | sphericity |  W_spher |      p_spher |
| ------ | -------: | -: | -------: | --------: | -------: | --------: | -------: | -------: | ---------- | -------: | -----------: |
| method | 1.592925 |  3 | 0.530975 | 10.840375 | 0.000006 |  0.000252 | 0.044781 | 0.596322 | False      | 0.178621 | 2.323996e-07 |
| Error  | 3.526649 | 72 | 0.048981 |       NaN |      NaN |       NaN |      NaN |      NaN | NaN        |      NaN |          NaN |
```

```
Multiple Comparison of Means - Tukey HSD, FWER=0.05                
===================================================================================
        group1                 group2         meandiff p-adj   lower  upper  reject
-----------------------------------------------------------------------------------
        CHEMPROP-multi CHEMPROP-single_efflux   0.0238  0.999 -0.4162 0.4638  False
        CHEMPROP-multi               RF-multi  -0.2656 0.3959 -0.7055 0.1744  False
        CHEMPROP-multi       RF-single_efflux  -0.2076 0.6069 -0.6476 0.2323  False
CHEMPROP-single_efflux               RF-multi  -0.2894 0.3192 -0.7293 0.1506  False
CHEMPROP-single_efflux       RF-single_efflux  -0.2315 0.5177 -0.6714 0.2085  False
              RF-multi       RF-single_efflux   0.0579 0.9859  -0.382 0.4979  False
-----------------------------------------------------------------------------------
```

This is also visible in the color coded plot. 

So, in this limited demo, we do see RF-multitask modelling being better for at least one of the tasks (Papp) but not distinguishable for efflux. 

I'd like to see how HLM/MLM compare as a task affinity group. I think this repo has gotten a little messy (I don't think I will reuse the numbering scheme in the future). Let's start by refactoring `scripts/00_make_dataset.py` as `scripts/13_make_hlm_mlm_dataset.py` so that we can grab the HLM/MLM data as well. 


```sh
conda activate admet-data
python scripts/13_make_hlm_mlm_dataset.py
```

```sh
Wrote:
 - data/raw/hlm_mlm_train.csv: (5326, 3)
 - data/raw/hlm_mlm_test.csv:  (2282, 3)

Missing labels in TRAIN:
HLM CLint    1567
MLM CLint     804

Missing labels in TEST:
HLM CLint    1500
MLM CLint    1112
```

We will use our training data only, and repeat the 5x5 scaffold CV split. We'll still do the in-between step of making the temporal data split, as it was part of some of our dataset cleaning previously. We may return to it later. We'll refactor `scripts/01_clean_and_split_temporal.py` to take args we previously set as defaults. We will again stick to paired labels only.



```sh
python scripts/01_clean_and_split_temporal.py \
  --train_csv data/raw/hlm_mlm_train.csv \
  --test_csv  data/raw/hlm_mlm_test.csv \
  --out_prefix hlm_mlm \
  --y_cols "HLM CLint,MLM CLint"

=== HF TRAIN raw -> cleaned ===
start rows: 5326
dropped (missing/unparseable in required cols): 1586
rows after dropna: 3740
dropped duplicates by SMILES: 0
rows after dedupe: 3740
final missing counts (should be 0):
SMILES       0
HLM CLint    0
MLM CLint    0

=== HF TEST raw -> cleaned ===
start rows: 2282
dropped (missing/unparseable in required cols): 1749
rows after dropna: 533
dropped duplicates by SMILES: 0
rows after dedupe: 533
final missing counts (should be 0):
SMILES       0
HLM CLint    0
MLM CLint    0

=== Temporal split summary (based on row order) ===
train rows: 2992
val rows:   748
val is last 748 rows of cleaned train

Wrote:
  data/processed/hlm_mlm_train_paired_temporal.csv shape= (2992, 3)
  data/processed/hlm_mlm_val_paired_temporal.csv shape= (748, 3)
  data/processed/hlm_mlm_test_paired.csv shape= (533, 3)
```

Now we need to make our scaffold splits as before. Again, the code is refactored to take arguments.

```sh
conda activate chemprop
python scripts/07_make_scaffold_cv5x5.py \
  --train_csv data/processed/hlm_mlm_train_paired_temporal.csv \
  --val_csv   data/processed/hlm_mlm_val_paired_temporal.csv \
  --out_trainval data/processed/hlm_mlm_trainval_paired.csv \
  --out_splits  data/splits/hlm_mlm_scaffold_cv5x5.csv \
  --out_cv_root data/cv5x5_hlm_mlm

Wrote data/processed/hlm_mlm_trainval_paired.csv with 3740 rows
Wrote data/splits/hlm_mlm_scaffold_cv5x5.csv with 18700 assignments (should be N * 5)

==============================
Sanity check: scaffold CV 5x5
==============================
N molecules: 3740
Unique scaffolds: 1795
n_repeats: 5 n_folds: 5
Repeat 0: OK (each row_id appears exactly once)
Repeat 1: OK (each row_id appears exactly once)
Repeat 2: OK (each row_id appears exactly once)
Repeat 3: OK (each row_id appears exactly once)
Repeat 4: OK (each row_id appears exactly once)

Validation fold sizes (rows and % of dataset):
  repeat 0, fold 0: val_rows=748 (20.0%), val_scaffolds=334, scaffold_overlap_with_train=0
  repeat 0, fold 1: val_rows=749 (20.0%), val_scaffolds=407, scaffold_overlap_with_train=0
  repeat 0, fold 2: val_rows=747 (20.0%), val_scaffolds=351, scaffold_overlap_with_train=0
  repeat 0, fold 3: val_rows=749 (20.0%), val_scaffolds=363, scaffold_overlap_with_train=0
  repeat 0, fold 4: val_rows=747 (20.0%), val_scaffolds=340, scaffold_overlap_with_train=0
  repeat 1, fold 0: val_rows=748 (20.0%), val_scaffolds=402, scaffold_overlap_with_train=0
  repeat 1, fold 1: val_rows=747 (20.0%), val_scaffolds=387, scaffold_overlap_with_train=0
  repeat 1, fold 2: val_rows=749 (20.0%), val_scaffolds=328, scaffold_overlap_with_train=0
  repeat 1, fold 3: val_rows=747 (20.0%), val_scaffolds=320, scaffold_overlap_with_train=0
  repeat 1, fold 4: val_rows=749 (20.0%), val_scaffolds=358, scaffold_overlap_with_train=0
  repeat 2, fold 0: val_rows=746 (19.9%), val_scaffolds=350, scaffold_overlap_with_train=0
  repeat 2, fold 1: val_rows=745 (19.9%), val_scaffolds=392, scaffold_overlap_with_train=0
  repeat 2, fold 2: val_rows=758 (20.3%), val_scaffolds=291, scaffold_overlap_with_train=0
  repeat 2, fold 3: val_rows=746 (19.9%), val_scaffolds=371, scaffold_overlap_with_train=0
  repeat 2, fold 4: val_rows=745 (19.9%), val_scaffolds=391, scaffold_overlap_with_train=0
  repeat 3, fold 0: val_rows=747 (20.0%), val_scaffolds=373, scaffold_overlap_with_train=0
  repeat 3, fold 1: val_rows=747 (20.0%), val_scaffolds=336, scaffold_overlap_with_train=0
  repeat 3, fold 2: val_rows=747 (20.0%), val_scaffolds=354, scaffold_overlap_with_train=0
  repeat 3, fold 3: val_rows=750 (20.1%), val_scaffolds=392, scaffold_overlap_with_train=0
  repeat 3, fold 4: val_rows=749 (20.0%), val_scaffolds=340, scaffold_overlap_with_train=0
  repeat 4, fold 0: val_rows=749 (20.0%), val_scaffolds=303, scaffold_overlap_with_train=0
  repeat 4, fold 1: val_rows=748 (20.0%), val_scaffolds=422, scaffold_overlap_with_train=0
  repeat 4, fold 2: val_rows=748 (20.0%), val_scaffolds=373, scaffold_overlap_with_train=0
  repeat 4, fold 3: val_rows=747 (20.0%), val_scaffolds=296, scaffold_overlap_with_train=0
  repeat 4, fold 4: val_rows=748 (20.0%), val_scaffolds=401, scaffold_overlap_with_train=0

Repeat-to-repeat overlap check (are fold assignments changing?)
  fold 0: repeat0 vs repeat1 Jaccard=0.105
  fold 0: repeat0 vs repeat2 Jaccard=0.091
  fold 0: repeat0 vs repeat3 Jaccard=0.134
  fold 0: repeat0 vs repeat4 Jaccard=0.095
  fold 1: repeat0 vs repeat1 Jaccard=0.141
  fold 1: repeat0 vs repeat2 Jaccard=0.112
  fold 1: repeat0 vs repeat3 Jaccard=0.101
  fold 1: repeat0 vs repeat4 Jaccard=0.106
  fold 2: repeat0 vs repeat1 Jaccard=0.125
  fold 2: repeat0 vs repeat2 Jaccard=0.075
  fold 2: repeat0 vs repeat3 Jaccard=0.091
  fold 2: repeat0 vs repeat4 Jaccard=0.134
  fold 3: repeat0 vs repeat1 Jaccard=0.106
  fold 3: repeat0 vs repeat2 Jaccard=0.114
  fold 3: repeat0 vs repeat3 Jaccard=0.105
  fold 3: repeat0 vs repeat4 Jaccard=0.137
  fold 4: repeat0 vs repeat1 Jaccard=0.100
  fold 4: repeat0 vs repeat2 Jaccard=0.070
  fold 4: repeat0 vs repeat3 Jaccard=0.172
  fold 4: repeat0 vs repeat4 Jaccard=0.074
Wrote per-fold CV CSVs under: data/cv5x5_hlm_mlm
```

Now, we can run our baseline. Again, refactor and run. Apologies as the scripts look kind of messy here. I think this whole repo should be cleaned up before it's public facing. 

```sh
python scripts/08_train_rf_baseline_cv5x5.py \
  --trainval_csv data/processed/hlm_mlm_trainval_paired.csv \
  --splits_csv   data/splits/hlm_mlm_scaffold_cv5x5.csv \
  --out_dir      results/cv5x5_hlm_mlm/rf_ecfp4 \
  --y_cols       "HLM CLint,MLM CLint"
```

Hm. We can continue like this but I am not sure it's a good idea as even our prior results will likely change as I want refactor various elements, like log transforms. 

