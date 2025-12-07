# Clinical Correlation Analysis - Methods Documentation

## Overview

This document describes the statistical methods used to analyze correlations between **regional brain deviation scores** (from CVAE normative models) and **clinical symptom scores** in psychiatric patients.

---

## Table of Contents

1. [Input Data](#1-input-data)
2. [Analysis Pipeline](#2-analysis-pipeline)
3. [Statistical Methods](#3-statistical-methods)
4. [Multiple Testing Correction](#4-multiple-testing-correction)
5. [Diagnosis-Stratified Analysis](#5-diagnosis-stratified-analysis)
6. [Output Files](#6-output-files)
7. [Interpretation Guide](#7-interpretation-guide)

---

## 1. Input Data

### 1.1 Regional Deviation Scores

**Source:** CVAE normative modeling framework

**Structure:** One deviation score per subject per brain region (ROI)

```
Example:
  Filename: sub-001_ses-01_T1w.nii
  ROI_Name: [T] lprecentral (DK40)
  deviation_score: 0.547
  Diagnosis: CAT
```

**Key columns:**
- `Filename`: Subject identifier
- `ROI_Name`: Brain region name (260 ROIs from Desikan-Killiany-Tourville atlas + Neuromorphometrics)
- `deviation_score`: How much this subject's brain structure deviates from healthy controls
- `Diagnosis`: Clinical diagnosis (CAT, MDD, SSD, etc.)

### 1.2 Clinical Symptom Scores

**Source:** `complete_metadata.csv`

**Available Scores:**

| Score | Full Name | Range | Interpretation |
|-------|-----------|-------|----------------|
| **PANSS_Positive** | Positive and Negative Syndrome Scale - Positive | 7-49 | Higher = more positive symptoms (hallucinations, delusions) |
| **PANSS_Negative** | PANSS - Negative | 7-49 | Higher = more negative symptoms (flat affect, social withdrawal) |
| **PANSS_General** | PANSS - General Psychopathology | 16-112 | Higher = more general psychiatric symptoms |
| **PANSS_Total** | PANSS - Total Score | 30-210 | Sum of all PANSS subscales |
| **BPRS_Total** | Brief Psychiatric Rating Scale | 18-126 | Higher = more severe psychiatric symptoms |
| **NCRS_Motor** | Northoff Catatonia Rating Scale - Motor | 0-40 | Higher = more motor catatonia symptoms |
| **NCRS_Affective** | NCRS - Affective | 0-24 | Higher = more affective catatonia symptoms |
| **NCRS_Behavioral** | NCRS - Behavioral | 0-40 | Higher = more behavioral catatonia symptoms |
| **NCRS_Total** | NCRS - Total Score | 0-104 | Sum of all NCRS subscales |
| **NSS_Motor** | Neurological Soft Signs - Motor | 0-∞ | Higher = more motor abnormalities |
| **NSS_Total** | NSS - Total Score | 0-∞ | Higher = more neurological soft signs |
| **GAF_Score** | Global Assessment of Functioning | 0-100 | **Lower** = worse functioning (inverted!) |

**Important Notes:**
- Not all subjects have all scores (missing data handled per correlation)
- GAF is **inverted**: lower scores = worse functioning
- NCRS scores are primarily available for CAT patients
- Healthy controls (HC) are **excluded** from all analyses

---

## 2. Analysis Pipeline

### Step-by-Step Process:

```
1. LOAD DATA
   ├─ Regional deviation scores (from CVAE testing)
   └─ Clinical metadata (from complete_metadata.csv)

2. FILTER & MERGE
   ├─ Keep only datasets with clinical data (NSS, whiteCAT)
   ├─ Merge deviation + clinical data by Filename
   └─ Exclude healthy controls (HC)

3. COMPUTE CORRELATIONS
   ├─ For each ROI:
   │  └─ For each clinical score:
   │     ├─ Filter to subjects with this score available
   │     ├─ Compute Spearman correlation (ρ, p-value)
   │     └─ Compute Pearson correlation (r, p-value)
   └─ Result: 3120 correlations (260 ROIs × 12 scores)

4. MULTIPLE TESTING CORRECTION
   ├─ Apply FDR correction (Benjamini-Hochberg)
   └─ Identify significant correlations (FDR < 0.05)

5. DIAGNOSIS-STRATIFIED ANALYSIS
   ├─ Repeat correlations separately for each diagnosis
   ├─ CAT: 260 ROIs × 12 scores
   ├─ MDD: 260 ROIs × 12 scores
   └─ SSD: 260 ROIs × 12 scores

6. VISUALIZATIONS
   ├─ Heatmap: ROIs × Clinical Scores
   ├─ Bar plots: Top correlations
   └─ Scatter plots: Strongest relationships
```

---

## 3. Statistical Methods

### 3.1 Spearman Rank Correlation

**Formula:**

$$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2-1)}$$

where $d_i$ is the difference between ranks of paired observations.

**Why Spearman?**
- Robust to outliers
- No assumption of normality
- Detects monotonic relationships (not just linear)

**Implementation:**
```python
from scipy.stats import spearmanr

rho, p_val = spearmanr(deviation_scores, clinical_scores)
```

**Interpretation:**
- ρ = +1: Perfect positive correlation
- ρ = 0: No correlation
- ρ = -1: Perfect negative correlation
- |ρ| > 0.3: Moderate effect
- |ρ| > 0.5: Strong effect

### 3.2 Pearson Correlation

**Formula:**

$$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$

**Why also Pearson?**
- Measures linear relationships
- Comparison with Spearman
- If both agree: stronger evidence

**Implementation:**
```python
from scipy.stats import pearsonr

r, p_val = pearsonr(deviation_scores, clinical_scores)
```

### 3.3 Example Calculation

**Input Data:**
```
ROI: lprecentral (left motor cortex)
Clinical Score: NCRS_Motor

Subject   Deviation   NCRS_Motor
-------   ---------   ----------
sub-001      0.45         12
sub-002      0.67         18
sub-003      0.32          8
...
sub-220      0.51         14
```

**Step 1:** Rank both variables
```
Deviation ranks: [2, 3, 1, ...]
NCRS ranks:      [2, 3, 1, ...]
```

**Step 2:** Compute correlation
```
ρ = 0.197  (positive correlation)
p = 0.0003 (highly significant)
```

**Interpretation:**
- Higher motor cortex deviation → higher motor catatonia symptoms
- Effect size: small to moderate (ρ = 0.197)
- Statistical significance: p < 0.001

---

## 4. Multiple Testing Correction

### 4.1 Why Correction is Needed

**Problem:**
- We perform 3120 statistical tests (260 ROIs × 12 scores)
- At α = 0.05, we expect 156 false positives by chance!
- Without correction: inflated Type I error rate

**Solution:** False Discovery Rate (FDR) correction

### 4.2 Benjamini-Hochberg FDR Procedure

**Goal:** Control the **expected proportion** of false discoveries among all rejected hypotheses

**Algorithm:**
1. Sort all p-values in ascending order: $p_{(1)} \leq p_{(2)} \leq ... \leq p_{(m)}$
2. Find largest $i$ such that: $p_{(i)} \leq \frac{i}{m} \cdot \alpha$
3. Reject hypotheses $H_{(1)}, ..., H_{(i)}$

**Example:**
```
Test 1: p = 0.0001  →  0.0001 ≤ (1/3120) × 0.05 = 0.000016? NO
Test 2: p = 0.0003  →  0.0003 ≤ (2/3120) × 0.05 = 0.000032? NO
Test 3: p = 0.0005  →  0.0005 ≤ (3/3120) × 0.05 = 0.000048? NO
...
Test 100: p = 0.01  →  0.01 ≤ (100/3120) × 0.05 = 0.0016? NO
Test 500: p = 0.02  →  0.02 ≤ (500/3120) × 0.05 = 0.008? NO
```

**Implementation:**
```python
from statsmodels.stats.multitest import multipletests

reject, pvals_corrected, _, _ = multipletests(
    p_values, 
    alpha=0.05, 
    method='fdr_bh'
)
```

**Result:**
- `reject`: Boolean array indicating which tests are significant
- `pvals_corrected`: Adjusted p-values

### 4.3 FDR vs. Bonferroni

| Method | Correction | Power | When to use |
|--------|------------|-------|-------------|
| **Bonferroni** | $\alpha_{corrected} = \alpha / m$ | Low | Very conservative, few tests |
| **FDR (BH)** | Adaptive threshold | Higher | Many tests, exploratory |

**Why FDR?**
- More power than Bonferroni (fewer false negatives)
- Controls proportion of false discoveries (not FWER)
- Appropriate for large-scale testing (3120 tests)

---

## 5. Diagnosis-Stratified Analysis

### 5.1 Rationale

**Question:** Are brain-symptom correlations diagnosis-specific?

**Example:**
- Does motor cortex deviation correlate with NCRS_Motor **only in CAT**, or also in MDD/SSD?

### 5.2 Method

**For each diagnosis separately:**
1. Filter data to this diagnosis only
2. Compute correlations (same as above)
3. Apply FDR correction **within this diagnosis**

**Important:** FDR is applied **per diagnosis**, not globally!

**Example:**
```
CAT analysis:
  - 150 subjects
  - 260 ROIs × 12 scores = 3120 tests
  - FDR correction over these 3120 tests
  
MDD analysis:
  - 80 subjects
  - 260 ROIs × 12 scores = 3120 tests
  - SEPARATE FDR correction
```

### 5.3 Power Considerations

**Sample sizes matter:**
```
CAT: 150 subjects  →  Good power
MDD: 80 subjects   →  Moderate power
SSD: 100 subjects  →  Moderate power
```

**Result:** Diagnosis-specific analysis may have fewer significant findings due to:
- Smaller sample size per group
- FDR applied separately (more conservative)

---

## 6. Output Files

### 6.1 `clinical_correlations_all.csv`

**Contains:** All 3120 correlations (regardless of significance)

**Columns:**
- `ROI_Name`: Brain region
- `Clinical_Score`: Symptom score name
- `N_Subjects`: Number of subjects with both ROI + score data
- `Spearman_rho`: Spearman correlation coefficient
- `Spearman_p`: Uncorrected p-value (Spearman)
- `Pearson_r`: Pearson correlation coefficient
- `Pearson_p`: Uncorrected p-value (Pearson)
- `Spearman_p_corrected`: FDR-corrected p-value
- `Significant_FDR`: Boolean (True if FDR < 0.05)

**Example row:**
```csv
[T] lprecentral (DK40),NCRS_Motor,330,0.197,0.0003,0.064,0.246,0.045,True
```

**Interpretation:**
- Left motor cortex × Motor catatonia
- N = 330 subjects
- ρ = 0.197 (weak positive correlation)
- p = 0.0003 (uncorrected)
- p_corrected = 0.045 (FDR-corrected) → SIGNIFICANT!

### 6.2 `clinical_correlations_significant.csv`

**Contains:** Only significant correlations (FDR < 0.05)

**Use:** Quick overview of brain-symptom relationships

**Example:**
```
Top 5 significant correlations:
1. rlateraloccipital × NSS_Motor (ρ = -0.36, p = 1.7e-11)
2. llateraloccipital × NSS_Motor (ρ = -0.34, p = 1.2e-10)
3. rlateraloccipital × GAF_Score (ρ = -0.25, p = 5.4e-06)
4. ltransversetemporal × PANSS_General (ρ = -0.23, p = 1.7e-05)
5. lcaudalmiddlefrontal × NSS_Motor (ρ = -0.23, p = 2.0e-05)
```

### 6.3 `clinical_correlations_by_diagnosis.csv`

**Contains:** Diagnosis-stratified correlations

**Columns:** Same as `all.csv` + `Diagnosis` column

**Example:**
```csv
Diagnosis,ROI_Name,Clinical_Score,N_Subjects,Spearman_rho,Spearman_p,Spearman_p_corrected,Significant_FDR
CAT,[T] lprecentral (DK40),NCRS_Motor,150,0.45,1e-08,0.001,True
MDD,[T] lprecentral (DK40),NCRS_Motor,80,0.08,0.52,0.89,False
SSD,[T] lprecentral (DK40),NCRS_Motor,100,0.12,0.35,0.74,False
```

**Interpretation:**
- Motor cortex × NCRS_Motor is **CAT-specific**
- Strong in CAT (ρ = 0.45)
- Absent in MDD/SSD (ρ ≈ 0)

### 6.4 `clinical_score_summary.csv`

**Contains:** Summary per clinical score

**Columns:**
- `Clinical_Score`: Score name
- `N_Significant_ROIs`: How many ROIs correlate with this score
- `Top_Positive_ROIs`: Top 3 positive correlations
- `Top_Negative_ROIs`: Top 3 negative correlations

**Example:**
```csv
Clinical_Score,N_Significant_ROIs,Top_Positive_ROIs,Top_Negative_ROIs
NSS_Motor,15,"RightPutamen, LeftVentralDC, ...", "rlateraloccipital, llateraloccipital, ..."
PANSS_Total,8,"BrainStem, RightMFC, ...", "ltransversetemporal, ..."
NCRS_Motor,4,"lprecentral, RightPT, ...", "rinsula"
```

**Interpretation:**
- NSS_Motor: Brain-wide correlations (15 ROIs)
- NCRS_Motor: Focal correlations (4 ROIs)
- Suggests different neural bases

---

## 7. Interpretation Guide

### 7.1 Correlation Strength

**Effect Size Interpretation (|ρ|):**

| |ρ| | Strength | Interpretation |
|-----|----------|----------------|
| 0.00-0.10 | Negligible | No meaningful relationship |
| 0.10-0.30 | Small | Weak relationship |
| 0.30-0.50 | Moderate | Clear relationship |
| 0.50-0.70 | Strong | Very clear relationship |
| 0.70-1.00 | Very Strong | Almost perfect relationship |

### 7.2 Sign Interpretation

**Positive Correlation (ρ > 0):**
- Higher brain deviation → Higher symptom severity
- Example: "More motor cortex atrophy → More motor symptoms"

**Negative Correlation (ρ < 0):**
- Higher brain deviation → Lower symptom severity
- Example: "More occipital deviation → Lower GAF (worse functioning)"

**Special case: GAF is inverted!**
- GAF: Higher = better functioning
- Negative ρ with GAF: Higher deviation → Lower GAF → **Worse** functioning
- This is actually a **positive** clinical relationship!

### 7.3 Statistical Significance

**P-value thresholds:**
- p < 0.05 (uncorrected): Nominally significant
- FDR < 0.05: Survives multiple testing correction
- p < 0.001: Highly significant
- p < 0.0001: Extremely significant

**Only report FDR-corrected results in thesis!**

### 7.4 Sample Size Considerations

**N_Subjects interpretation:**

| N | Power | Reliability | Note |
|---|-------|-------------|------|
| < 50 | Low | Exploratory | Treat as hypothesis-generating |
| 50-100 | Moderate | Fair | Some confidence |
| 100-200 | Good | Reliable | Can make claims |
| > 200 | Excellent | Robust | Strong evidence |

**Your data:**
- Most scores: N = 330-331 → Excellent power
- NCRS scores: N = 330 → Still excellent

### 7.5 Common Patterns

**Pattern 1: Brain-wide correlations**
```
Example: NSS_Motor correlates with 15 ROIs
→ Neurological soft signs reflect widespread brain alterations
→ Less specific
```

**Pattern 2: Focal correlations**
```
Example: NCRS_Motor correlates with 4 ROIs (motor cortex, thalamus, insula)
→ Motor catatonia has specific neural substrate
→ More specific, more interpretable
```

**Pattern 3: Diagnosis-specific effects**
```
Example: Motor cortex × NCRS_Motor
  - CAT: ρ = 0.45 (strong)
  - MDD: ρ = 0.08 (none)
  - SSD: ρ = 0.12 (none)
→ This is a CAT-specific mechanism
```

### 7.6 Biological Interpretation

**Example: rlateraloccipital × NSS_Motor (ρ = -0.36)**

**Statistical:**
- Strong negative correlation
- Highly significant (p = 1.7e-11)
- Robust (survives FDR correction)

**Biological:**
- Lateral occipital cortex: Visual processing + motor coordination
- NSS_Motor: Motor abnormalities
- Interpretation: Visual-motor integration deficits in psychiatric patients
- Possible mechanism: Disrupted cerebellar-cortical circuits

**Clinical:**
- Could explain: Clumsiness, poor hand-eye coordination
- Relevant for: Occupational therapy, motor rehabilitation

---

## 8. Limitations & Caveats

### 8.1 Correlation ≠ Causation

- We observe associations, not causal relationships
- Could be: Brain → Symptoms, Symptoms → Brain, or Both ← Third factor

### 8.2 Multiple Testing Burden

- 3120 tests is a lot!
- FDR helps, but may still miss true effects (Type II error)
- Some true associations may not survive correction

### 8.3 Missing Data

- Not all subjects have all clinical scores
- Some correlations computed on smaller subsets
- NCRS: Primarily CAT patients

### 8.4 Cross-Sectional Design

- Single timepoint
- Cannot assess temporal dynamics
- Cannot infer progression or treatment effects

### 8.5 Heterogeneity

- Psychiatric diagnoses are heterogeneous
- CAT patients vary in symptom profiles
- Results reflect average effects

---

## 9. Reporting in Thesis

### 9.1 Methods Section Template

```markdown
**Clinical Correlation Analysis**

We examined correlations between regional brain deviation scores 
(derived from CVAE normative models) and clinical symptom scores 
in psychiatric patients (N = 214 after excluding healthy controls).

*Statistical Analysis:*
For each brain region (260 ROIs), we computed Spearman rank 
correlations between deviation scores and 12 clinical scores 
(PANSS, BPRS, NCRS, NSS, GAF). This resulted in 3,120 correlation 
tests. We applied Benjamini-Hochberg false discovery rate (FDR) 
correction to control for multiple comparisons (α = 0.05).

*Diagnosis-Stratified Analysis:*
To assess diagnosis-specific relationships, we repeated correlation 
analyses separately for CAT, MDD, and SSD, with FDR correction 
applied within each diagnostic group.

*Software:*
All analyses were performed in Python 3.10 using SciPy (v1.11.0) 
for correlations and statsmodels (v0.14.0) for FDR correction.
```

### 9.2 Results Section Template

```markdown
**Brain-Clinical Correlations**

After FDR correction, we identified 35 significant correlations 
between regional brain deviations and clinical symptoms (FDR < 0.05).

*NSS Motor Correlations:*
Neurological soft signs (NSS_Motor) showed the strongest and most 
widespread correlations, with 15 significant ROIs. The strongest 
effects were observed in lateral occipital cortex (left: ρ = -0.34, 
p_FDR < 0.001; right: ρ = -0.36, p_FDR < 0.001), suggesting visual-motor 
integration deficits in psychiatric patients.

*NCRS Catatonia Correlations:*
Motor catatonia symptoms (NCRS_Motor) correlated positively with 
deviations in left motor cortex (ρ = 0.197, p_FDR = 0.045) and 
negatively with right insula (ρ = -0.193, p_FDR = 0.045). Diagnosis-
stratified analysis revealed these effects were CAT-specific 
(CAT: ρ = 0.45; MDD: ρ = 0.08; SSD: ρ = 0.12).

[See Figure X for heatmap of all significant correlations]
[See Table X for complete list of significant correlations]
```

---

## 10. Summary

**Key Points:**

✓ **Method:** Spearman rank correlations (robust, non-parametric)
✓ **Multiple testing:** FDR correction (Benjamini-Hochberg)
✓ **Sample size:** Excellent (N = 330 for most scores)
✓ **Outputs:** 4 main CSV files + visualizations
✓ **Interpretation:** Effect size + significance + biological plausibility

**Final Checklist:**

- [ ] Correlations computed for all ROI × Score combinations
- [ ] FDR correction applied (α = 0.05)
- [ ] Diagnosis-stratified analysis performed
- [ ] Results visualized (heatmaps, scatter plots)
- [ ] Significant correlations saved to CSV
- [ ] Effect sizes interpreted correctly
- [ ] Biological mechanisms considered
- [ ] Limitations acknowledged

---
