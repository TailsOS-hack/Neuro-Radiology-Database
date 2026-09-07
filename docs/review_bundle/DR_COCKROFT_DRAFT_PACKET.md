# Draft review packet — Dr. Cockroft

**Prepared September 7, 2026 · Internal mentor review · Not sent**

Dear Dr. Cockroft,

I would appreciate your review of the dementia component of our leakage-audited public brain MRI benchmark. The manuscript retains the tumor specialist and single-head comparator to preserve the original study design. The new supplement adds reproducible Grad-CAM examples and a quantitative check of how the specialist scores respond to image masking.

**What the existing evidence supports.** The accepted exact-deduplicated dementia specialist achieved 0.9991 internal strict-test accuracy (8,806 images); the conservative dHash sensitivity result was 0.9975 (8,881 images). These are public-image benchmark results, not clinical dementia diagnosis or patient-level validation. The integrated single 8-class model remains slightly stronger than the hierarchical model in the accepted primary run (0.9972 vs 0.9963 accuracy).

**What the new explainability analysis adds.** All 128 sampled images per specialist produced valid CAMs. In the dementia sample, masking the highest-CAM 20% of pixels reduced predicted-class confidence by a mean of 0.4453, versus 0.6901 for random masking; the paired difference was −0.2448. The tumor comparison was also negative (−0.3780). We therefore do not claim that highlighted regions are preferentially important under this test. Spatially different masks and out-of-distribution perturbations limit interpretation. The gallery includes seven dementia and eight tumor low-confidence/error examples, selected by a fixed rule before inspecting maps. There is no expert anatomical validation.

**Claim boundaries retained.** The dementia source is an augmented public dataset; patient identifiers and acquisition metadata are unavailable, so patient-level leakage remains possible. Dataset-source shortcuts remain plausible. Confidence is not calibrated clinical probability (integrated-model ECE: 0.1327 single-head; 0.1526 hierarchical). No independent external cohort has been evaluated. No retraining was needed: the accepted checkpoints were recovered and checked against archived probabilities; this analysis did not tune models on test images.

**Three decisions for review**

1. Should the negative masking comparison and Grad-CAM galleries remain a transparent supplementary analysis, rather than a central interpretability contribution?
2. Is a dementia-focused framing appropriate while retaining the combined benchmark design, or should the broader title and scope remain?
3. What independent cohort and patient-level metadata would be required for a subsequent validation study, and which venue best fits the present internal benchmark?

**Suggested reading order**

- [Full manuscript](../MANUSCRIPT_FULL_DRAFT.md): new Methods §2.9 and Results §3.6; existing limitations retained.
- [Grad-CAM gallery](explainability/index.html): readable individual panels; Figures 8a–8b are supplement candidates.
- [Quantitative summary](explainability/SUMMARY.md) and [protocol/reproduction notes](README.md).
- [Dataset provenance](../DATASET_PROVENANCE.md), [accepted results tables](../PUBLICATION_RESULTS_TABLES.md), and [figure captions](../FIGURE_CAPTIONS.md).

Before submission, authorship/affiliations, venue formatting, formal references, dataset attribution and image-use terms, and institutional ethics wording still require completion. This packet is ready for scientific feedback, not submission or deployment claims.
