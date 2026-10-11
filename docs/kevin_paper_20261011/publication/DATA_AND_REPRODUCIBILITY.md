# Data, source licenses and reproducibility

This export reports a research experiment, not a clinically ready system. The
legacy four labels are not verified CDR stages. Legacy person/augmentation-family
identifiers are unavailable, and the reused 100-image comparison cannot establish
clinical superiority over neuroradiologists. The clinical AD/control task is
separate and showed weak generalization with tiny samples.

The legacy dataset is [Singhal's augmented Kaggle derivative](https://www.kaggle.com/datasets/aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented).
Local provenance records Apache 2.0 for that derivative, LGPL 3.0 for its declared
upstream, and unresolved original-source license compatibility. Existing public
benchmark JPEGs support the requested Grad-CAM panels; this does not resolve the
source chain. No newly acquired clinical MRI is redistributed here.

The acquired clinical sources were [Besga, Zenodo 3935636](https://doi.org/10.5281/zenodo.3935636)
([associated article](https://pmc.ncbi.nlm.nih.gov/articles/PMC7658647/); recorded
CC BY 4.0), and [OpenNeuro ds007561, version 1.0.0](https://openneuro.org/datasets/ds007561/versions/1.0.0)
(DOI 10.18112/openneuro.ds007561.v1.0.0; recorded CC0). Besga acquisition included
78 source participants, of whom 54 AD/control participants entered the primary
clinical task; 24 bipolar participants were excluded. OpenNeuro included 20 source
participants: 15 binary evaluable and 5 MCI prediction-only. Nine slices per person
do not increase the independent sample count. Official raw releases from the ten
recommended sources were not newly acquired.

Source/version/access evidence is in [the dataset registry](datasets/dataset_registry.json)
and [recommendation crosswalk](datasets/recommended_dataset_crosswalk.json).
This export reports recorded license metadata, not a new license determination.
Raw acquired MRI, participant tables, clinical subject prediction rows, private
correspondence, SSH/account/resource metadata, logs and local ZIPs remain local.
Some source references consequently describe private artifacts rather than files
distributed here; omitted navigation points to this explanation.

The three legacy prepared manifests are under `prepared/`; their paths refer to
the existing repository's legacy image files. The original clinical manifest
remains local because it joins source participants and diagnoses to slices.
Its hash, aggregate split counts and preprocessing code are retained. Reproducing
clinical processing requires obtaining the cited releases and rebuilding the
documented private manifest. Public aggregate JSON exports remove only subject
records/private operational fields and normalize paths. They are not substitutes
for exact frozen-input files when running hash-guarded analysis scripts.

The six original trainer/evaluator/analysis source families are preserved in their
repository locations, identified by STAGING_ALLOWLIST.txt and the source hashes.
These are research runners; some frozen orchestrators retain their original
container layout. The public report can be read without that runtime. The six
research checkpoints are separately retained with Git LFS under
`models/kevin_20261011/`; the production baseline is unchanged.

To rebuild this curated export after all diagnostics finish, from the repository:

```sh
python3 docs/kevin_paper_20261011/publication/export_publication.py --require-complete
```

The generator does not perform inference, upload data, change original evidence,
stage Git files or provision resources. It validates source stability, public
links and bounded private-content checks. A clean checkout can read all exported
reports; regeneration additionally requires the locally preserved frozen results.
