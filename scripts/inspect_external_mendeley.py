"""Inspect the locally acquired CC BY 4.0 cohort without publishing DICOM identifiers."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import pydicom

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/external/mendeley_kcjt4v658x_v2'
OUT = ROOT/'docs/final_review_20260920/external_acquisition.json'


def main():
    groups = defaultdict(set)
    sequences, shapes, modality = Counter(), Counter(), Counter()
    identifiers_present, errors = Counter(), []
    count = 0
    for path in sorted((DATA/'extracted').rglob('*.dcm')):
        relative = path.relative_to(DATA/'extracted')
        label, folder = relative.parts[1:3]
        groups[label].add(folder)
        try:
            ds = pydicom.dcmread(path)
            pixels = ds.pixel_array
            count += 1
            shapes[str(pixels.shape)] += 1
            modality[str(ds.get('Modality', 'missing'))] += 1
            sequences[str(ds.get('SeriesDescription', 'missing'))] += 1
            for key in ['PatientName', 'PatientID', 'PatientBirthDate', 'AccessionNumber', 'InstitutionName']:
                identifiers_present[key] += bool(str(ds.get(key, '')).strip())
        except Exception as exc:
            errors.append({'path': str(relative), 'type': type(exc).__name__})
    result = {
        'dataset_doi': '10.17632/kcjt4v658x.2',
        'source_url': 'https://data.mendeley.com/datasets/kcjt4v658x/2',
        'download_url': 'https://data.mendeley.com/public-api/zip/kcjt4v658x/download/2',
        'retrieved_utc_date': '2026-09-20', 'license': 'CC BY 4.0',
        'attribution': "Fathi, Sina; Ahmadi, Ali; Almasi-Dooghaee, Mostafa; Sadegh, Melika (2023), Alzheimer's disease MRI images, Mendeley Data, V2, doi:10.17632/kcjt4v658x.2",
        'archive_bytes': (DATA/'source.zip').stat().st_size,
        'archive_sha256': hashlib.sha256((DATA/'source.zip').read_bytes()).hexdigest(),
        'dicom_files_decoded': count, 'dicom_errors': errors,
        'jpg_thumbnails': len(list((DATA/'extracted').rglob('*.jpg'))),
        'subject_folder_counts': {k:len(v) for k,v in groups.items()},
        'modality_counts': dict(modality), 'pixel_shapes': dict(shapes),
        'series_description_counts': dict(sequences),
        'nonempty_identifier_field_counts': dict(identifiers_present),
        'stage_mapping': 'Not established: AD/MCI/NC are not the four training stage labels.',
        'independence': 'Separately described hospital source; training-patient independence not established.',
        'external_performance': 'Not evaluated; raw files kept local and excluded from Git.',
    }
    OUT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'dicom_errors'}, indent=2))
    print('DICOM decoding failures:', len(errors))


if __name__ == '__main__':
    main()
