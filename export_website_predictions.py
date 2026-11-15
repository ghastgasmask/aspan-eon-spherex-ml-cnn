"""Generate the static CNN predictions used by the GitHub Pages viewer."""
import argparse
import csv
import io
import json
import re
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from zipfile import ZipFile

import torch

from predict import FitsClassifier


SITE_ALIASES = {
    "r-dra": "r-draconis",
    "rw-aur": "rw-aurigae",
    "wise-0855": "wise-0855-0714",
}


def slugify(name):
    name = name.lower().replace("−", "-").replace("'", "")
    slug = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    return SITE_ALIASES.get(slug, slug)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("site", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(2)
    classifier = FitsClassifier(args.checkpoint)
    run = json.loads(args.checkpoint.with_name("training.json").read_text())
    splits = {
        slugify(name): split
        for split in ("train", "validation", "test")
        for name in run[f"{split}_objects"]
    }
    site_index = json.loads((args.site / "data/index.json").read_text(encoding="utf-8"))
    site_slugs = {row["slug"] for row in site_index["objects"]}
    catalog_path, = sorted(args.dataset.glob("catalogs*.zip"))

    with ZipFile(catalog_path) as catalog:
        with catalog.open("catalogs/metadata.csv") as table:
            rows = list(csv.DictReader(io.TextIOWrapper(table, encoding="utf-8-sig")))

    groups = defaultdict(list)
    for row in rows:
        slug = slugify(row["name"])
        if slug in site_slugs:
            groups[slug].append(row["file"])

    predictions = {}
    with ExitStack() as stack:
        locations = {}
        for path in sorted(args.dataset.glob("fits*.zip")):
            archive = stack.enter_context(ZipFile(path))
            for name in archive.namelist():
                if not name.lower().endswith(".fits"):
                    continue
                if name in locations:
                    raise ValueError(f"Duplicate archive path: {name}")
                locations[name] = archive

        for slug, paths in sorted(groups.items()):
            observations = [io.BytesIO(locations[path].read(path)) for path in paths]
            predictions[slug] = classifier.predict(observations)
            predictions[slug]["split"] = splits[slug]
            if len(predictions) % 20 == 0:
                print(f"Classified {len(predictions)} targets", flush=True)

    if not predictions:
        raise ValueError("No matches between the website and the FITS catalog")
    output = args.site / "data/cnn-predictions.json"
    output.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(predictions)} predictions to {output}")
    print("Targets without predictions:", ", ".join(sorted(site_slugs - predictions.keys())) or "none")


if __name__ == "__main__":
    main()
