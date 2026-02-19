# Must-Know Caveats

This page documents compatibility issues, version constraints, and known workarounds for the visualization package.

## Python Version: Must Be 3.12

FiftyOne 1.13.0 officially supports Python 3.9–3.12. It does **not** support Python 3.13. The project pins `python >=3.12,<3.13` in `pyproject.toml` to ensure compatibility.

## FiftyOne Nested Union Bug (v1.13.0)

FiftyOne 1.13.0 has a bug in `fiftyone/server/aggregations.py` line 121: a nested `Union[AggregateResult, AggregationQueryTimeout]` where `AggregateResult` is itself a union. This crashes strawberry-graphql's schema converter.

**Status**: Fix is in PR #7021, targeting v1.13.1 (not yet released).

**Workaround**: Manually patch the installed file at `.pixi/envs/dev/lib/python3.12/site-packages/fiftyone/server/aggregations.py` — flatten the nested union to list all 6 aggregation types + `AggregationQueryTimeout` directly.

**Note**: This patch is lost on `pixi install`. Re-apply after reinstalling dependencies.

## strawberry-graphql Pin

strawberry-graphql 0.292.0 introduced a breaking change (removed `graphiql` kwarg). FiftyOne is incompatible with this version. The project pins `strawberry-graphql>=0.262.4,<0.292` in pixi dev dependencies.

## FiftyOne Is a Dev Dependency

FiftyOne is declared in `[dependency-groups] dev`, not in runtime dependencies. It is not needed at training time. The `[project.optional-dependencies] viz` group contains only lightweight extras (scikit-learn, umap-learn, rerun-sdk).

Import paths in `fiftyone_loader.py` are **not** lazy-guarded because the module is only used from the browse script or interactive sessions where fiftyone is already installed.

## Large Dataset Subsampling

The full dataset is ~300GB. The loader takes every Nth frame per episode (`subsample_rate`, default 10) to keep FiftyOne datasets manageable. For a dataset with 26,550 frames, `subsample_rate=10` produces ~2,655 samples.

Increase `subsample_rate` for faster loading on very large datasets. Decrease it for more detailed inspection.

## JPEG Idempotency

Frame images are exported as JPEG files. The save is idempotent: if a file already exists with matching width and height, it is skipped. This means re-running the loader on the same dataset is fast after the first run.

If you change image preprocessing or the source dataset changes, delete the output directory to force re-export.

## Persistent vs Non-Persistent Datasets

By default, FiftyOne datasets are non-persistent (deleted when the Python process exits). Use `--persistent` to store datasets in FiftyOne's MongoDB backend so they survive across sessions.

Persistent datasets can be listed and loaded later:

```python
import fiftyone as fo
print(fo.list_datasets())
dataset = fo.load_dataset("lerobot_pusht")
```
