# Input catalogues (not distributed)

The example configs in `configs/` and the test and benchmark scripts expect two ASCII files here:

| file | columns | used as |
|---|---|---|
| `data.data` | `x y z w`, whitespace separated (weights in column 4) | data catalogue |
| `random.txt` | `x y z` | random catalogue |

Coordinates are Cartesian, in the same units as `min_sep`/`max_sep`.

The configs assume a periodic box of side 1000 (`metric: Periodic`, `period: 1000`). For other
files, adjust `file_name`, `rand_file_name`, the `*_col` keys and `period` in the configs, or
override them on the command line:

```bash
python run_corr.py configs/p3pcfconfig.yaml backend=cuda file_name=my_data.txt rand_file_name=my_rand.txt
```

The analysis in `report/` used 10,000 data points and 500,000 randoms.
