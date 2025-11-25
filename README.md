# quant_jegadeesh_titman_rep
- Paper representation "Returns to Buying Winners and Selling  Losers: Implications for  Stock Market Efficiency" by NARASIMHAN JEGADEESH and SHERIDAN TITMAN

# data
- python 코드가 있는 곳에 'data' 폴더를 만들고, 그 안에 CRSP 파일을 넣어두세요
- load_and_prepare_data 함수에서 파일경로와 파일명에 맞게 data_dir, file_name 을 수정하세요

# run
$python run.py

# result

| J / K | My Code AVG (%) | Paper AVG (%) | My Code T-stat | Paper T-stat | My Code STD (%) |
|---|---|---|---|---|---|
| 3 / 3 | 0.41 | 0.48 | 1.80 | 2.08 | 3.92 |
| 3 / 6 | 0.60 | 0.64 | 2.98 | 3.01 | 3.44 |
| 3 / 9 | 0.67 | 0.70 | 3.81 | 3.59 | 3.01 |
| 3 / 12 | 0.72 | 0.77 | 4.63 | 4.19 | 2.67 |
| 6 / 3 | 0.71 | 0.77 | 2.55 | 2.76 | 4.75 |
| 6 / 6 | 0.92 | 0.95 | 3.69 | 3.73 | 4.26 |
| 6 / 9 | 0.99 | 1.02 | 4.52 | 4.24 | 3.75 |
| 6 / 12 | 0.85 | 0.93 | 4.13 | 4.02 | 3.51 |
| 9 / 3 | 0.89 | 1.04 | 3.03 | 3.74 | 5.00 |
| 9 / 6 | 1.08 | 1.18 | 4.13 | 4.54 | 4.48 |
| 9 / 9 | 0.99 | 1.11 | 4.04 | 4.41 | 4.16 |
| 9 / 12 | 0.79 | 0.86 | 3.45 | 3.53 | 3.92 |
| 12 / 3 | 1.00 | 1.31 | 3.46 | 4.43 | 4.93 |
| 12 / 6 | 1.00 | 1.22 | 3.68 | 4.49 | 4.62 |
| 12 / 9 | 0.88 | 1.10 | 3.46 | 4.14 | 4.32 |
| 12 / 12 | 0.66 | 0.83 | 2.71 | 3.19 | 4.14 |