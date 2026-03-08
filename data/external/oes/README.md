# OES Data Directory

## Download Instructions

BLS blocks automated downloads. You must download manually:

1. Go to https://www.bls.gov/oes/tables.htm
2. Scroll to "Occupational Employment and Wage Statistics" section
3. Download national data files for years 2015-2023
4. Look for links like "National" under each year's data

### Direct Links (open in browser)

- 2023: https://www.bls.gov/oes/special-requests/oesm23nat.zip
- 2022: https://www.bls.gov/oes/special-requests/oesm22nat.zip
- 2021: https://www.bls.gov/oes/special-requests/oesm21nat.zip
- 2020: https://www.bls.gov/oes/special-requests/oesm20nat.zip
- 2019: https://www.bls.gov/oes/special-requests/oesm19nat.zip
- 2018: https://www.bls.gov/oes/special-requests/oesm18nat.zip
- 2017: https://www.bls.gov/oes/special-requests/oesm17nat.zip
- 2016: https://www.bls.gov/oes/special-requests/oesm16nat.zip
- 2015: https://www.bls.gov/oes/special-requests/oesm15nat.zip

### After Download

Extract each zip file:
```bash
cd data/external/oes
for f in *.zip; do
  dir="${f%.zip}"
  mkdir -p "$dir"
  unzip -o "$f" -d "$dir"
done
```

### Expected Files

After extraction, you should have:
```
data/external/oes/
├── oesm23nat/
│   └── national_M2023_dl.xlsx
├── oesm22nat/
│   └── national_M2022_dl.xlsx
...
```

The code will look for files matching:
- `national_M{YYYY}_dl.xlsx`
- `national_M{YYYY}_dl.xls`
