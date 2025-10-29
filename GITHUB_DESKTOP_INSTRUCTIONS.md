# Instructions for GitHub Desktop

## Updated .gitignore
I've updated your `.gitignore` file to ignore large data files.

## Remove Large Files Using GitHub Desktop

### Option 1: Use Repository's Terminal (Easiest)

1. **In GitHub Desktop**: Go to `Repository` → `Open in Git Bash` (or `Open in Command Prompt`)

2. **Run these commands**:
   ```bash
   git rm -r --cached data_raw/
   git rm -r --cached data_clean/*.geojson
   git rm -r --cached data_clean/*.topojson
   git rm -r --cached data_clean/*.csv
   ```

3. **Go back to GitHub Desktop**: You should see files removed in the Changes tab

4. **Commit**: Write commit message "Remove large data files from Git tracking"

5. **Push**: Click "Push origin"

### Option 2: Manual Removal in GitHub Desktop

1. **Open GitHub Desktop**
2. In the **Changes** tab, find the large files (in `data_raw/` and `data_clean/`)
3. **Right-click** on each large file
4. Select **"Discard"**
5. **Commit** the changes
6. **Push** to GitHub

## After Pushing

The files will be removed from GitHub but will **still exist on your computer**. They're now ignored by Git thanks to the updated `.gitignore`.

## For Future Commits

The `.gitignore` now includes:
- `data_raw/` - All raw data files
- `data_clean/*.geojson` - Large GeoJSON files
- `data_clean/*.topojson` - TopoJSON files  
- `data_clean/*.csv` - All CSV files

These files will **not** be committed to GitHub anymore.

