# Fix Git Large Files Error

## Problem
You're getting "file too large" error when pushing to GitHub because large data files in `data_raw/` and `data_clean/` were committed to Git history.

## Solution: Remove Large Files from Git History

### Step 1: Open Git Bash (or Command Prompt with Git installed)

### Step 2: Navigate to your project folder
```bash
cd "c:\Users\annad\Documents\Appalachian Map Chatbot"
```

### Step 3: Remove files from Git cache (but keep them on disk)
```bash
git rm -r --cached data_raw/
git rm -r --cached data_clean/*.geojson
git rm -r --cached data_clean/*.topojson
```

### Step 4: Commit the removal
```bash
git add .gitignore
git commit -m "Remove large data files from Git tracking"
```

### Step 5: Force push to update remote repository
```bash
git push origin main --force
```

⚠️ **Warning**: The `--force` flag will overwrite history. Make sure you're on the correct branch!

## Alternative: Start Fresh Repository

If you haven't pushed much yet, you can:

1. Delete `.git` folder in your project
2. Run `git init`
3. Add all files (large files will be ignored by .gitignore)
4. Commit and push

## Files Now Ignored by .gitignore

- `data_raw/` - Raw data files
- `data_clean/*.geojson` - Large GeoJSON files (~30MB)
- `data_clean/*.topojson` - TopoJSON files
- `data_clean/*.csv` - CSV data files

## Note for Vercel Deployment

For Vercel deployment, you'll need the data files. Options:
1. Upload data files to a CDN/storage service
2. Use environment variables for data storage
3. Keep a minimal subset of data files in the repo

