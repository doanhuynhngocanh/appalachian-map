# Vercel Deployment - Data Files Fix

## Changes Made:

1. **Updated requirements.txt** to use flexible version ranges (>=) instead of exact versions
2. **Updated .gitignore** to only ignore the very large 30MB GeoJSON file, keeping all other data files

## Files Now Committed to Git:

✅ **INCLUDED** (needed for app to run):
- `data_clean/*.csv` - All CSV data files
- `data_clean/appalachia_snapshot.topojson` - Smaller TopoJSON file (used for map)
- `data_clean/appalachia_geom.geojson` - Geometry file
- All other necessary data files

❌ **IGNORED** (too large for GitHub):
- `data_clean/appalachia_snapshot.geojson` - 30MB file (won't be used anyway)
- `data_raw/` - Raw data folder

## Next Steps:

1. **Commit these changes** in GitHub Desktop
2. **Push to GitHub**
3. **Redeploy on Vercel**

The app should now deploy successfully with all necessary data files included!

## If Still Having Issues:

If Vercel still complains about file size, you can host the TopoJSON on a CDN and load it dynamically in the frontend code.

