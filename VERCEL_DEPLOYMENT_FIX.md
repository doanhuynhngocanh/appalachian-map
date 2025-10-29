# Vercel Deployment Fix

## Changes Made:

1. **Updated requirements.txt** - Updated to newer versions compatible with Vercel:
   - Flask 3.0.3 (was 3.0.0)
   - pandas 2.2.2 (was 2.1.4)
   - numpy 1.26.4 (was 1.24.3)
   - openai 1.54.5 (was 1.3.0)
   - python-dotenv 1.0.1
   - openpyxl 3.1.5

2. **Created runtime.txt** - Specifies Python 3.11 (more stable than 3.12)

3. **Updated vercel.json** - Added explicit Python 3.11 runtime configuration

## If Deploy Still Fails:

### Option 1: Try Minimal Requirements

Create a minimal `requirements.txt`:

```
Flask>=3.0.0
pandas>=2.0.0
numpy>=1.24.0
openai>=1.0.0
python-dotenv>=1.0.0
openpyxl>=3.1.0
```

### Option 2: Use Vercel's Environment Variables for Python Version

In Vercel Dashboard → Settings → General:
- Set Python Version to 3.11

### Option 3: Alternative - Deploy to Railway or Render

These platforms handle Flask apps more easily:
- **Railway**: https://railway.app
- **Render**: https://render.com

## Important Notes:

- Make sure large data files are NOT in the repo (add to .gitignore)
- Set OPENAI_API_KEY in Vercel environment variables
- First deploy may take 5-10 minutes due to package installation

