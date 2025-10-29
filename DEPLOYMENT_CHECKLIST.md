# Deployment Checklist for Vercel

## ✅ Completed Tasks

- [x] Created `vercel.json` configuration for Flask app
- [x] Created `.vercelignore` to exclude unnecessary files
- [x] Created `.gitignore` for version control
- [x] Simplified `requirements.txt` (removed geopandas and heavy dependencies)
- [x] Cleaned up temporary files (temp_fips.txt, temp_fips_clean.txt)
- [x] Removed test file (index_export_fix.txt)

## 📋 Pre-Deployment Tasks

### 1. Environment Variables
Create a `.env` file locally (or set in Vercel dashboard):
```
OPENAI_API_KEY=your_actual_openai_api_key
```

### 2. Git Repository
```bash
# If not already initialized
git init

# Stage all files
git add .

# Commit
git commit -m "Prepare for Vercel deployment"

# Push to GitHub
git remote add origin <your-github-repo-url>
git push -u origin main
```

### 3. Deploy to Vercel

1. Go to https://vercel.com
2. Click "Add New Project"
3. Import your GitHub repository
4. Add environment variable:
   - Key: `OPENAI_API_KEY`
   - Value: Your actual OpenAI API key
5. Click "Deploy"

## ⚠️ Important Notes

- **File Size**: This project includes ~30MB of GeoJSON data files. Vercel has limits on build size, so consider:
  - Using CDN for large data files, OR
  - Converting to TopimbJSON (already done) to reduce size, OR
  - Using external database/storage
  
- **Cold Starts**: Flask apps on Vercel have cold start delays. First request may take 5-10 seconds.

- **API Limits**: OpenAI API usage will be billed to your account.

## 🧪 Testing After Deployment

1. Open your Vercel URL
2. Test map loading
3. Test county data display
4. Test chatbot queries
5. Test map export (if working locally)

## 📝 Post-Deployment

- Monitor Vercel dashboard for build logs
- Check function logs for any errors
- Monitor OpenAI API usage

