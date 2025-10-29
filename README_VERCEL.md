# Deploying to Vercel

## Pre-deployment Checklist

- [x] Created `vercel.json` configuration
- [x] Created `.vercelignore` to exclude unnecessary files
- [x] Created `.gitignore` 
- [x] Simplified `requirements.txt` (removed geopandas and spatial dependencies)
- [ ] Set up environment variables in Vercel dashboard

## Deployment Steps

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo-url>
git push -u origin main
```

### 2. Deploy to Vercel

1. Go to [vercel.com](https://vercel.com)
2. Click "New Project"
3. Import your GitHub repository
4. Configure environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
5. Click "Deploy"

### 3. Environment Variables

Add these in the Vercel dashboard under Project Settings > Environment Variables:

- `OPENAI_API_KEY`: Your OpenAI API key (required)

## Important Notes

- The app uses Flask (not Next.js)
- Static files are served from `/public` and `/templates` folders
- Large data files (~425 counties) are included in the build
- OpenAI API is required for natural language processing
_images
