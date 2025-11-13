# 🚀 Quick Start Guide

## ✅ What's Done

Your TIFF→LAS web app is ready with:
- ✅ 100MB file upload support
- ✅ Code pushed to GitHub: https://github.com/Ch3fC0d3/tifflas
- ✅ Google Vision API integration (optional OCR)
- ✅ Ready to deploy to Railway

## 📋 Next Steps

### 1. Deploy to Railway (5 minutes)

**Option A: Via Website (Easiest)**
1. Go to https://railway.app
2. Click "Start a New Project"
3. Choose "Deploy from GitHub repo"
4. Select `Ch3fC0d3/tifflas`
5. Done! Railway gives you a live URL

**Option B: Via CLI**
```bash
cd D:\Users\gabep\Desktop\tifflas
railway login
railway init
railway up
```

### 2. Set Up Google Vision API (Optional - for OCR)

Follow the detailed guide: `GOOGLE_VISION_SETUP.md`

**Quick summary:**
1. Create Google Cloud project
2. Enable Vision API
3. Create service account
4. Download JSON key
5. Add to Railway as environment variable

**Railway Environment Variable:**
- Name: `GOOGLE_VISION_CREDENTIALS_JSON`
- Value: (paste entire JSON key content)

### 3. Test Your App

**Without OCR (works immediately):**
- Upload TIFF file (up to 100MB)
- Manually enter depth and curve values
- Download LAS file

**With OCR (after Google Vision setup):**
- Upload TIFF file
- App auto-detects scale numbers
- Auto-detects track boundaries
- Download LAS file

## 📁 Project Structure

```
tifflas/
├── web_app.py              # Main Flask app (100MB support)
├── templates/index.html    # Web interface
├── requirements.txt        # Python dependencies
├── railway.json           # Railway deployment config
├── GOOGLE_VISION_SETUP.md # OCR setup guide
├── RAILWAY_DEPLOY.md      # Deployment guide
└── .gitignore            # Security (excludes API keys)
```

## 🔑 Important Files

- **web_app.py**: Main application
  - Supports 100MB files
  - Loads Google Vision credentials from environment
  - Works with or without OCR

- **requirements.txt**: All dependencies
  - Flask, OpenCV, NumPy, Pandas
  - Google Cloud Vision (for OCR)
  - Gunicorn (for production)

- **.gitignore**: Security
  - Prevents committing API keys
  - Excludes sensitive files

## 💰 Cost Estimate

**Railway:**
- Free: $5 credit/month
- Typical usage: $0.50-2/month
- Covers 2-10 months of light use

**Google Vision API:**
- Free: 1,000 OCR requests/month
- After: $1.50 per 1,000 requests
- For 100 users/month: Likely FREE

## 🎯 Your URLs

**GitHub:** https://github.com/Ch3fC0d3/tifflas
**Railway:** (will be provided after deployment)

## 🆘 Troubleshooting

**App won't start:**
- Check Railway logs for errors
- Verify requirements.txt is correct

**OCR not working:**
- Check environment variable is set
- Verify JSON key is valid
- Check Google Cloud Console for API errors

**File upload fails:**
- Check file size (max 100MB)
- Verify Railway has enough resources

## 📞 Support

- Railway docs: https://docs.railway.app
- Google Vision docs: https://cloud.google.com/vision/docs
- GitHub issues: https://github.com/Ch3fC0d3/tifflas/issues

## 🎉 Ready to Deploy!

1. Deploy to Railway (5 min)
2. Test with a TIFF file
3. (Optional) Add Google Vision for OCR
4. Share your URL with users!
