# 🔍 Google Vision API Setup Guide

## Step-by-Step Instructions

### Step 1: Create Google Cloud Project

1. **Go to Google Cloud Console**
   - Open: https://console.cloud.google.com
   - Sign in with your Google account

2. **Create a new project**
   - Click the project dropdown at the top (next to "Google Cloud")
   - Click "NEW PROJECT"
   - Project name: `tifflas-ocr` (or any name you like)
   - Click "CREATE"
   - Wait for the project to be created (takes ~30 seconds)

### Step 2: Enable Billing (Required for Vision API)

1. **Set up billing**
   - Go to: https://console.cloud.google.com/billing
   - Click "ADD BILLING ACCOUNT" or "LINK A BILLING ACCOUNT"
   - Enter your credit card info
   - **Don't worry**: Vision API has 1,000 FREE requests/month
   - You won't be charged unless you exceed the free tier

### Step 3: Enable Vision API

1. **Enable the API**
   - Go to: https://console.cloud.google.com/apis/library/vision.googleapis.com
   - Make sure your project (`tifflas-ocr`) is selected at the top
   - Click "ENABLE"
   - Wait for it to enable (~10 seconds)

### Step 4: Create Service Account

1. **Go to Service Accounts**
   - Open: https://console.cloud.google.com/iam-admin/serviceaccounts
   - Make sure your project is selected

2. **Create service account**
   - Click "CREATE SERVICE ACCOUNT"
   - Service account name: `tifflas-vision`
   - Service account ID: (auto-filled, leave as is)
   - Click "CREATE AND CONTINUE"

3. **Grant permissions**
   - Role: Select "Cloud Vision AI" → "Cloud Vision API User"
   - Click "CONTINUE"
   - Click "DONE"

### Step 5: Create and Download JSON Key

1. **Create key**
   - Find your new service account in the list
   - Click the three dots (⋮) on the right
   - Select "Manage keys"
   - Click "ADD KEY" → "Create new key"
   - Key type: **JSON** (selected by default)
   - Click "CREATE"

2. **Save the file**
   - A JSON file will download automatically
   - File name: something like `tifflas-ocr-abc123.json`
   - **IMPORTANT**: Save this file securely - it's your API credential!
   - Suggested location: `D:\Users\gabep\Desktop\tifflas\google-vision-key.json`

### Step 6: Add Key to .gitignore (Security!)

**CRITICAL**: Never commit your API key to GitHub!

The `.gitignore` file should already contain:
```
google-vision-key.json
*.json
```

If not, add it now to prevent accidentally pushing your key to GitHub.

### Step 7: Configure Your App

#### For Local Testing:

1. **Set environment variable** (Windows PowerShell):
   ```powershell
   $env:GOOGLE_APPLICATION_CREDENTIALS="D:\Users\gabep\Desktop\tifflas\google-vision-key.json"
   python web_app.py
   ```

2. **Or add to your code** (temporary, for testing):
   Edit `web_app.py` and add after imports:
   ```python
   import os
   os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'D:\Users\gabep\Desktop\tifflas\google-vision-key.json'
   ```

#### For Railway Deployment:

1. **Open your JSON key file** in a text editor
2. **Copy the entire contents** (it's one long JSON object)
3. **Go to Railway dashboard**:
   - Your project → "Variables" tab
   - Click "New Variable"
   - Name: `GOOGLE_VISION_CREDENTIALS_JSON`
   - Value: Paste the entire JSON content
   - Click "Add"

4. **Update `web_app.py`** to load from environment variable:

I'll create this code update for you next.

### Step 8: Verify It Works

1. **Test locally**:
   ```bash
   python web_app.py
   ```
   - Open http://localhost:5000
   - Upload an image
   - Check the console for "✅ Google Vision API: Available"

2. **Check for errors**:
   - If you see "⚠️ Not configured", check your environment variable
   - If you see authentication errors, verify your JSON key is correct

## 📊 Free Tier Limits

- **1,000 OCR requests/month**: FREE
- After 1,000: $1.50 per 1,000 requests
- For 100 users/month: Likely stays FREE
- Set up billing alerts to monitor usage

## 🔒 Security Best Practices

✅ **DO:**
- Keep your JSON key file secure and private
- Add `*.json` to `.gitignore`
- Use environment variables for deployment
- Rotate keys periodically (every 90 days)

❌ **DON'T:**
- Commit JSON keys to GitHub
- Share your key publicly
- Hardcode keys in your source code
- Leave unused keys active

## 🎯 Next Steps

After setup:
1. Test OCR locally with a well log image
2. Deploy to Railway with environment variable
3. Monitor usage in Google Cloud Console
4. Set up billing alerts (optional but recommended)

## 📝 Summary

You now have:
- ✅ Google Cloud project created
- ✅ Vision API enabled
- ✅ Service account with proper permissions
- ✅ JSON key downloaded
- ✅ Ready to integrate with your app

Need help with any step? Let me know which part you're on!
