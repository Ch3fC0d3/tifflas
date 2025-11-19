# Railway Deployment Setup

## Required Environment Variables

Your Railway app needs these environment variables configured:

### 1. Google Vision API (for OCR)
**Option A: JSON credentials as environment variable (recommended for Railway)**
```
GOOGLE_VISION_CREDENTIALS_JSON={"type":"service_account","project_id":"your-project",...}
```

**Option B: File path (not recommended for Railway)**
```
GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
```

### 2. AI Provider (Gemini, OpenAI, or Hugging Face)

The app will try **Google Gemini first**, then **OpenAI**, then **Hugging Face** (if their credentials are set).

**Option A: Google Gemini (recommended if you have a Google account)**

```env
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL_ID=gemini-1.5-flash
```

If you don't set `GEMINI_MODEL_ID`, the app defaults to `gemini-1.5-flash`.

**Option B: OpenAI**

```env
OPENAI_API_KEY=sk-...your-key-here...
OPENAI_MODEL_ID=gpt-3.5-turbo
```

You can also use `gpt-4o-mini` or other chat-capable models if your account allows it.

**Option C: Hugging Face Inference**

```env
HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
HF_MODEL_ID=meta-llama/Llama-3.2-3B-Instruct
```

Hugging Face support depends on your account and which models are available via the `hf-inference` provider.

## How to Set Environment Variables in Railway

1. Go to your Railway project dashboard
2. Click on your service (tifflas-production)
3. Go to **Variables** tab
4. Add each variable:
   - Click **+ New Variable**
   - Enter variable name and value
   - Click **Add**

## Getting the Credentials

### Google Vision API Key
1. Go to https://console.cloud.google.com
2. Create/select a project
3. Enable "Cloud Vision API"
4. Go to **APIs & Services** → **Credentials**
5. Create a **Service Account Key**
6. Download the JSON file
7. Copy the **entire JSON content** (minified, no line breaks)
8. Paste into `GOOGLE_VISION_CREDENTIALS_JSON` variable in Railway

### Hugging Face API Token
1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access is enough)
3. Copy the token (starts with `hf_`)
4. Paste into `HF_API_TOKEN` variable in Railway
5. Set `HF_MODEL_ID` to `meta-llama/Llama-3.2-3B-Instruct` (or another model)

## Testing After Setup

After setting environment variables:
1. Railway will auto-redeploy
2. Test OCR: Upload a TIFF and check if "Jump to label" works
3. Test AI: Click "Generate Curve Fields" and check if AI insights appear
4. Test AI Chat: Type a question in the AI chat box

## Current Status

❌ **OCR Search** - Not working (needs Google Vision credentials)
❌ **AI Chat** - Failing with 500 error (needs HF credentials)
✅ **Core Digitization** - Working (curve detection, LAS generation)

## Troubleshooting

### Check if variables are set
The app already includes a `/debug-env` route. It looks like this in `web_app.py`:
```python
@app.route('/debug-env')
def debug_env():
    """Debug endpoint to check environment variable configuration."""
    return jsonify({
        'HF_API_TOKEN': 'set' if HF_API_TOKEN else 'missing',
        'HF_MODEL_ID': HF_MODEL_ID or 'missing',
        'OPENAI_API_KEY': 'set' if OPENAI_API_KEY else 'missing',
        'OPENAI_MODEL_ID': OPENAI_MODEL_ID or 'missing',
        'GEMINI_API_KEY': 'set' if GEMINI_API_KEY else 'missing',
        'GEMINI_MODEL_ID': GEMINI_MODEL_ID or 'missing',
        'VISION_API_AVAILABLE': VISION_API_AVAILABLE,
        'GOOGLE_VISION_CREDENTIALS_JSON': 'set' if os.getenv('GOOGLE_VISION_CREDENTIALS_JSON') else 'missing',
        'GOOGLE_APPLICATION_CREDENTIALS': 'set' if os.getenv('GOOGLE_APPLICATION_CREDENTIALS') else 'missing'
    })
```

Then visit: `https://tifflas-production.up.railway.app/debug-env`

### Common Issues
- **OCR not working**: Google Vision credentials not set or invalid
- **AI chat 500 error**: Gemini/OpenAI/HF credentials not set or model ID invalid
- **AI chat timeout**: Model is loading (first request can take 20-30 seconds)
