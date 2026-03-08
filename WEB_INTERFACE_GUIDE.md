# 🌐 VQA Web Interface - User Guide

## ✅ Server Status: **RUNNING**

The VQA web interface is now **live and running**!

---

## 🔗 Access the Interface

### Local Access:
```
http://localhost:7860
```

### Network Access (from other devices on same network):
```
http://<your-ip-address>:7860
```

To find your IP address:
```bash
hostname -I | awk '{print $1}'
```

---

## 📸 How to Use

### 1. **Upload an Image**
   - Click on "📸 Upload Image" area
   - Select an image from your computer
   - Supported formats: JPG, PNG, WebP, etc.

### 2. **Ask a Question**
   - Type your question in the "❓ Ask a Question" box
   - Questions can be about:
     - **Objects**: "What is in the image?"
     - **Colors**: "What color is the car?"
     - **Actions**: "What is the person doing?"
     - **Locations**: "Is this indoors or outdoors?"
     - **Counting**: "How many people are in the image?"
     - **Yes/No**: "Is there a dog in the image?"

### 3. **Get Answer**
   - Click "🚀 Get Answer" button or press Enter
   - The AI will analyze the image and provide an answer
   - Answer appears in the "💡 Answer" box

### 4. **Ask More Questions**
   - Keep asking questions about the same image
   - History of all Q&A pairs is displayed below
   - Each question is timestamped

### 5. **Try Another Image**
   - Click "🔄 Clear" to start fresh
   - Upload a new image and repeat

---

## 💡 Example Questions

### What Questions:
- "What is in the image?"
- "What is the person doing?"
- "What is the weather like?"

### Where Questions:
- "Where is this photo taken?"
- "Is this indoors or outdoors?"

### Color Questions:
- "What color is the shirt?"
- "What color is the sky?"

### Counting Questions:
- "How many people are in the image?"
- "How many cars?"

### Yes/No Questions:
- "Is there a dog?"
- "Is it daytime?"
- "Is anyone smiling?"

---

## 🎯 Features

### ✅ Real-time Inference
- Instant AI-powered answers
- GPU accelerated (NVIDIA GTX 1650 Ti)
- Powered by BLIP model (384M parameters)

### ✅ Question History
- All Q&A pairs saved in session
- Timestamped for reference
- Scrollable history view

### ✅ User-Friendly Interface
- Clean, modern design
- Intuitive controls
- Responsive layout

---

## 🖼️ Tips for Best Results

### Image Quality:
- ✅ Use clear, well-lit images
- ✅ Higher resolution is better
- ✅ Avoid heavily filtered or distorted images

### Question Format:
- ✅ Be specific and clear
- ✅ One question at a time
- ✅ Use simple language

### What Works Best:
- ✅ Questions about visible objects
- ✅ Color and appearance queries
- ✅ Counting questions
- ✅ Basic scene understanding

### What May Be Challenging:
- ⚠️ Abstract or subjective questions
- ⚠️ Questions requiring world knowledge
- ⚠️ Very small or unclear objects

---

## 🔧 Technical Details

### Model Information:
- **Model**: Salesforce/blip-vqa-base
- **Parameters**: 384,672,572 (384M)
- **Architecture**: Vision Transformer + BERT + Decoder
- **Framework**: PyTorch + HuggingFace Transformers

### Performance:
- **Device**: CUDA (NVIDIA GeForce GTX 1650 Ti)
- **Inference Speed**: 1-2 seconds per question
- **Memory**: ~2GB GPU RAM

### Interface:
- **Framework**: Gradio
- **Port**: 7860
- **Protocol**: HTTP
- **Browser**: Any modern browser

---

## 🚀 Running the Server

### Start Server:
```bash
cd "/media/nekoshou/New Volume1/VQA"
/home/nekoshou/miniconda3/envs/vqa_env/bin/python web_inference.py
```

### Stop Server:
- Press `Ctrl+C` in the terminal
- Or kill the process:
```bash
pkill -f web_inference.py
```

### Check Server Status:
```bash
netstat -tuln | grep 7860
# or
ss -tuln | grep 7860
```

---

## 📊 Example Session

```
Image: Photo of a woman with a dog on beach

Q1 [14:30:15]: What is in the image?
A1: woman and dog

Q2 [14:30:22]: What is the woman doing?
A2: petting dog

Q3 [14:30:30]: Is this indoors or outdoors?
A3: outdoors

Q4 [14:30:45]: What color is the dog?
A4: brown
```

---

## 🛠️ Troubleshooting

### Interface not loading?
1. Check server is running: `netstat -tuln | grep 7860`
2. Try accessing: `http://localhost:7860`
3. Clear browser cache
4. Try different browser

### Slow responses?
- First inference is slower (model loading)
- Subsequent inferences are faster
- Large images take longer to process

### Errors in answers?
- Model provides best-effort answers
- Accuracy depends on image quality
- Complex questions may have varied results

---

## 🔒 Security Notes

- Server is accessible on local network (0.0.0.0:7860)
- No authentication required (demo mode)
- For production use, add authentication
- Don't expose to public internet without security

---

## 📝 Keyboard Shortcuts

- **Enter**: Submit question (when in question box)
- **Ctrl+C**: Stop server (in terminal)

---

## ✨ Advanced Features

### Share with Others (Optional):
To create a public link that works from anywhere:

1. Edit `web_inference.py`
2. Change `share=False` to `share=True`
3. Restart server
4. Get public URL from output

**Note**: Public links are temporary (72 hours)

---

## 📞 Support

If you encounter issues:
1. Check this guide
2. Review terminal output for errors
3. Ensure GPU drivers are up to date
4. Verify all dependencies installed

---

## 🎉 Enjoy!

Your VQA web interface is ready to use. Upload images, ask questions, and explore the capabilities of visual AI!

**Current Status**: ✅ **RUNNING ON PORT 7860**

Access now at: **http://localhost:7860**

---

*Last Updated: March 8, 2026*
*Server Status: Active*
*Model: Loaded and Ready*
