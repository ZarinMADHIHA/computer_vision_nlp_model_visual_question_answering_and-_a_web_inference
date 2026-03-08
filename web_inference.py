"""
Web-based VQA Inference Interface using Gradio
Allows users to upload images and ask questions
"""
import os
import sys
from pathlib import Path

# Set cache directories
os.environ['TRANSFORMERS_CACHE'] = '/media/nekoshou/New Volume1/VQA/.cache/transformers'
os.environ['HF_HOME'] = '/media/nekoshou/New Volume1/VQA/.cache/huggingface'
os.environ['TORCH_HOME'] = '/media/nekoshou/New Volume1/VQA/.cache/torch'

# Create cache directories
for cache_dir in [os.environ['TRANSFORMERS_CACHE'], os.environ['HF_HOME'], os.environ['TORCH_HOME']]:
    os.makedirs(cache_dir, exist_ok=True)

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

print("=" * 70)
print("VQA Web Inference Interface")
print("=" * 70)

import torch
from PIL import Image
import gradio as gr
from model_strategy import ModelFactory
from datetime import datetime

# Global variables for model
model_strategy = None
device = None

def initialize_model():
    """Initialize the VQA model"""
    global model_strategy, device

    print("\n🔧 Initializing VQA Model...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ Using device: {device}")

    # Create BLIP strategy
    model_strategy = ModelFactory.create_model("blip")
    model_strategy.load_model("Salesforce/blip-vqa-base", device)

    print(f"✓ Model loaded: {model_strategy.get_model_name()}")
    print(f"✓ Parameters: {sum(p.numel() for p in model_strategy.model.parameters()):,}")
    print("✓ Model ready for inference!")

    return model_strategy

def answer_question(image, question, history=None):
    """
    Generate answer for a question about an image

    Args:
        image: PIL Image or numpy array
        question: Question text
        history: Previous Q&A pairs (for display)

    Returns:
        answer, updated_history
    """
    global model_strategy

    if model_strategy is None:
        return "⚠️ Model not initialized. Please wait...", history

    if image is None:
        return "⚠️ Please upload an image first.", history

    if not question or question.strip() == "":
        return "⚠️ Please enter a question.", history

    try:
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Generate answer
        answer = model_strategy.generate_answer(image, question)

        # Update history
        if history is None:
            history = []

        timestamp = datetime.now().strftime("%H:%M:%S")
        history.append({
            "question": question,
            "answer": answer,
            "timestamp": timestamp
        })

        return answer, history

    except Exception as e:
        return f"❌ Error: {str(e)}", history

def format_history(history):
    """Format Q&A history for display"""
    if not history:
        return "No questions asked yet."

    formatted = "### Question & Answer History\n\n"
    for i, qa in enumerate(history, 1):
        formatted += f"**Q{i}** [{qa['timestamp']}]: {qa['question']}\n"
        formatted += f"**A{i}**: {qa['answer']}\n\n"

    return formatted

def create_interface():
    """Create Gradio interface"""

    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .question-box textarea {
        font-size: 16px;
    }
    .answer-box textarea {
        font-size: 18px;
        font-weight: bold;
        color: #2563eb;
    }
    .history-box {
        background-color: #f8fafc;
        padding: 15px;
        border-radius: 8px;
    }
    """

    with gr.Blocks(css=custom_css, title="VQA Web Interface") as interface:
        gr.Markdown("""
        # 🤖 Visual Question Answering System

        Upload an image and ask questions about it. The AI will analyze the image and provide answers.

        **Powered by**: BLIP (Bootstrapping Language-Image Pre-training) Model
        """)

        # State for storing history
        history_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                # Image input
                image_input = gr.Image(
                    label="📸 Upload Image",
                    type="pil",
                    height=400
                )

                # Image info
                gr.Markdown("""
                **Supported formats**: JPG, PNG, WebP, etc.

                **Tips**:
                - Use clear, well-lit images
                - Questions can be about objects, colors, actions, locations, etc.
                """)

            with gr.Column(scale=1):
                # Question input
                question_input = gr.Textbox(
                    label="❓ Ask a Question",
                    placeholder="e.g., What is in the image? What color is the car?",
                    lines=2,
                    elem_classes="question-box"
                )

                # Submit button
                submit_btn = gr.Button("🚀 Get Answer", variant="primary", size="lg")

                # Answer output
                answer_output = gr.Textbox(
                    label="💡 Answer",
                    lines=3,
                    interactive=False,
                    elem_classes="answer-box"
                )

                # Clear button
                clear_btn = gr.Button("🔄 Clear", variant="secondary")

        # History display
        gr.Markdown("---")
        history_display = gr.Markdown(
            value="### Question & Answer History\n\nNo questions asked yet.",
            elem_classes="history-box"
        )

        # Example questions
        gr.Markdown("""
        ### 📝 Example Questions

        Try these types of questions:
        - **What**: What is in the image? What is the person doing?
        - **Where**: Where is this photo taken? Is this indoors or outdoors?
        - **How many**: How many people are in the image? How many cars?
        - **What color**: What color is the shirt? What color is the sky?
        - **Yes/No**: Is there a dog? Is it daytime? Is anyone smiling?
        """)

        # Example images (if you have sample images)
        gr.Markdown("### 🖼️ Or Try These Examples")

        gr.Examples(
            examples=[
                ["test_image.jpg", "What color is this image?"],
            ],
            inputs=[image_input, question_input],
            label="Sample Questions"
        )

        # Event handlers
        def submit_question(image, question, history):
            answer, new_history = answer_question(image, question, history)
            history_text = format_history(new_history)
            return answer, history_text, new_history

        def clear_all():
            return None, "", "", "### Question & Answer History\n\nNo questions asked yet.", []

        submit_btn.click(
            fn=submit_question,
            inputs=[image_input, question_input, history_state],
            outputs=[answer_output, history_display, history_state]
        )

        # Also allow Enter key to submit
        question_input.submit(
            fn=submit_question,
            inputs=[image_input, question_input, history_state],
            outputs=[answer_output, history_display, history_state]
        )

        clear_btn.click(
            fn=clear_all,
            outputs=[image_input, question_input, answer_output, history_display, history_state]
        )

        # Footer
        gr.Markdown("""
        ---
        **Model**: Salesforce/blip-vqa-base | **Framework**: PyTorch + Transformers | **Interface**: Gradio

        *Note: The model provides answers based on visual understanding. Results may vary depending on image quality and question complexity.*
        """)

    return interface

def main():
    """Main function to run the web interface"""
    print("\n" + "=" * 70)
    print("Starting VQA Web Interface")
    print("=" * 70)

    # Check if gradio is installed
    try:
        import gradio as gr
        print("✓ Gradio installed")
    except ImportError:
        print("❌ Gradio not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
        import gradio as gr
        print("✓ Gradio installed successfully")

    # Initialize model
    initialize_model()

    # Create and launch interface
    print("\n🌐 Creating web interface...")
    interface = create_interface()

    print("\n" + "=" * 70)
    print("🚀 Launching Web Server")
    print("=" * 70)
    print("\nThe interface will open in your browser automatically.")
    print("If not, copy the URL shown below and paste it in your browser.")
    print("\nTo stop the server, press Ctrl+C in the terminal.")
    print("=" * 70 + "\n")

    # Launch with options
    interface.launch(
        server_name="0.0.0.0",  # Allow access from other devices on network
        server_port=7860,
        share=False,  # Set to True to create a public link
        inbrowser=True,  # Auto-open in browser
        show_error=True
    )

if __name__ == "__main__":
    main()
