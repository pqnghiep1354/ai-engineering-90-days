#!/usr/bin/env python3
"""
Demo API server for Environmental LLM.

Usage:
    python scripts/demo_api.py \
        --model_path models/phi2-climate-lora \
        --base_model microsoft/phi-2 \
        --port 8000
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


def create_fastapi_app(model_path: str, base_model: Optional[str] = None):
    """Create FastAPI application."""
    
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    
    from src.inference import EnvironmentalLLM
    
    # Initialize app
    app = FastAPI(
        title="Environmental LLM API",
        description="Fine-tuned LLM for environmental domain Q&A",
        version="1.0.0",
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Load model
    logger.info(f"Loading model: {model_path}")
    llm = EnvironmentalLLM(
        model_path=model_path,
        base_model=base_model,
    )
    logger.info("Model loaded successfully")
    
    # Request/Response models
    class GenerateRequest(BaseModel):
        instruction: str
        input: str = ""
        max_tokens: int = 256
        temperature: float = 0.7
        top_p: float = 0.9
    
    class GenerateResponse(BaseModel):
        response: str
        instruction: str
        input: str
    
    class ChatMessage(BaseModel):
        role: str  # user or assistant
        content: str
    
    class ChatRequest(BaseModel):
        messages: List[ChatMessage]
        max_tokens: int = 256
        temperature: float = 0.7
    
    class ChatResponse(BaseModel):
        response: str
        messages: List[ChatMessage]
    
    class HealthResponse(BaseModel):
        status: str
        model: str
    
    # Endpoints
    @app.get("/", tags=["Info"])
    async def root():
        return {
            "name": "Environmental LLM API",
            "version": "1.0.0",
            "model": model_path,
            "endpoints": ["/generate", "/chat", "/health"],
        }
    
    @app.get("/health", response_model=HealthResponse, tags=["Info"])
    async def health():
        return HealthResponse(
            status="healthy",
            model=model_path,
        )
    
    @app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
    async def generate(request: GenerateRequest):
        """Generate response for an instruction."""
        try:
            response = llm.generate(
                instruction=request.instruction,
                input_text=request.input,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
            
            return GenerateResponse(
                response=response,
                instruction=request.instruction,
                input=request.input,
            )
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/chat", response_model=ChatResponse, tags=["Chat"])
    async def chat(request: ChatRequest):
        """Chat with conversation history."""
        try:
            # Convert messages to history format
            history = []
            for i in range(0, len(request.messages) - 1, 2):
                if i + 1 < len(request.messages):
                    history.append({
                        "user": request.messages[i].content,
                        "assistant": request.messages[i + 1].content,
                    })
            
            # Get latest user message
            user_message = request.messages[-1].content if request.messages else ""
            
            response = llm.chat(
                message=user_message,
                history=history if history else None,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            
            # Add assistant response to messages
            updated_messages = request.messages + [
                ChatMessage(role="assistant", content=response)
            ]
            
            return ChatResponse(
                response=response,
                messages=updated_messages,
            )
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


def create_gradio_interface(model_path: str, base_model: Optional[str] = None):
    """Create Gradio interface."""
    try:
        import gradio as gr
    except ImportError:
        logger.error("Gradio not installed. Install with: pip install gradio")
        return None
    
    from src.inference import EnvironmentalLLM
    
    # Load model
    logger.info(f"Loading model: {model_path}")
    llm = EnvironmentalLLM(
        model_path=model_path,
        base_model=base_model,
    )
    logger.info("Model loaded successfully")
    
    def generate_response(instruction, input_text, max_tokens, temperature):
        """Generate response."""
        response = llm.generate(
            instruction=instruction,
            input_text=input_text,
            max_new_tokens=int(max_tokens),
            temperature=temperature,
        )
        return response
    
    def chat_response(message, history):
        """Chat response for Gradio chatbot."""
        # Convert history
        formatted_history = []
        for h in history:
            formatted_history.append({
                "user": h[0],
                "assistant": h[1] if h[1] else "",
            })
        
        response = llm.chat(
            message=message,
            history=formatted_history if formatted_history else None,
        )
        return response
    
    # Create interface
    with gr.Blocks(
        title="Environmental LLM",
        theme=gr.themes.Soft(),
    ) as interface:
        gr.Markdown(
            """
            # 🌍 Environmental Domain LLM
            
            A fine-tuned language model specialized in environmental, climate, and ESG topics.
            """
        )
        
        with gr.Tabs():
            # Q&A Tab
            with gr.TabItem("Q&A"):
                with gr.Row():
                    with gr.Column():
                        instruction_input = gr.Textbox(
                            label="Question/Instruction",
                            placeholder="What is climate change?",
                            lines=2,
                        )
                        context_input = gr.Textbox(
                            label="Context (optional)",
                            placeholder="Additional context...",
                            lines=2,
                        )
                        with gr.Row():
                            max_tokens = gr.Slider(
                                minimum=50,
                                maximum=500,
                                value=256,
                                label="Max Tokens",
                            )
                            temperature = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.7,
                                label="Temperature",
                            )
                        generate_btn = gr.Button("Generate", variant="primary")
                    
                    with gr.Column():
                        output = gr.Textbox(
                            label="Response",
                            lines=10,
                        )
                
                generate_btn.click(
                    fn=generate_response,
                    inputs=[instruction_input, context_input, max_tokens, temperature],
                    outputs=output,
                )
                
                # Examples
                gr.Examples(
                    examples=[
                        ["What is climate change?", ""],
                        ["Explain ESG investing.", ""],
                        ["What are the effects of sea level rise?", ""],
                        ["How does renewable energy help the environment?", ""],
                        ["Biến đổi khí hậu ảnh hưởng đến Việt Nam như thế nào?", ""],
                    ],
                    inputs=[instruction_input, context_input],
                )
            
            # Chat Tab
            with gr.TabItem("Chat"):
                chatbot = gr.Chatbot(
                    label="Environmental Expert Chat",
                    height=400,
                )
                msg = gr.Textbox(
                    label="Your Message",
                    placeholder="Ask about environmental topics...",
                )
                clear = gr.Button("Clear")
                
                msg.submit(chat_response, [msg, chatbot], [chatbot])
                msg.submit(lambda: "", None, [msg])
                clear.click(lambda: None, None, chatbot, queue=False)
        
        gr.Markdown(
            """
            ---
            **Portfolio Project #4** - Environmental Domain LLM Fine-tuning
            """
        )
    
    return interface


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Demo API server")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model",
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model name (required for LoRA)",
    )
    
    parser.add_argument(
        "--interface",
        type=str,
        choices=["fastapi", "gradio"],
        default="gradio",
        help="Interface type",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run server",
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind",
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("🤖 Environmental LLM Demo")
    print("=" * 60 + "\n")
    
    if args.interface == "fastapi":
        import uvicorn
        app = create_fastapi_app(args.model_path, args.base_model)
        logger.info(f"Starting FastAPI server on http://{args.host}:{args.port}")
        logger.info(f"Swagger docs: http://{args.host}:{args.port}/docs")
        uvicorn.run(app, host=args.host, port=args.port)
        
    elif args.interface == "gradio":
        interface = create_gradio_interface(args.model_path, args.base_model)
        if interface:
            logger.info(f"Starting Gradio interface on http://{args.host}:{args.port}")
            interface.launch(
                server_name=args.host,
                server_port=args.port,
                share=False,
            )


if __name__ == "__main__":
    main()
