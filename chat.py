"""
Chat interface, dynamically load and interact with various LLM models at runtime.
"""

import streamlit as st
import torch
import os
import gc

from load_llm_util import load_base_model_for_inference, load_fft_model_for_inference, load_lora_model_for_inference, load_qlora_model_for_inference

import warnings

warnings.filterwarnings("ignore")

# Base Model ID from Hugging Face
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# FFT Model Path (Directory containing tokenizer_config.json, model.safetensor, etc.)
# FFT_PATH = "<SET_PATH>" 
FFT_PATH = "./Qwen2.5-0.5B-Instruct_hinglish_finetune/Qwen2.5-0.5B-Instruct/full_finetune"

# LoRA Adapter Path (Directory containing adapter_config.json, etc.)
# LORA_ADAPTER_PATH = "<SET_PATH>" 
LORA_ADAPTER_PATH = "./Qwen2.5-3B-Instruct_hinglish_finetune/Qwen2.5-3B-Instruct/lora_finetune" 

# QLoRA Adapter Path (Directory containing adapter_config.json, etc.)
# QLORA_ADAPTER_PATH = "<SET_PATH>" 
QLORA_ADAPTER_PATH = "./Qwen2.5-3B-Instruct_hinglish_finetune/Qwen2.5-3B-Instruct/qlora_finetune" 


# System Prompt defining the chatbot's persona
SYSTEM_PROMPT = "You are a helpful college friend who talks only in Hinglish (a mix of Hindi and English used in urban India). Be casual, friendly, and use common Hinglish slang. Do not use formal Hindi or pure English."

# --- Model Loading Functions ---

# @st.cache_resource is crucial to load the model only once per session!
@st.cache_resource
def load_base_model(model_path):
    """Loads base model for inference."""
    st.write(f"Loading base model: {model_path}")

    try:
        model, tokenizer = load_base_model_for_inference(model_path)
    except Exception as e:
        st.error(f"Could not load model from path '{model_path}'. Error: {e}.")

    st.write("Base Model and Tokenizer ready.")
    return model, tokenizer

@st.cache_resource
def load_fft_model(model_path):
    """Loads full finetuned model for inference."""
    st.write(f"Loading FFT model: {model_path}")

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"FFT Model path not found: {model_path}")

    try:
        model, tokenizer = load_fft_model_for_inference(model_path)
    except Exception as e:
        st.error(f"Could not load model from path '{model_path}'. Error: {e}.")

    st.write("FFT Model and Tokenizer ready.")
    return model, tokenizer

@st.cache_resource
def load_lora_model(_base_model_id, _adapter_path):
    """Loads a LoRA model (base + adapters) for inference."""
    st.write(f"Loading LoRA model: Base='{_base_model_id}', Adapter='{_adapter_path}'")

    if not os.path.isdir(_adapter_path):
        raise FileNotFoundError(f"LoRA adapter path not found: {_adapter_path}")

    try:
        model, tokenizer = load_lora_model_for_inference(_base_model_id, _adapter_path)
    except Exception as e:
        st.error(f"Could not load model from path '{_adapter_path}'. Error: {e}.")

    st.write("LoRA Model and Tokenizer ready.")
    return model, tokenizer

@st.cache_resource
def load_qlora_model(_base_model_id, _adapter_path, _bnb_config):
    """Loads a QLoRA model (quantized base + adapters) for inference."""
    st.write(f"Loading QLoRA model: Base='{_base_model_id}', Adapter='{_adapter_path}'")

    if not os.path.isdir(_adapter_path):
        raise FileNotFoundError(f"QLoRA adapter path not found: {_adapter_path}")

    try:
        model, tokenizer = load_qlora_model_for_inference(_base_model_id, _adapter_path)
    except Exception as e:
        st.error(f"Could not load model from path '{_adapter_path}'. Error: {e}.")
   
    # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        st.write("Tokenizer pad_token set to eos_token.")
    model.config.pad_token_id = tokenizer.pad_token_id

    st.write("QLoRA Model and Tokenizer ready.")
    return model, tokenizer

# --- Generation Function ---

def generate_conversational_response(model, tokenizer, chat_history):
    """
    Generates a response from the model based on the chat history.
    Does not modify the input chat_history.
    """
    model.eval() # Set model to evaluation mode
    device = next(model.parameters()).device # Get device of the first parameter

    # Prepare the conversation in the format the model expects
    # The apply_chat_template function handles the specific formatting for the model.
    # Ensure the history includes the system prompt if needed by the template.

    # For Qwen2.5, the template usually expects the first message to be the system prompt.
    formatted_history = []
    if chat_history and chat_history[0].get("role") != "system":
         formatted_history.append({"role": "system", "content": SYSTEM_PROMPT})
    formatted_history.extend(chat_history)

    try:
        # Apply the chat template
        input_ids = tokenizer.apply_chat_template(
            formatted_history,
            add_generation_prompt=True, # Crucial for prompting the assistant's turn
            return_tensors="pt"
        ).to(device) # Move input IDs to the model's device

        # Create attention mask (important for models, especially with padding)
        attention_mask = torch.ones_like(input_ids)

        # Generation parameters
        generation_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,     # Max tokens to generate for the response
            temperature=0.7,       # Controls randomness (lower = more deterministic)
            top_p=0.9,             # Nucleus sampling probability
            top_k=50,              # Limits sampling to top K tokens
            do_sample=True,        # Enable sampling
            pad_token_id=tokenizer.eos_token_id # Use EOS token for padding
        )

        with torch.no_grad(): # Disable gradient calculation for inference
            outputs = model.generate(**generation_kwargs)

        # Decode only the newly generated tokens (after the input sequence)
        response_ids = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    except Exception as e:
        st.error(f"Error during generation: {e}")
        response = "Sorry yaar, kuch gadbad ho gayi generation mein. Dobara try karega?"

    return response

# --- Streamlit App ---

st.set_page_config(page_title="Hinglish Chat Buddy", page_icon="🤖", layout="wide")
st.title("Hinglish Chat Buddy")

# --- Model Loading UI ---
if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = False
    st.session_state["messages"] = [] # Initialize messages only once

if not st.session_state["model_loaded"]:
    st.subheader("Select and Load Model")
    model_choice = st.radio(
        "Choose adapter type to load:",
        ("Base", "FFT", "LoRA", "QLoRA"),
        key="model_choice_radio",
        horizontal=True
    )

    if st.button(f"Load {model_choice} Model", key="load_model_button"):
        st.info(f"Loading {model_choice} model... This might take a few minutes.")
        progress_bar = st.progress(0, text="Initializing...")
        try:
            if model_choice == "Base":
                progress_bar.progress(30, text="Loading Base model...")
                model, tokenizer = load_base_model(MODEL_ID)

            elif model_choice == "FFT":
                progress_bar.progress(30, text="Loading FFT model...")
                model, tokenizer = load_fft_model(FFT_PATH)

            elif model_choice == "LoRA":
                progress_bar.progress(30, text="Loading LoRA model...")
                model, tokenizer = load_lora_model(MODEL_ID, LORA_ADAPTER_PATH)

            else: # QLoRA
                progress_bar.progress(30, text="Loading QLoRA model...")
                model, tokenizer = load_qlora_model(MODEL_ID, QLORA_ADAPTER_PATH)

            progress_bar.progress(90, text="Finalizing setup...")
            st.session_state["model"] = model
            st.session_state["tokenizer"] = tokenizer
            st.session_state["model_loaded"] = True
            st.session_state["selected_model_type"] = model_choice # Store which model was loaded
            progress_bar.progress(100, text="Model loaded successfully!")
            st.success(f"{model_choice} model loaded successfully!")
            st.info("Reloading chat interface...")
            st.rerun() # Rerun the script to display the chat interface

        except FileNotFoundError as fnf_error:
             st.error(f"Fatal Error: {fnf_error}")
             st.error("Please ensure the adapter path in the script is correct and points to a valid directory containing the adapter files.")
             st.stop()
        except Exception as e:
            st.error(f"Fatal Error: Could not load the model or adapters. Details: {e}")
            st.error("Possible issues: Incorrect MODEL_ID, incompatible libraries (torch, transformers, peft, accelerate, bitsandbytes), insufficient GPU memory, or Hugging Face Hub connection issue.")
            st.stop() # Stop execution if loading fails
    # Keep showing the loading UI until the button is clicked and loading succeeds
    st.stop()

# --- Chat Interface (only runs if model is loaded) ---

# Add a button to clear history and optionally unload model
col1, col2 = st.sidebar.columns([3,1])
with col1:
    if st.sidebar.button("Clear Chat History", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()
with col2:
    st.sidebar.write("")

st.sidebar.markdown("---")
st.sidebar.subheader("Session Management")
if st.sidebar.button("Unload Model & End Session", key="kill_session"):
    st.sidebar.info("Unloading model and releasing resources...")

    st.cache_resource.clear()

    keys_to_delete = ["model", "tokenizer", "model_loaded", "messages", "selected_model_type"]
    # Delete objects from session state FIRST
    for key in keys_to_delete:
        if key in st.session_state:
            if key == "model" or key == "tokenizer":
                 try:
                     del st.session_state[key]
                 except Exception as del_e:
                     st.sidebar.warning(f"Could not explicitly delete state['{key}']: {del_e}")
            else:
                 del st.session_state[key]
    
    # Clear GPU memory
    try:
        gc.collect() # Run Python garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # Clear PyTorch's CUDA cache
        st.sidebar.success("Model unloaded and resources released (attempted).")
    except Exception as e:
        st.sidebar.error(f"Error during resource release: {e}")

    # Set model_loaded to False explicitly AFTER clearing everything
    st.session_state["model_loaded"] = False
    st.rerun() # Rerun to go back to the model selection screen

# Display which model is currently loaded
st.sidebar.markdown("---")
st.sidebar.write(f"**Model Loaded:** {st.session_state.get('selected_model_type', 'N/A')}")


# Initialize chat history if it doesn't exist
if not st.session_state.messages:
    # Add the initial assistant message
    st.session_state.messages.append({"role": "assistant", "content": "Hey! Kya chal raha hai?"})

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Kuch pooch bhai... (Ask something...)"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Soch raha hoon... (Thinking...)"):
            # Retrieve model and tokenizer from session state
            model = st.session_state["model"]
            tokenizer = st.session_state["tokenizer"]

            # Prepare history slice for generation (exclude system prompt if handled by template)
            # Send message history that apply_chat_template will use
            # Exclude the initial system prompt from the explicit list if apply_chat_template handles it
            history_for_generation = st.session_state.messages[:] # Create a copy

            assistant_response = generate_conversational_response(
                model,
                tokenizer,
                history_for_generation # Pass the prepared history
            )
            message_placeholder.markdown(assistant_response)

    # Add assistant response to chat history *after* generation
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})