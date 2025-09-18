import argparse
import asyncio
import json
import os
import queue
import re
import sys
import threading
import time
import wave
from typing import AsyncGenerator, Generator, List, Optional, Tuple

import numpy as np
import requests
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI

from decoder import convert_to_audio
from shortform_replacements import post_process_text


def format_text_for_cache(text: str) -> str:
    """Format text with sentences on separate lines and empty lines between them."""
    # Split text into sentences using regex
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Filter out empty sentences and strip whitespace
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    # Join sentences with double newlines (empty line between them)
    return "\n\n".join(sentences)


# Load environment variables from .env file
load_dotenv()

# LM Studio API settings
API_URL = os.getenv("API_URL", "http://127.0.0.1:1234/v1")
HEADERS = {"Content-Type": "application/json"}

# Preprocessing LLM settings
PREPROCESSING_MODEL = os.getenv("PREPROCESSING_MODEL", "openai/gpt-oss-20b")
ENABLE_PREPROCESSING = os.getenv("ENABLE_PREPROCESSING", "true").lower() == "true"

# Model parameters
SAMPLE_RATE = 24000  # SNAC model uses 24kHz
DEFAULT_CHUNK_SIZE = 300  # Default chunk size for long texts

# Available voices based on the Orpheus-TTS repository
AVAILABLE_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
DEFAULT_VOICE = "tara"  # Best voice according to documentation

# Special token IDs for Orpheus model
START_TOKEN_ID = 128259
END_TOKEN_IDS = [128009, 128260, 128261, 128257]
CUSTOM_TOKEN_PREFIX = "<custom_token_"

PREPROCESSING_SYSTEM_PROMPT = """You are pre-processing a story meant to be read out loud.
Expand common shortforms to their full phrase for reading.
Censor profanities by replacing them with just the first letter.

You can add emotion to the speech by adding the following tags:
```xml
<giggle>
<laugh>
<chuckle>
<sigh>
<cough>
<sniffle>
<groan>
<yawn>
<gasp>
```

DO NOT change any other words, sentence structure, or the meaning of the text.
Return only the modified text without any additional commentary.

Examples (not an exhaustive list):
tbh -> to be honest
idk -> I don't know
rly -> really
ppl -> people
u -> you
SIL -> sister in law
AITA -> Am I The A
AH -> A Hole
wtf -> what the F
"""


def format_prompt(prompt: str, voice: str = DEFAULT_VOICE) -> str:
    """Format prompt for Orpheus model with voice prefix and special tokens."""
    if voice not in AVAILABLE_VOICES:
        print(
            f"Warning: Voice '{voice}' not recognized. Using '{DEFAULT_VOICE}' instead."
        )
        voice = DEFAULT_VOICE

    # Format similar to how engine_class.py does it with special tokens
    formatted_prompt = f"{voice}: {prompt}"

    # Add special token markers for the LM Studio API
    special_start = "<|audio|>"  # Using the additional_special_token from config
    special_end = "<|eot_id|>"  # Using the eos_token from config

    return f"{special_start}{formatted_prompt}{special_end}"


def generate_tokens_from_api(
    prompt: str,
    voice: str = DEFAULT_VOICE,
) -> Generator[str, None, None]:
    """Generate tokens from text using OpenAI SDK streaming."""
    formatted_prompt = format_prompt(prompt, voice)
    print(f"Generating speech for: {formatted_prompt}")

    try:
        # Initialize OpenAI client
        client = OpenAI(
            base_url=API_URL,
            api_key="not-needed",  # LM Studio doesn't require real API key
        )

        # Create streaming completion
        stream = client.completions.create(
            model="orpheus-3b-0.1-ft",
            prompt=formatted_prompt,
            stream=True,  # Enable streaming
        )

        # Process the stream
        token_counter = 0
        for chunk in stream:
            if chunk.choices[0].text:
                token_counter += 1
                yield chunk.choices[0].text

        print("Token generation complete")

    except Exception as e:
        print(f"Error: Token generation failed with error: {e}")
        return


def turn_token_into_id(token_string: str, index: int) -> Optional[int]:
    """Convert token string to numeric ID for audio processing."""
    # Strip whitespace
    token_string = token_string.strip()

    # Find the last token in the string
    last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)

    if last_token_start == -1:
        return None

    # Extract the last token
    last_token = token_string[last_token_start:]

    # Process the last token
    if last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            token_id = int(number_str) - 10 - ((index % 7) * 4096)
            return token_id
        except ValueError:
            return None
    else:
        return None


async def tokens_decoder(
    token_gen: AsyncGenerator[str, None],
) -> AsyncGenerator[bytes, None]:
    """Asynchronous token decoder that converts token stream to audio stream."""
    buffer: List[int] = []
    count = 0
    async for token_text in token_gen:
        token = turn_token_into_id(token_text, count)
        if token is not None and token > 0:
            buffer.append(token)
            count += 1

            # Convert to audio when we have enough tokens
            if count % 7 == 0 and count > 27:
                buffer_to_proc = buffer[-28:]
                audio_samples = convert_to_audio(buffer_to_proc)
                if audio_samples is not None:
                    yield audio_samples


def tokens_decoder_sync(
    syn_token_gen: Generator[str, None, None], output_file: Optional[str] = None
) -> List[bytes]:
    """Synchronous wrapper for the asynchronous token decoder."""
    audio_queue: queue.Queue[Optional[bytes]] = queue.Queue()
    audio_segments: List[bytes] = []

    # If output_file is provided, prepare WAV file
    wav_file = None
    if output_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        wav_file = wave.open(output_file, "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)

    # Convert the synchronous token generator into an async generator
    async def async_token_gen():
        for token in syn_token_gen:
            yield token

    async def async_producer():
        async for audio_chunk in tokens_decoder(async_token_gen()):
            audio_queue.put(audio_chunk)
        audio_queue.put(None)  # Sentinel to indicate completion

    def run_async():
        asyncio.run(async_producer())

    # Start the async producer in a separate thread
    thread = threading.Thread(target=run_async)
    thread.start()

    # Process audio as it becomes available
    while True:
        audio = audio_queue.get()
        if audio is None:
            break

        audio_segments.append(audio)

        # Write to WAV file if provided
        if wav_file:
            wav_file.writeframes(audio)

    # Close WAV file if opened
    if wav_file:
        wav_file.close()

    thread.join()

    # Calculate and print duration
    duration = (
        sum([len(segment) // (2 * 1) for segment in audio_segments]) / SAMPLE_RATE
    )
    print(f"Generated {len(audio_segments)} audio segments")
    print(f"Generated {duration:.2f} seconds of audio")

    return audio_segments


def stream_audio(audio_buffer: bytes) -> None:
    """Stream audio buffer to output device."""
    if len(audio_buffer) == 0:
        return

    # Convert bytes to NumPy array (16-bit PCM)
    audio_data = np.frombuffer(audio_buffer, dtype=np.int16)

    # Normalize to float in range [-1, 1] for playback
    audio_float = audio_data.astype(np.float32) / 32767.0

    # Play the audio
    sd.play(audio_float, SAMPLE_RATE)
    sd.wait()


def generate_speech_from_api(
    prompt: str,
    voice: str = DEFAULT_VOICE,
    output_file: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> List[bytes]:
    """Generate speech from text using Orpheus model via LM Studio API."""
    # Split text into chunks if it's too long
    chunks: List[str] = split_text_into_chunks(prompt, chunk_size)

    if len(chunks) == 1:
        # Single chunk - use original method
        return tokens_decoder_sync(
            generate_tokens_from_api(
                prompt=prompt,
                voice=voice,
            ),
            output_file=output_file,
        )
    else:
        # Multiple chunks - process each and merge audio
        print(f"Processing text in {len(chunks)} chunks...")
        all_audio_segments: List[bytes] = []

        for i, chunk in enumerate(chunks, 1):
            print(f"Processing chunk {i}/{len(chunks)}...")

            # Generate audio for this chunk (don't write to file yet)
            chunk_audio_segments: List[bytes] = tokens_decoder_sync(
                generate_tokens_from_api(
                    prompt=chunk,
                    voice=voice,
                ),
                output_file=None,  # Don't write individual chunks to file
            )

            all_audio_segments.extend(chunk_audio_segments)

        # Write merged audio to output file if specified
        if output_file:
            print(f"Merging audio and saving to {output_file}...")
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

            with wave.open(output_file, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(SAMPLE_RATE)

                for segment in all_audio_segments:
                    wav_file.writeframes(segment)

        # Calculate and print total duration
        total_duration = (
            sum([len(segment) // (2 * 1) for segment in all_audio_segments])
            / SAMPLE_RATE
        )
        print(f"Generated {len(all_audio_segments)} total audio segments")
        print(f"Generated {total_duration:.2f} seconds of audio")

        return all_audio_segments


def read_text_from_file(file_path: str) -> Optional[str]:
    """Read text content from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return None


def get_reddit_post_content(reddit_url: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract title and content from a Reddit post URL and cache it."""
    try:
        # Extract post ID from URL
        # Example: https://www.reddit.com/r/subreddit/comments/abc123/title/ -> abc123
        import re

        post_id_match = re.search(r"/comments/([a-zA-Z0-9]+)/", reddit_url)
        post_id = post_id_match.group(1) if post_id_match else "unknown"

        # Convert Reddit URL to JSON API endpoint
        if reddit_url.endswith("/"):
            json_url = reddit_url + ".json"
        else:
            json_url = reddit_url + "/.json"

        # Make request to Reddit's public JSON API
        headers = {"User-Agent": "orpheus-tts-local/1.0"}
        response = requests.get(json_url, headers=headers)

        if response.status_code != 200:
            print(
                f"Error: Failed to fetch Reddit post (status code: {response.status_code})"
            )
            return None, None

        data = response.json()

        # Extract post data from the JSON response
        if not data or len(data) == 0 or "data" not in data[0]:
            print("Error: Invalid Reddit API response format")
            return None, None

        post_data = data[0]["data"]["children"][0]["data"]
        title = post_data.get("title", "")
        content = post_data.get("selftext", "")

        # Combine title and content
        if content:
            full_content = f"{title} {content}"
        else:
            full_content = title

        # Create cache directory
        cache_dir = ".cache/reddit"
        os.makedirs(cache_dir, exist_ok=True)

        # Save original content to cache
        original_file = os.path.join(cache_dir, f"{post_id}_original.txt")
        try:
            with open(original_file, "w", encoding="utf-8") as f:
                f.write(f"URL: {reddit_url}\n")
                f.write(f"Title: {title}\n")
                f.write(f"Content: {content}\n")
                f.write(f"Combined: {full_content}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            print(f"Cached original Reddit content to: {original_file}")
        except Exception as e:
            print(f"Warning: Failed to cache original content: {e}")

        return full_content, post_id

    except requests.RequestException as e:
        print(f"Error fetching Reddit post: {e}")
        return None, None
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Error parsing Reddit data: {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None


def preprocess_text(
    text: str,
    skip_preprocessing: bool = False,
    preprocessing_model: str = "openai/gpt-oss-20b",
    cache_key: Optional[str] = None,
    source_type: str = "text",
) -> str:
    """
    Preprocess text to expand abbreviations and censor inappropriate words.
    Uses OpenAI SDK for cleaner response parsing.

    Args:
        text (str): The input text to preprocess
        skip_preprocessing (bool): If True, return original text without processing
        preprocessing_model (str, optional): Override model name for preprocessing
        cache_key (str, optional): Key for caching (e.g., post_id, filename)
        source_type (str): Type of source ('reddit', 'file', 'text')

    Returns:
        str: Processed text with expanded abbreviations and censored words
    """
    if skip_preprocessing or not ENABLE_PREPROCESSING:
        return text

    try:
        # Use provided values or fall back to environment variables
        model_name = preprocessing_model or PREPROCESSING_MODEL

        # Initialize OpenAI client with custom base URL for LM Studio
        client = OpenAI(
            base_url=API_URL,
            api_key="not-needed",  # LM Studio doesn't require real API key
        )

        print("Preprocessing text with LLM...")

        # Create completion using OpenAI SDK
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": PREPROCESSING_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )

        processed_text = response.choices[0].message.content
        if processed_text:
            processed_text = processed_text.strip()

        if processed_text:
            # Apply post-processing to catch any shortforms missed by the LLM
            final_processed_text = post_process_text(processed_text)

            # Save original and processed text using unified cache structure
            cache_dir = f".cache/{source_type}"
            os.makedirs(cache_dir, exist_ok=True)

            # Determine file names based on cache_key
            if cache_key:
                original_file = os.path.join(cache_dir, f"{cache_key}_original.txt")
                processed_file = os.path.join(cache_dir, f"{cache_key}_processed.txt")
            else:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                original_file = os.path.join(cache_dir, f"{timestamp}_original.txt")
                processed_file = os.path.join(cache_dir, f"{timestamp}_processed.txt")

            try:
                # Only save original if it doesn't already exist (for Reddit, it might already be saved)
                if not os.path.exists(original_file):
                    with open(original_file, "w", encoding="utf-8") as f:
                        f.write(text)
                    print(f"Original text saved to: {original_file}")

                with open(processed_file, "w", encoding="utf-8") as f:
                    f.write(format_text_for_cache(final_processed_text))
                print(f"Processed text saved to: {processed_file}")
            except Exception as e:
                print(f"Warning: Failed to save text files: {e}")

            return final_processed_text
        else:
            print(
                "Warning: Empty response from preprocessing, applying fallback hardcoded replacements"
            )
            # Apply hardcoded replacements as fallback
            fallback_processed_text = post_process_text(text)

            # Save fallback processed text to cache
            cache_dir = f".cache/{source_type}"
            os.makedirs(cache_dir, exist_ok=True)

            if cache_key:
                original_file = os.path.join(cache_dir, f"{cache_key}_original.txt")
                processed_file = os.path.join(
                    cache_dir, f"{cache_key}_processed_fallback.txt"
                )
            else:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                original_file = os.path.join(cache_dir, f"{timestamp}_original.txt")
                processed_file = os.path.join(
                    cache_dir, f"{timestamp}_processed_fallback.txt"
                )

            try:
                if not os.path.exists(original_file):
                    with open(original_file, "w", encoding="utf-8") as f:
                        f.write(text)
                    print(f"Original text saved to: {original_file}")

                with open(processed_file, "w", encoding="utf-8") as f:
                    f.write(format_text_for_cache(fallback_processed_text))
                print(f"Fallback processed text saved to: {processed_file}")
            except Exception as e:
                print(f"Warning: Failed to save fallback text files: {e}")

            return fallback_processed_text

    except Exception as e:
        print(f"Warning: Preprocessing failed with error: {e}")
        print("Applying fallback hardcoded replacements")
        # Apply hardcoded replacements as fallback
        fallback_processed_text = post_process_text(text)

        # Save fallback processed text to cache
        cache_dir = f".cache/{source_type}"
        os.makedirs(cache_dir, exist_ok=True)

        if cache_key:
            original_file = os.path.join(cache_dir, f"{cache_key}_original.txt")
            processed_file = os.path.join(
                cache_dir, f"{cache_key}_processed_fallback.txt"
            )
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            original_file = os.path.join(cache_dir, f"{timestamp}_original.txt")
            processed_file = os.path.join(
                cache_dir, f"{timestamp}_processed_fallback.txt"
            )

        try:
            if not os.path.exists(original_file):
                with open(original_file, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Original text saved to: {original_file}")

            with open(processed_file, "w", encoding="utf-8") as f:
                f.write(format_text_for_cache(fallback_processed_text))
            print(f"Fallback processed text saved to: {processed_file}")
        except Exception as e:
            print(f"Warning: Failed to save fallback text files: {e}")

        return fallback_processed_text


def split_text_into_chunks(
    text: str, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> list[str]:
    """Split long text into smaller chunks at sentence boundaries."""
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    current_chunk: str = ""

    # Split into sentences using regex
    sentences = re.split(r"(?<=[.!?])\s+", text)

    for sentence in sentences:
        # If adding this sentence would exceed chunk size, start a new chunk
        if current_chunk and len(current_chunk + " " + sentence) > chunk_size:
            if current_chunk:  # Don't add empty chunks
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Handle case where a single sentence is longer than chunk_size
    final_chunks: List[str] = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            final_chunks.append(chunk)
        else:
            # Split long sentences by words
            words = chunk.split()
            current_word_chunk: str = ""
            for word in words:
                if (
                    current_word_chunk
                    and len(current_word_chunk + " " + word) > chunk_size
                ):
                    final_chunks.append(current_word_chunk.strip())
                    current_word_chunk = word
                else:
                    if current_word_chunk:
                        current_word_chunk += " " + word
                    else:
                        current_word_chunk = word
            if current_word_chunk:
                final_chunks.append(current_word_chunk.strip())

    return final_chunks


def load_cached_text(
    input_source: str, source_type: str, cache_key: Optional[str] = None
) -> Optional[str]:
    """
    Load processed text from cache based on input source and type.

    Args:
        input_source: The original input (file path, reddit url, or text)
        source_type: Type of source ('reddit', 'file', 'text')
        cache_key: Optional explicit cache key (auto-inferred if None)

    Returns:
        Cached processed text or None if not found
    """
    if cache_key is None:
        # Auto-infer cache key based on source type
        if source_type == "reddit":
            # Extract post ID from Reddit URL
            import re
            post_id_match = re.search(r"/comments/([a-zA-Z0-9]+)/", input_source)
            if not post_id_match:
                print(f"Error: Could not extract post ID from Reddit URL: {input_source}")
                return None
            cache_key = post_id_match.group(1)
        elif source_type == "file":
            # Use filename without extension as cache key
            cache_key = os.path.splitext(os.path.basename(input_source))[0]
        else:  # source_type == "text"
            print("Error: Cannot use --audio-only with direct text input (no persistent cache key)")
            return None

    # Construct cache file path
    cache_dir = f".cache/{source_type}"
    cache_file = os.path.join(cache_dir, f"{cache_key}_processed.txt")

    # Try to load the cached text
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            cached_text = f.read().strip()
        print(f"Loaded cached processed text from: {cache_file}")
        return cached_text
    except FileNotFoundError:
        print(f"Error: Cached processed text not found: {cache_file}")
        print(f"Run with --text-only first to generate the processed text cache.")
        return None
    except Exception as e:
        print(f"Error reading cached text from {cache_file}: {e}")
        return None


def list_available_voices() -> None:
    """List all available voices with the recommended one marked."""
    print("Available voices (in order of conversational realism):")
    for voice in AVAILABLE_VOICES:
        marker = "★" if voice == DEFAULT_VOICE else " "
        print(f"{marker} {voice}")
    print(f"\nDefault voice: {DEFAULT_VOICE}")

    print("\nAvailable emotion tags:")
    print("<laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>")


def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Orpheus Text-to-Speech using LM Studio API"
    )
    parser.add_argument("--text", type=str, help="Text to convert to speech")
    parser.add_argument("--file", type=str, help="Read text from file")
    parser.add_argument("--reddit", type=str, help="Read text from Reddit post URL")
    parser.add_argument(
        "--voice",
        type=str,
        default=DEFAULT_VOICE,
        help=f"Voice to use (default: {DEFAULT_VOICE})",
    )
    parser.add_argument("--output", type=str, help="Output WAV file path")
    parser.add_argument(
        "--list-voices", action="store_true", help="List available voices"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size for processing long texts (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip text preprocessing (abbreviation expansion and censoring)",
    )
    parser.add_argument(
        "--preprocessing-model",
        type=str,
        help="Model name for preprocessing (overrides PREPROCESSING_MODEL env var)",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Run only Stage 1 (text preprocessing) and save to cache",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Run only Stage 2 (audio generation) using cached processed text",
    )

    args = parser.parse_args()

    if args.list_voices:
        list_available_voices()
        return

    # Validate flags
    if args.text_only and args.audio_only:
        print("Error: Cannot specify both --text-only and --audio-only")
        return

    if args.audio_only and args.text:
        print("Error: Cannot use --audio-only with --text (direct text input has no persistent cache)")
        return

    # Check that only one input source is provided
    input_sources = [args.text, args.file, args.reddit]
    provided_sources = [source for source in input_sources if source is not None]

    if len(provided_sources) > 1:
        print(
            "Error: Please provide only one input source (--text, --file, or --reddit)"
        )
        return

    # Get text from the specified input source
    prompt = None
    reddit_post_id = None

    if args.file:
        prompt = read_text_from_file(args.file)
        if prompt is None:
            return
    elif args.reddit:
        prompt, reddit_post_id = get_reddit_post_content(args.reddit)
        if prompt is None:
            return
    elif args.text:
        prompt = args.text
    else:
        # Fall back to existing behavior for backward compatibility
        if len(sys.argv) > 1 and sys.argv[1] not in (
            "--voice",
            "--output",
            "--file",
            "--reddit",
        ):
            prompt = " ".join([arg for arg in sys.argv[1:] if not arg.startswith("--")])
        else:
            prompt = input("Enter text to synthesize: ")
            if not prompt:
                prompt = "Hello, I am Orpheus, an AI assistant with emotional speech capabilities."

    # Handle different modes
    if args.audio_only:
        # Mode: Audio generation only from cached text
        # Determine source type for cache loading
        if args.reddit:
            source_type = "reddit"
            input_source = args.reddit
        elif args.file:
            source_type = "file"
            input_source = args.file
        else:
            print("Error: --audio-only requires --file or --reddit")
            return

        # Load cached processed text
        processed_text = load_cached_text(input_source, source_type)
        if processed_text is None:
            return

        # Default output file if none provided
        output_file = args.output
        if not output_file:
            # Create outputs directory if it doesn't exist
            os.makedirs("outputs", exist_ok=True)
            # Generate a filename based on the voice and a timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"outputs/{args.voice}_{timestamp}.wav"
            print(f"No output file specified. Saving to {output_file}")

        # Generate speech
        print("Generating audio from cached processed text...")
        start_time = time.time()
        generate_speech_from_api(
            prompt=processed_text,
            voice=args.voice,
            chunk_size=getattr(args, "chunk_size", DEFAULT_CHUNK_SIZE),
            output_file=output_file,
        )
        end_time = time.time()

        print(f"Speech generation completed in {end_time - start_time:.2f} seconds")
        print(f"Audio saved to {output_file}")

    elif args.text_only:
        # Mode: Text preprocessing only (save to cache)
        # Initialize processed_text
        processed_text = prompt

        # Preprocess text based on source type
        if not getattr(args, "skip_preprocessing", False):
            if reddit_post_id:
                # For Reddit posts, use post_id as cache key
                processed_text = preprocess_text(
                    prompt,
                    skip_preprocessing=False,
                    preprocessing_model=getattr(args, "preprocessing_model", None)
                    or PREPROCESSING_MODEL,
                    cache_key=reddit_post_id,
                    source_type="reddit",
                )
            elif args.file:
                # For files, use filename without extension as cache key
                file_base = os.path.splitext(os.path.basename(args.file))[0]
                processed_text = preprocess_text(
                    prompt,
                    skip_preprocessing=False,
                    preprocessing_model=getattr(args, "preprocessing_model", None)
                    or PREPROCESSING_MODEL,
                    cache_key=file_base,
                    source_type="file",
                )
            else:
                # For direct text input, no cache key (will use timestamp)
                processed_text = preprocess_text(
                    prompt,
                    skip_preprocessing=False,
                    preprocessing_model=getattr(args, "preprocessing_model", None)
                    or PREPROCESSING_MODEL,
                    cache_key=None,
                    source_type="text",
                )
        else:
            print("Warning: --skip-preprocessing is set, text will be saved without preprocessing")

        print("Text preprocessing completed. Processed text has been cached.")
        print("Use --audio-only to generate audio from the cached text.")

    else:
        # Mode: Both stages (default behavior)
        # Default output file if none provided
        output_file = args.output
        if not output_file:
            # Create outputs directory if it doesn't exist
            os.makedirs("outputs", exist_ok=True)
            # Generate a filename based on the voice and a timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"outputs/{args.voice}_{timestamp}.wav"
            print(f"No output file specified. Saving to {output_file}")

        # Initialize processed_text
        processed_text = prompt

        # Preprocess text based on source type
        if not getattr(args, "skip_preprocessing", False):
            if reddit_post_id:
                # For Reddit posts, use post_id as cache key
                processed_text = preprocess_text(
                    prompt,
                    skip_preprocessing=False,
                    preprocessing_model=getattr(args, "preprocessing_model", None)
                    or PREPROCESSING_MODEL,
                    cache_key=reddit_post_id,
                    source_type="reddit",
                )
            elif args.file:
                # For files, use filename without extension as cache key
                file_base = os.path.splitext(os.path.basename(args.file))[0]
                processed_text = preprocess_text(
                    prompt,
                    skip_preprocessing=False,
                    preprocessing_model=getattr(args, "preprocessing_model", None)
                    or PREPROCESSING_MODEL,
                    cache_key=file_base,
                    source_type="file",
                )
            else:
                # For direct text input, no cache key (will use timestamp)
                processed_text = preprocess_text(
                    prompt,
                    skip_preprocessing=False,
                    preprocessing_model=getattr(args, "preprocessing_model", None)
                    or PREPROCESSING_MODEL,
                    cache_key=None,
                    source_type="text",
                )

        # Generate speech
        start_time = time.time()
        generate_speech_from_api(
            prompt=processed_text,
            voice=args.voice,
            chunk_size=getattr(args, "chunk_size", DEFAULT_CHUNK_SIZE),
            output_file=output_file,
        )
        end_time = time.time()

        print(f"Speech generation completed in {end_time - start_time:.2f} seconds")
        print(f"Audio saved to {output_file}")


if __name__ == "__main__":
    main()
