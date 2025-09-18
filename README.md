# Orpheus-TTS-Local

A lightweight client for running [Orpheus TTS](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) locally using LM Studio API.

## Features

- 🎧 High-quality Text-to-Speech using the Orpheus TTS model
- 💻 Completely local - no cloud API keys needed
- 🔊 Multiple voice options (tara, leah, jess, leo, dan, mia, zac, zoe)
- 📝 Multiple input sources: direct text, files, or Reddit posts
- ⚙️ Configurable API endpoint via environment variables
- 💾 Save audio to WAV files

## Quick Setup

1. Install [LM Studio](https://lmstudio.ai/) 
2. Download the [Orpheus TTS model (orpheus-3b-0.1-ft-q4_k_m.gguf)](https://huggingface.co/isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF) in LM Studio
3. Load the Orpheus model in LM Studio
4. Start the local server in LM Studio (default: http://127.0.0.1:1234)
5. Install dependencies:
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
   Or using uv (recommended):
   ```
   python3 -m venv venv
   source venv/bin/activate
   uv pip install -r requirements.txt
   ```
6. Run the script:
   ```
   python gguf_orpheus.py --text "Hello, this is a test" --voice tara
   ```

## Usage

### Input Sources

You can provide text input in three ways:

**Direct text input:**
```bash
python gguf_orpheus.py --text "Your text here" --voice tara --output "output.wav"
```

**From a file:**
```bash
python gguf_orpheus.py --file "path/to/textfile.txt" --voice tara --output "output.wav"
```

**From a Reddit post:**
```bash
python gguf_orpheus.py --reddit "https://www.reddit.com/r/subreddit/comments/postid/title/" --voice tara --output "output.wav"
```

### Options

- `--text TEXT`: The text to convert to speech
- `--file FILE`: Read text from a file
- `--reddit REDDIT`: Read text from Reddit post URL
- `--voice VOICE`: The voice to use (default: tara)
- `--output OUTPUT`: Output WAV file path (default: auto-generated filename)
- `--list-voices`: Show available voices
- `--temperature TEMPERATURE`: Temperature for generation (default: 0.6)
- `--top_p TOP_P`: Top-p sampling parameter (default: 0.9)
- `--repetition_penalty REPETITION_PENALTY`: Repetition penalty (default: 1.1)

**Note:** You can only use one input source at a time (--text, --file, or --reddit).

## Configuration

By default, the tool connects to LM Studio at `http://127.0.0.1:1234`. To use a different API endpoint, create a `.env` file in the project directory:

```bash
# .env
API_URL=http://192.168.1.99:1234/v1/completions
```

The tool will automatically load this configuration on startup.

## Available Voices

- tara - Best overall voice for general use (default)
- leah
- jess
- leo
- dan
- mia
- zac
- zoe

## Emotion
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

## License

Apache 2.0

