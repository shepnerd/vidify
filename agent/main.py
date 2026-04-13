import click
import sys
from agent.extensions.skills.video_io import load_video
from agent.core.orchestrator import run, normalize_mode
from agent.config import load_config, get_default_config
from agent.extensions.models.vllm_openai_client import resolve_model_name
from agent.core.events import event_bus, EventType
from agent.core.logging_config import setup_logging
from agent.core.cli_progress import CLIProgressDisplay
import logging

logging.basicConfig(level=logging.INFO)


def _plain_event_handler(event):
    """Fallback plain-text handler for non-TTY or JSON log mode."""
    icon = {
        EventType.SKILL_START: ">>>",
        EventType.SKILL_COMPLETE: "[ok]",
        EventType.SKILL_ERROR: "[!!]",
        EventType.SKILL_SKIPPED: "[--]",
        EventType.PROGRESS: "...",
        EventType.WORKFLOW_START: "===",
        EventType.WORKFLOW_COMPLETE: "===",
    }.get(event.type, "   ")
    pct = f"[{event.progress_pct:5.1f}%]" if event.progress_pct else ""
    click.echo(f"  {icon} {pct} {event.message}", err=True)

@click.group()
@click.option('--config', default='config.yaml', help='Path to config file')
@click.option('--log-format', default='text', type=click.Choice(['text', 'json']),
              help='Log format: text (human-readable) or json (structured)')
@click.pass_context
def cli(ctx, config, log_format):
    ctx.ensure_object(dict)
    ctx.obj['config'] = {**get_default_config(), **load_config(config)}
    ctx.obj['log_format'] = log_format
    setup_logging(log_format=log_format)

@cli.command()
@click.argument('source_type', type=click.Choice(['youtube', 'url', 'local']))
@click.argument('uri')
@click.option('--mode', default='detailed', type=click.Choice(['brief', 'quick', 'detailed', 'highlights', 'index', 'ask', 'report', 'live']))
@click.option('--cache-root', default='./cache')
@click.option('--question', default=None)
@click.option('--max-frames', type=int, default=128)
@click.option('--whisper-model', default=None, help='Whisper model size override')
@click.option('--fps', type=float, default=None,
              help='Frame sampling rate (frames per second). Overrides scene-based sampling.')
@click.option('--force-visual', is_flag=True, help='Force visual processing even when transcript is sufficient')
@click.option('--include-web-search', is_flag=True, help='Enhance results with web search')
@click.option('--google-api-key', default=None, help='Google Custom Search API key')
@click.option('--google-search-engine-id', default=None, help='Google Custom Search Engine ID')
@click.option('--direct-model', is_flag=True, help='Load the model in-process via transformers')
@click.option('--model-path', default=None, help='Model path or HuggingFace model ID for direct mode')
@click.option('--tokenizer-path', default=None, help='Tokenizer path override for direct mode')
@click.option('--stream-source', default='webcam', type=click.Choice(['webcam', 'stream']),
              help='Source for live mode')
@click.option('--stream-url', default=None, help='RTMP/HTTP URL for live stream mode')
@click.option('--interactive', is_flag=True, help='Interactive mode')
@click.pass_context
def analyze(ctx, source_type, uri, mode, cache_root, question, max_frames, whisper_model, fps,
            force_visual, include_web_search, google_api_key, google_search_engine_id,
            direct_model, model_path, tokenizer_path, stream_source, stream_url, interactive):
    """Analyze a video."""
    mode = normalize_mode(mode)
    cfg = ctx.obj['config']
    cfg.update({
        'source_type': source_type,
        'uri': uri,
        'mode': mode,
        'cache_root': cache_root,
        'question': question,
        'max_frames': max_frames,
        'stream_source': stream_source,
        'stream_url': stream_url,
    })
    if whisper_model is not None:
        cfg['whisper_model'] = whisper_model
    if include_web_search:
        cfg['include_web_search'] = True
    if google_api_key:
        cfg['google_api_key'] = google_api_key
    if google_search_engine_id:
        cfg['google_search_engine_id'] = google_search_engine_id
    if direct_model:
        cfg['direct_model'] = True
    if model_path:
        cfg['model_path'] = model_path
    if tokenizer_path:
        cfg['tokenizer_path'] = tokenizer_path
    if fps is not None:
        cfg['frame_strategy'] = 'fps'
        cfg['frame_fps'] = fps
    if force_visual:
        cfg['force_visual'] = True
    if not cfg.get('direct_model'):
        cfg['llm_model'] = resolve_model_name(cfg.get('llm_model'), cfg.get('llm_base_url'))

    if interactive:
        # Interactive mode: prompt for inputs
        cfg['uri'] = click.prompt('Video URI', default=cfg['uri'])
        cfg['mode'] = normalize_mode(click.prompt(
            'Mode',
            type=click.Choice(['brief', 'quick', 'detailed', 'highlights', 'index', 'ask', 'report']),
            default=cfg['mode'],
        ))
        mode = cfg['mode']

    # Choose display mode: rich progress for TTY, plain text for JSON/pipe
    use_rich = (ctx.obj.get('log_format', 'text') != 'json'
                and sys.stderr.isatty())

    if use_rich:
        click.echo(f"Analyzing {uri} (mode={mode})...", err=True)
        with CLIProgressDisplay(event_bus):
            asset = load_video(source_type, uri, cache_root)
            result = run(asset, mode, cfg)
    else:
        event_bus.subscribe(None, _plain_event_handler)
        click.echo(f"Analyzing {uri} (mode={mode})...", err=True)
        asset = load_video(source_type, uri, cache_root)
        result = run(asset, mode, cfg)

    click.echo(result)

@cli.command()
@click.argument('source_type', type=click.Choice(['youtube', 'url', 'local']))
@click.argument('uri')
@click.option('--cache-root', default='./cache')
@click.option('--chat-model', default=None,
              help='LLM model for chat responses (default: same as analysis model)')
@click.option('--chat-api-base', default=None,
              help='API base URL for chat LLM (default: same as analysis)')
@click.option('--chat-api-key', default=None,
              help='API key for chat LLM (default: EMPTY for local vLLM)')
@click.option('--vision-api-base', default=None,
              help='API base for visual analysis (default: same as chat)')
@click.option('--vision-model', default=None,
              help='Model for visual analysis (default: same as chat)')
@click.option('--direct', is_flag=True,
              help='Load model in-process via transformers (no vLLM server needed)')
@click.option('--model-path', default=None,
              help='HuggingFace model ID or local path for --direct mode')
@click.option('--dtype', default='auto', type=click.Choice(['auto', 'bfloat16', 'float16', 'float32']),
              help='Model dtype for --direct mode')
@click.pass_context
def chat(ctx, source_type, uri, cache_root, chat_model, chat_api_base,
         chat_api_key, vision_api_base, vision_model, direct, model_path, dtype):
    """Interactive video Q&A — explore a video through conversation."""
    from agent.chat import VideoChat, run_chat_repl
    from agent.extensions.skills.persist import load_analysis

    cfg = ctx.obj['config']

    # Load or download video
    asset = load_video(source_type, uri, cache_root)

    # Load cached analysis (or run one)
    try:
        analysis = load_analysis(asset.cache_dir)
        click.echo(f"Loaded cached analysis from {asset.cache_dir}", err=True)
    except Exception:
        click.echo("No cached analysis found. Running brief analysis first...", err=True)
        with CLIProgressDisplay(event_bus):
            analysis_result = run(asset, "brief", cfg)
        analysis = load_analysis(asset.cache_dir)

    if direct:
        # Pure transformers mode — no server required
        from agent.extensions.models.transformers_client import TransformersVLClient

        path = model_path or cfg.get("model_path") or "Qwen/Qwen2.5-VL-7B-Instruct"
        click.echo(f"Loading model in-process: {path} (dtype={dtype})...", err=True)
        direct_client = TransformersVLClient(path, dtype=dtype)

        session = VideoChat(
            asset=asset,
            analysis=analysis,
            direct_client=direct_client,
        )
    else:
        # Server mode — requires running vLLM
        from agent.extensions.models.vllm_openai_client import make_client
        from openai import OpenAI

        base_url = chat_api_base or cfg.get("llm_base_url", "http://localhost:8000/v1")
        model = chat_model or cfg.get("llm_model", "qwen3.5-9b")
        model = resolve_model_name(model, base_url)
        api_key = chat_api_key or "EMPTY"

        chat_client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)

        if vision_api_base:
            vis_client = make_client(vision_api_base)
        else:
            vis_client = chat_client
        vis_model = vision_model or model

        session = VideoChat(
            asset=asset,
            analysis=analysis,
            chat_client=chat_client,
            chat_model=model,
            vision_client=vis_client,
            vision_model=vis_model,
        )

    run_chat_repl(session)


if __name__ == "__main__":
    cli()
