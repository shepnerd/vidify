import click
import sys
from agent.extensions.skills.video_io import load_video
from agent.core.orchestrator import run
from agent.config import load_config, get_default_config
from agent.core.events import event_bus, EventType
from agent.core.logging_config import setup_logging
import logging

logging.basicConfig(level=logging.INFO)


def _cli_event_handler(event):
    """Print real-time progress events to stderr for CLI users."""
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
    setup_logging(log_format=log_format)

@cli.command()
@click.argument('source_type', type=click.Choice(['youtube', 'url', 'local']))
@click.argument('uri')
@click.option('--mode', default='detailed', type=click.Choice(['quick', 'detailed', 'highlights', 'index', 'ask', 'report', 'live']))
@click.option('--cache-root', default='./cache')
@click.option('--question', default=None)
@click.option('--max-frames', type=int, default=128)
@click.option('--stream-source', default='webcam', type=click.Choice(['webcam', 'stream']),
              help='Source for live mode')
@click.option('--stream-url', default=None, help='RTMP/HTTP URL for live stream mode')
@click.option('--interactive', is_flag=True, help='Interactive mode')
@click.pass_context
def analyze(ctx, source_type, uri, mode, cache_root, question, max_frames, stream_source, stream_url, interactive):
    """Analyze a video."""
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

    if interactive:
        # Interactive mode: prompt for inputs
        cfg['uri'] = click.prompt('Video URI', default=cfg['uri'])
        cfg['mode'] = click.prompt('Mode', type=click.Choice(['quick', 'detailed', 'highlights', 'index', 'ask', 'report']), default=cfg['mode'])

    # Subscribe to pipeline events for real-time CLI progress
    event_bus.subscribe(None, _cli_event_handler)

    click.echo(f"Analyzing {uri} (mode={mode})...", err=True)
    asset = load_video(source_type, uri, cache_root)
    result = run(asset, mode, cfg)
    click.echo(result)

if __name__ == "__main__":
    cli()