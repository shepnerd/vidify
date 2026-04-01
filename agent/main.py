import click
from agent.extensions.skills.video_io import load_video
from agent.core.orchestrator import run
from agent.config import load_config, get_default_config
import logging

logging.basicConfig(level=logging.INFO)

@click.group()
@click.option('--config', default='config.yaml', help='Path to config file')
@click.pass_context
def cli(ctx, config):
    ctx.ensure_object(dict)
    ctx.obj['config'] = {**get_default_config(), **load_config(config)}

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

    with click.progressbar(length=100, label='Processing') as bar:
        asset = load_video(source_type, uri, cache_root)
        bar.update(20)
        result = run(asset, mode, cfg)
        bar.update(80)
        click.echo(result)
        bar.update(100)

if __name__ == "__main__":
    cli()